#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <algorithm>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/lfm3/hip/ops.hpp"
#include "src/models/lfm3/hip/q8_prefill.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"
namespace gufo::models::lfm3::gpu {
namespace {
__device__ float Value(const Tensor& w, std::size_t i) {
  if (w.type == core::GgmlType::kF32)
    return static_cast<const float*>(w.data)[i];
  if (w.type == core::GgmlType::kF16)
    return __half2float(static_cast<const __half*>(w.data)[i]);
  if (w.type == core::GgmlType::kBF16)
    return float(static_cast<const hip_bfloat16*>(w.data)[i]);
  if (w.type == core::GgmlType::kQ8_0) {
    const auto& b = static_cast<const gufo::hip::Q8_0Block*>(w.data)[i / 32];
    return __half2float(b.d) * b.qs[i % 32];
  }
  return NAN;  // The host binder rejects unsupported formats.
}
__global__ void Embedding(Tensor w, const std::uint32_t* ids, float* y, int n) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < std::size_t(n) * w.cols)
    y[i] = Value(w, std::size_t(ids[i / w.cols]) * w.cols + i % w.cols);
}
__global__ void F16Dense(Tensor w, const float* x, float* y, int n) {
  int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32,
      token = blockIdx.y;
  if (row >= int(w.rows) || token >= n)
    return;
  float sum = 0;
  for (int k = lane; k < int(w.cols); k += 32)
    sum = fmaf(Value(w, std::size_t(row) * w.cols + k),
               x[std::size_t(token) * w.cols + k], sum);
  for (int d = 16; d; d /= 2)
    sum += __shfl_down(sum, d, 32);
  if (!lane)
    y[std::size_t(token) * w.rows + row] = sum;
}
// Encode each row into three signed base-256 digits with one power-of-two
// scale, retaining about 22 bits relative to the row maximum. The constant
// row scale can be applied after the K reduction. This is qualified against
// the independent FP64 projection formula, not assumed to be lossless.
__global__ void QuantizeFixedPoint(const float* x, unsigned char* quant,
                                   std::size_t stride, int n, int K) {
  int row = blockIdx.x, tid = threadIdx.x;
  float maximum = 0;
  for (int k = tid; k < K; k += 256)
    if (row < n)
      maximum = fmaxf(maximum, fabsf(x[std::size_t(row) * K + k]));
  for (int d = 16; d; d /= 2)
    maximum = fmaxf(maximum, __shfl_down(maximum, d, 32));
  __shared__ float maxima[8], low_scale;
  if (!(tid % 32))
    maxima[tid / 32] = maximum;
  __syncthreads();
  if (!tid) {
    maximum = 0;
    for (float value : maxima)
      maximum = fmaxf(maximum, value);
    float scale =
        maximum > 0 ? ldexpf(1.0F, max(-149, ilogbf(maximum) - 22)) : 0;
    if (maximum > scale * (126.0F * 65536.0F))
      scale *= 2;
    low_scale = scale;
  }
  __syncthreads();
  for (int k = tid; k < K; k += 256) {
    float value = row < n ? x[std::size_t(row) * K + k] : 0;
    int integer = low_scale > 0 ? __float2int_rn(value / low_scale) : 0;
    int low = ((integer + 128) & 255) - 128;
    int remain = (integer - low) / 256;
    int middle = ((remain + 128) & 255) - 128;
    int high = (remain - middle) / 256;
    const int digits[3] = {high, middle, low};
    const float scales[3] = {low_scale * 65536, low_scale * 256, low_scale};
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
      auto* tile = quant + pass * stride +
                   (std::size_t(row / 16) * (K / 32) + k / 32) *
                       gufo::hip::kQ8ActTileBytes;
      reinterpret_cast<signed char*>(
          tile)[((k % 32) / 16) * 256 + (row % 16) * 16 + k % 16] =
          digits[pass];
      if (!(k % 32))
        reinterpret_cast<float*>(
            tile + gufo::hip::kQ8ActScaleOffset)[row % 16] = scales[pass];
    }
  }
}
__global__ void Convolution(const float* p, const float* w, float* history,
                            float* y, int n, int H, int L) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= H)
    return;
  // One owner per channel retains the six delayed products in registers.
  float past[31];
  for (int t = 0; t < L - 1; ++t)
    past[t] = history[ch * (L - 1) + t];
  for (int row = 0; row < n; ++row) {
    auto base = std::size_t(row) * 3 * H;
    float bx = p[base + ch] * p[base + 2 * H + ch], sum = 0;
    for (int t = 0; t < L; ++t)
      sum = fmaf(w[ch * L + t], t == L - 1 ? bx : past[t], sum);
    y[std::size_t(row) * H + ch] = sum * p[base + H + ch];
    for (int t = 1; t < L - 1; ++t)
      past[t - 1] = past[t];
    if (L > 1)
      past[L - 2] = bx;
  }
  for (int t = 0; t < L - 1; ++t)
    history[ch * (L - 1) + t] = past[t];
}
__global__ void Rotary(float* x, const float* factors, int n, int heads, int D,
                       int position, float theta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * heads * D / 2)
    return;
  int pair = i % (D / 2), head = i / (D / 2), token = head / heads;
  float angle =
      (position + token) * float(pow(double(theta), double(-2.0F * pair / D)));
  if (factors)
    angle /= factors[pair];
  float sn, cs;
  sincosf(angle, &sn, &cs);
  float a = x[head * D + pair], b = x[head * D + pair + D / 2];
  x[head * D + pair] = a * cs - b * sn;
  x[head * D + pair + D / 2] = b * cs + a * sn;
}
// One wave owns a query/head. Four independent dot chains preserve the
// original four 32-lane sums, without a block barrier for every cached token.
__global__ void Attn(const float* q, const float* k, const float* v,
                     const float* gate, const float* ck, const float* cv,
                     float* y, int heads, int kh, int D, int position,
                     int capacity, int window) {
  int h = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
  int query = position + row;
  int first = window > 0 ? max(0, query + 1 - window) : 0;
  int kvh = h / (heads / kh), stride = kh * D;
  float query_values[4], acc[4] = {};
  for (int part = 0; part < 4; ++part)
    query_values[part] =
        q[(std::size_t(row) * heads + h) * D + lane + part * 32];
  float maximum = -INFINITY, denominator = 0;
  constexpr int kLookahead = 8;
  for (int base = first; base <= query; base += kLookahead) {
    float keys[kLookahead][4], values[kLookahead][4];
#pragma unroll
    for (int j = 0; j < kLookahead; ++j) {
      int t = base + j;
      if (t <= query) {
        const float* key = t >= position
                               ? k + std::size_t(t - position) * stride
                               : ck + std::size_t(t % capacity) * stride;
        const float* value = t >= position
                                 ? v + std::size_t(t - position) * stride
                                 : cv + std::size_t(t % capacity) * stride;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          keys[j][part] = key[kvh * D + lane + part * 32];
          values[j][part] = value[kvh * D + lane + part * 32];
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kLookahead; ++j) {
      if (base + j <= query) {
        float dot[4];
#pragma unroll
        for (int part = 0; part < 4; ++part)
          dot[part] = query_values[part] * keys[j][part];
        for (int offset = 16; offset; offset /= 2)
#pragma unroll
          for (int part = 0; part < 4; ++part)
            dot[part] += __shfl_down(dot[part], offset, 32);
        float score = __shfl(
            (dot[0] + dot[1] + dot[2] + dot[3]) * rsqrtf(float(D)), 0, 32);
        float next = fmaxf(maximum, score);
        float alpha = expf(maximum - next), beta = expf(score - next);
        denominator = denominator * alpha + beta;
        maximum = next;
#pragma unroll
        for (int part = 0; part < 4; ++part)
          acc[part] = acc[part] * alpha + beta * values[j][part];
      }
    }
  }
  for (int part = 0; part < 4; ++part)
    y[(std::size_t(row) * heads + h) * D + lane + part * 32] =
        (acc[part] / denominator) / (1 + expf(-gate[row * heads + h]));
}
__global__ void Cache(const float* k, const float* v, float* ck, float* cv,
                      int n, int stride, int position, int capacity) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // Only the last capacity rows have distinct destinations.
  int kept = min(n, capacity), first = n - kept;
  if (i >= std::size_t(kept) * stride)
    return;
  auto src = std::size_t(first) * stride + i;
  auto dst = std::size_t((position + first + i / stride) % capacity) * stride +
             i % stride;
  ck[dst] = k[src];
  cv[dst] = v[src];
}
}  // namespace
void Embed(const Tensor& w, const std::uint32_t* tokens, float* out, int n,
           hipStream_t s) {
  Embedding<<<(std::size_t(n) * w.cols + 255) / 256, 256, 0, s>>>(w, tokens,
                                                                  out, n);
}
void Dense(const Tensor& w, const float* x, float* out, int n, hipStream_t s,
           DenseWorkspace workspace) {
  if (n >= 32 && w.type == core::GgmlType::kQ8_0 && workspace.quantized) {
    const auto stride = gufo::hip::QuantizedActivationBytes(n, w.cols);
    QuantizeFixedPoint<<<(n + 15) / 16 * 16, 256, 0, s>>>(
        x, static_cast<unsigned char*>(workspace.quantized), stride, n, w.cols);
    ResidualQ8Gemm<><<<dim3((n + 63) / 64, (w.rows + 63) / 64), 256, 0, s>>>(
        w.data, static_cast<const unsigned char*>(workspace.quantized), stride,
        out, n, w.rows, w.cols);
    return;
  }
  if (w.type == core::GgmlType::kF16)
    F16Dense<<<dim3((w.rows + 3) / 4, n), 128, 0, s>>>(w, x, out, n);
  else if (n == 1)
    gufo::hip::LaunchGEMV(w.data, w.type, x, out, w.rows, w.cols, s);
  else if (quant::QuantizedBlockElements(w.type))
    gufo::hip::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, out, n, w.rows,
                                          w.cols, s);
  else if (w.type == core::GgmlType::kBF16)
    gufo::hip::LaunchExactBf16GEMMFp32SmallBatch(w.data, x, out, n, w.rows,
                                                 w.cols, s);
  else
    for (int i = 0; i < n; ++i)
      gufo::hip::LaunchGEMV(w.data, w.type, x + std::size_t(i) * w.cols,
                            out + std::size_t(i) * w.rows, w.rows, w.cols, s);
}
void ShortConv(const float* p, const float* w, float* h, float* out, int n,
               int H, int L, hipStream_t s) {
  Convolution<<<(H + 127) / 128, 128, 0, s>>>(p, w, h, out, n, H, L);
}
void Rope(float* x, const float* f, int n, int h, int d, int p, float theta,
          hipStream_t s) {
  Rotary<<<(n * h * d / 2 + 255) / 256, 256, 0, s>>>(x, f, n, h, d, p, theta);
}
void Attention(const float* q, const float* k, const float* v, const float* g,
               const float* ck, const float* cv, float* out, int n, int h,
               int kh, int d, int p, int cap, int win, hipStream_t s) {
  Attn<<<dim3(h, n), 32, 0, s>>>(q, k, v, g, ck, cv, out, h, kh, d, p, cap,
                                 win);
}
void StoreKV(const float* k, const float* v, float* ck, float* cv, int n,
             int stride, int p, int cap, hipStream_t s) {
  Cache<<<(std::size_t(std::min(n, cap)) * stride + 255) / 256, 256, 0, s>>>(
      k, v, ck, cv, n, stride, p, cap);
}
}  // namespace gufo::models::lfm3::gpu
