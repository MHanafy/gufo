#ifndef GUFO_MODELS_LFM3_HIP_Q8_PREFILL_HPP_
#define GUFO_MODELS_LFM3_HIP_Q8_PREFILL_HPP_
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
namespace gufo::models::lfm3::gpu {
// Three balanced base-256 activation digits retain approximately 23 bits.
// Stage each packed weight once; accumulate each digit independently in FP32
// at the GGUF K32 scale boundaries, then apply the power-of-two row scales.
template<int BM = 64, int BN = 64, int BK = 2, int WM = 2, int WN = 4>
__launch_bounds__(WM * WN * 32) __global__
    void ResidualQ8Gemm(const void* weight, const unsigned char* quantized,
                        std::size_t quant_stride, float* out, int tokens, int M,
                        int K) {
  using gufo::hip::int32x4_t;
  constexpr int RT = BM / 16, TT = BN / 16;
  constexpr int RM = RT / WM, RN = TT / WN;
  static_assert((WM * WN == 8 || WM * WN == 4) && RT % WM == 0 && TT % WN == 0);
  __shared__ int32x4_t sa[BK][RT][32];
  __shared__ float sw[BK][RT][16];
  __shared__ int32x4_t sb[3][BK][TT][32];

  int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  int wr = wave / WN, wt = wave % WN, sub = lane % 16, half = lane / 16;
  int row_base = blockIdx.y * BM, token_base = blockIdx.x * BN;
  int blocks = K / 32;
  float acc[3][RM][RN][8] = {};
  float scales[3][RN];
#pragma unroll
  for (int pass = 0; pass < 3; ++pass)
#pragma unroll
    for (int j = 0; j < RN; ++j) {
      int token = token_base + (wt * RN + j) * 16 + sub;
      const auto* tile =
          quantized + pass * quant_stride +
          std::size_t(token / 16) * blocks * gufo::hip::kQ8ActTileBytes;
      scales[pass][j] = token < ((tokens + 15) / 16) * 16
                            ? reinterpret_cast<const float*>(
                                  tile + gufo::hip::kQ8ActScaleOffset)[sub]
                            : 0;
    }
  for (int base = 0; base < blocks; base += BK) {
    for (int index = tid; index < BM * BK; index += WM * WN * 32) {
      int r = index / BK, kb = index % BK, global_row = row_base + r;
      int block = base + kb;
      int32x4_t lo{}, hi{};
      float scale = 0;
      if (global_row < M && block < blocks) {
        const auto& b = static_cast<const gufo::hip::Q8_0Block*>(
            weight)[std::size_t(global_row) * blocks + block];
        __builtin_memcpy(&lo, b.qs, 16);
        __builtin_memcpy(&hi, b.qs + 16, 16);
        scale = __half2float(b.d);
      }
      sa[kb][r / 16][r % 16] = lo;
      sa[kb][r / 16][16 + r % 16] = hi;
      int slot = r % 2 == 0 ? (r % 16) / 2 : 8 + (r % 16) / 2;
      sw[kb][r / 16][slot] = scale;
    }
    for (int index = tid; index < 3 * BK * TT * 32; index += WM * WN * 32) {
      int l = index % 32, tile = (index / 32) % TT;
      int kb = (index / (32 * TT)) % BK, pass = index / (32 * TT * BK);
      int token_tile = token_base / 16 + tile;
      int32x4_t values{};
      if (token_tile * 16 < tokens && base + kb < blocks) {
        const auto* src = quantized + pass * quant_stride +
                          (std::size_t(token_tile) * blocks + base + kb) *
                              gufo::hip::kQ8ActTileBytes;
        values = reinterpret_cast<const int32x4_t*>(src)[l];
      }
      sb[pass][kb][tile][l] = values;
    }
    __syncthreads();
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
#pragma unroll
      for (int i = 0; i < RM; ++i) {
        int row_tile = wr * RM + i;
        int32x4_t a0 = sa[kb][row_tile][sub], a1 = sa[kb][row_tile][16 + sub];
#pragma unroll
        for (int j = 0; j < RN; ++j) {
          int token_tile = wt * RN + j;
#pragma unroll
          for (int pass = 0; pass < 3; ++pass) {
            auto b0 = sb[pass][kb][token_tile][sub],
                 b1 = sb[pass][kb][token_tile][16 + sub];
            gufo::hip::WmmaQuantAccumulator<32> c{};
            c = gufo::hip::WmmaQuant<32>(a0, b0, c);
            c = gufo::hip::WmmaQuant<32>(a1, b1, c);

#pragma unroll
            for (int l = 0; l < 8; ++l)
              acc[pass][i][j][l] +=
                  sw[kb][row_tile][half * 8 + l] * float(c[l]);
            __builtin_amdgcn_sched_barrier(0);
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int j = 0; j < RN; ++j)
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        int row = row_base + (wr * RM + i) * 16 + 2 * l + half;
        int token = token_base + (wt * RN + j) * 16 + sub;
        if (row < M && token < tokens)
          out[std::size_t(token) * M + row] = (acc[0][i][j][l] * scales[0][j] +
                                               acc[1][i][j][l] * scales[1][j]) +
                                              acc[2][i][j][l] * scales[2][j];
      }
}
}  // namespace gufo::models::lfm3::gpu
#endif
