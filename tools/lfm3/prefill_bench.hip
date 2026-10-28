#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/lfm3/hip/q8_prefill.hpp"
using namespace gufo::models::lfm3::gpu;
void Check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
struct Buffer {
  void* p{};
  explicit Buffer(std::size_t n) { Check(hipMalloc(&p, n)); }
  ~Buffer() {
    if (p)
      (void)hipFree(p);
  }
};
constexpr int N = 512;
template<int BM, int BN, int WM, int WN>
void Run(const void* w, const unsigned char* x, std::size_t stride, float* y,
         int M, int K) {
  ResidualQ8Gemm<BM, BN, 2, WM, WN>
      <<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), WM * WN * 32>>>(
          w, x, stride, y, N, M, K);
}
void Shape(int M, int K) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  std::vector<float> x(N * K);
  for (auto& v : x)
    v = dist(gen);
  std::size_t stride = gufo::hip::Q8ActTiledBytes(N, K);
  std::vector<unsigned char> quant(3 * stride);
  for (int token = 0; token < N; ++token) {
    float maximum = 0;
    for (int k = 0; k < K; ++k)
      maximum = std::max(maximum, std::abs(x[token * K + k]));
    float scale = std::ldexp(1.0F, std::max(-149, std::ilogb(maximum) - 22));
    if (maximum > scale * (126.0F * 65536))
      scale *= 2;
    for (int k = 0; k < K; ++k) {
      int integer = int(std::nearbyint(x[token * K + k] / scale));
      int low = ((integer + 128) & 255) - 128;
      int remain = (integer - low) / 256;
      int middle = ((remain + 128) & 255) - 128;
      int high = (remain - middle) / 256;
      int digits[3] = {high, middle, low};
      float scales[3] = {scale * 65536, scale * 256, scale};
      for (int pass = 0; pass < 3; ++pass) {
        auto* tile =
            quant.data() + pass * stride +
            ((token / 16) * (K / 32) + k / 32) * gufo::hip::kQ8ActTileBytes;
        reinterpret_cast<signed char*>(
            tile)[((k % 32) / 16) * 256 + (token % 16) * 16 + k % 16] =
            digits[pass];
        if (k % 32 == 0)
          std::memcpy(tile + 512 + (token % 16) * 4, &scales[pass], 4);
      }
    }
  }
  std::vector<gufo::hip::Q8_0Block> weights(std::size_t(M) * K / 32);
  for (auto& b : weights) {
    b.d = __float2half(0.001F + std::abs(dist(gen)));
    for (auto& q : b.qs)
      q = int(gen() % 255) - 127;
  }
  std::size_t weight_bytes = weights.size() * sizeof(weights[0]);
  Buffer dw(4 * weight_bytes), dx(quant.size()), dy(N * M * 4);
  for (int r = 0; r < 4; ++r)
    Check(hipMemcpy(static_cast<char*>(dw.p) + r * weight_bytes, weights.data(),
                    weight_bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(dx.p, quant.data(), quant.size(), hipMemcpyHostToDevice));
  using Fn = void (*)(const void*, const unsigned char*, std::size_t, float*,
                      int, int);
  struct Case {
    const char* name;
    Fn run;
  };
  Case cases[] = {
      {"32x32-2x2", Run<32, 32, 2, 2>},    {"64x32-4x1", Run<64, 32, 4, 1>},
      {"32x64-1x4", Run<32, 64, 1, 4>},    {"64x64-2x4", Run<64, 64, 2, 4>},
      {"64x32-4x2", Run<64, 32, 4, 2>},    {"32x64-2x4", Run<32, 64, 2, 4>},
      {"128x64-4x2", Run<128, 64, 4, 2>},  {"64x128-2x4", Run<64, 128, 2, 4>},
      {"128x128-2x4", Run<128, 128, 2, 4>}};
  for (auto c : cases) {
    c.run(dw.p, (unsigned char*)dx.p, stride, (float*)dy.p, M, K);
    Check(hipDeviceSynchronize());
    std::vector<float> y(N * M);
    Check(hipMemcpy(y.data(), dy.p, y.size() * 4, hipMemcpyDeviceToHost));
    double error = 0;
    for (int sample = 0; sample < 97; ++sample) {
      int row = (sample * 173) % M, token = (sample * 37) % N;
      double expected = 0;
      for (int k = 0; k < K; ++k) {
        auto& b = weights[row * (K / 32) + k / 32];
        expected += double(__half2float(b.d)) * b.qs[k % 32] * x[token * K + k];
      }
      error = std::max(error, std::abs(expected - y[token * M + row]));
    }
    if (error > 0.0001)
      throw std::runtime_error(std::string(c.name) + " numerical error " +
                               std::to_string(error));
    hipEvent_t start, end;
    Check(hipEventCreate(&start));
    Check(hipEventCreate(&end));
    std::vector<float> times;
    for (int rep = 0; rep < 7; ++rep) {
      Check(hipEventRecord(start));
      for (int r = 0; r < 8; ++r)
        c.run(static_cast<char*>(dw.p) + (r % 4) * weight_bytes,
              (unsigned char*)dx.p, stride, (float*)dy.p, M, K);
      Check(hipEventRecord(end));
      Check(hipEventSynchronize(end));
      float ms;
      Check(hipEventElapsedTime(&ms, start, end));
      times.push_back(ms * 1000 / 8);
    }
    std::sort(times.begin(), times.end());
    std::cout << M << "x" << K << " " << c.name << " us=" << times[3]
              << " max_error=" << error << std::endl;
    Check(hipEventDestroy(start));
    Check(hipEventDestroy(end));
  }
}
int main() {
  try {
    Shape(7680, 2048);
    Shape(2048, 7680);
    Shape(2048, 2048);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
