#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

#define WARP_SIZE 32

#define S_FP8_OFFSET 8.807f
#define S_FP8_OFFSET_EXP 6680.8477f
#define S_FP8_OFFSET_EXP_INV 0.0022326917f

#define div_ceil(M, N) (((M) + (N)-1) / (N))

enum class MaskMode {
    kNone = 0,
    kCausal = 1,
};

enum class DataType {
    kHalf,
    kInt8,
    kInt4,
    kE4M3,
    kE5M2,
};

enum class QuantGranularity {
    kPerTensor = 0,
    kPerBlock = 1,
    kPerWarp = 2,
    kPerThread = 3,
};

enum class ComputeUnit {
  kTensorCore,
  kCudaCore,
};

#define FINAL_MASK 0xffffffff
#define WARP_SIZE 32

#define S_FP8_OFFSET 8.807f
#define S_FP8_OFFSET_EXP 6680.8477f
#define S_FP8_OFFSET_EXP_INV 0.0022326917f

#define div_ceil(M, N) (((M) + (N)-1) / (N))

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FP8_CAST_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define RUNTIME_ASSERT(x) __brkpt()
#else
#include <assert.h>
#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

#ifndef USHORT_TYPE
#define USHORT_TYPE
typedef unsigned short ushort;
#endif

namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;
constexpr float log2e_recp = 1.0f / log2e;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return __shfl_xor_sync(0xffffffff, x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

}  // namespace math

__device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t *dest, float *source0, float *source1)
{
#ifdef FP8_CAST_ENABLED
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

namespace wgmma{
__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template <int stride, typename T>
__device__ uint64_t make_smem_desc(T* ptr) {
    static_assert(stride == 32 || stride == 64 || stride == 128);
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)(8 * stride)) << 32;
    desc |= ((stride == 128) ? 1llu : (stride == 64) ? 2llu : 3llu) << 62;
    return desc;
}

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k16_f16f16f32(float d[][8], T* sA, T* sB) {
    uint64_t desc_a = make_smem_desc<BK*2>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        " %64,"
        " %65,"
        " %66,  %67,  %68,  %69,  %70;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k16_f16f16f32(float d[][8], T* sA, T* sB) {
    uint64_t desc_a = make_smem_desc<BK*2>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31}, "
        " %32,"
        " %33,"
        " %34,  %35,  %36,  %37,  %38;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k16_f16f16f32(float d[][8], uint32_t RA[], T* sB) {
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        "{%64,  %65,  %66,  %67}, "
        " %68,"
        " %69,  %70,  %71, %72;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k16_f16f16f32(float d[][8], uint32_t RA[], T* sB) {
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31}, "
        "{%32,  %33,  %34,  %35}, "
        " %36,"
        " %37,  %38,  %39, %40;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB)));
}

template<int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k32_f8f8f32(float d[][8], uint32_t RA[], T* sB) {
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31}, "
        "{%32,  %33,  %34,  %35}, "
        " %36,"
        " %37,"
        " %38,  %39;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)),
            "n"(1), "n"(1));
}

template<int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k32_f8f8f32(float d[][8], uint32_t RA[], T* sB) {
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        "{%64,  %65,  %66,  %67}, "
        " %68,"
        " %69,"
        " %70,  %71;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)),
            "n"(1), "n"(1));
}

template<int ScaleD, int BK, typename T>
__device__ void wgmma_m64n128k32_s8s8s32(int32_t d[][8], T* sA, T* sB) {
    uint64_t desc_a = make_smem_desc<BK>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        " %64,"
        " %65,"
        " %66;\n"
        "}\n"
        : "+r"(d[0][0]), "+r"(d[0][1]), "+r"(d[0][2]), "+r"(d[0][3]), "+r"(d[0][4]), "+r"(d[0][5]), "+r"(d[0][6]), "+r"(d[0][7]),
          "+r"(d[1][0]), "+r"(d[1][1]), "+r"(d[1][2]), "+r"(d[1][3]), "+r"(d[1][4]), "+r"(d[1][5]), "+r"(d[1][6]), "+r"(d[1][7]),
          "+r"(d[2][0]), "+r"(d[2][1]), "+r"(d[2][2]), "+r"(d[2][3]), "+r"(d[2][4]), "+r"(d[2][5]), "+r"(d[2][6]), "+r"(d[2][7]),
          "+r"(d[3][0]), "+r"(d[3][1]), "+r"(d[3][2]), "+r"(d[3][3]), "+r"(d[3][4]), "+r"(d[3][5]), "+r"(d[3][6]), "+r"(d[3][7]),
          "+r"(d[4][0]), "+r"(d[4][1]), "+r"(d[4][2]), "+r"(d[4][3]), "+r"(d[4][4]), "+r"(d[4][5]), "+r"(d[4][6]), "+r"(d[4][7]),
          "+r"(d[5][0]), "+r"(d[5][1]), "+r"(d[5][2]), "+r"(d[5][3]), "+r"(d[5][4]), "+r"(d[5][5]), "+r"(d[5][6]), "+r"(d[5][7]),
          "+r"(d[6][0]), "+r"(d[6][1]), "+r"(d[6][2]), "+r"(d[6][3]), "+r"(d[6][4]), "+r"(d[6][5]), "+r"(d[6][6]), "+r"(d[6][7]),
          "+r"(d[7][0]), "+r"(d[7][1]), "+r"(d[7][2]), "+r"(d[7][3]), "+r"(d[7][4]), "+r"(d[7][5]), "+r"(d[7][6]), "+r"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)));
}

template<int ScaleD, int BK, typename T>
__device__ void wgmma_m64n64k32_s8s8s32(int32_t d[][8], T* sA, T* sB) {
    uint64_t desc_a = make_smem_desc<BK>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34;\n"
        "}\n"
        : "+r"(d[0][0]), "+r"(d[0][1]), "+r"(d[0][2]), "+r"(d[0][3]), "+r"(d[0][4]), "+r"(d[0][5]), "+r"(d[0][6]), "+r"(d[0][7]),
          "+r"(d[1][0]), "+r"(d[1][1]), "+r"(d[1][2]), "+r"(d[1][3]), "+r"(d[1][4]), "+r"(d[1][5]), "+r"(d[1][6]), "+r"(d[1][7]),
          "+r"(d[2][0]), "+r"(d[2][1]), "+r"(d[2][2]), "+r"(d[2][3]), "+r"(d[2][4]), "+r"(d[2][5]), "+r"(d[2][6]), "+r"(d[2][7]),
          "+r"(d[3][0]), "+r"(d[3][1]), "+r"(d[3][2]), "+r"(d[3][3]), "+r"(d[3][4]), "+r"(d[3][5]), "+r"(d[3][6]), "+r"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)));
}

template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename DTypeIn, typename T>
__device__ __forceinline__ void wgmma_f16f16f32(float d[WGMMA_N/16][8], T* sA, T* sB) {
    static_assert(std::is_same<DTypeIn, half>::value);

    static_assert(WGMMA_N == 128 || WGMMA_N == 64);
    if constexpr (WGMMA_N == 128) {
        wgmma_m64n128k16_f16f16f32<ScaleD, ScaleA, ScaleB, TransA, TransB, BK>(d, sA, sB);
    }
    else if constexpr (WGMMA_N == 64) {
        wgmma_m64n64k16_f16f16f32<ScaleD, ScaleA, ScaleB, TransA, TransB, BK>(d, sA, sB);
    }
}

template<int WGMMA_N, int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_s8s8s32(int32_t d[WGMMA_N/16][8], T* sA, T* sB) {
    static_assert(WGMMA_N == 128 || WGMMA_N == 64);
    if constexpr (WGMMA_N == 128) {
        wgmma_m64n128k32_s8s8s32<ScaleD, BK>(d, sA, sB);
    }
    else if constexpr (WGMMA_N == 64) {
        wgmma_m64n64k32_s8s8s32<ScaleD, BK>(d, sA, sB);
    }
}

template<int WGMMA_N, int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_f8f8f32(float d[][8], uint32_t* RA, T* sB) {
    static_assert(WGMMA_N == 128 || WGMMA_N == 64);
    if constexpr (WGMMA_N == 128) {
        wgmma_m64n128k32_f8f8f32<ScaleD, BK>(d, RA, sB);
    }
    else if constexpr (WGMMA_N == 64) {
        wgmma_m64n64k32_f8f8f32<ScaleD, BK>(d, RA, sB);
    }
}

} // namespace wgmma

template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, bool use_half_o_scale, bool exp_offset, bool fuse_scale=false, typename DTypeSVAccum>
__device__ __forceinline__ void update_mdo(float RS[][num_tiles_k][8], DTypeSVAccum RO[][num_tiles_v][8], float m[][2], float d[][2], const float &sm_scale)
{
  static_assert(std::is_same<DTypeSVAccum, half>::value || (!use_half_o_scale));
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      // assign the smallest value possible
      float m_prev = m[fq][k];
      float m_temp = -5000000.0f;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        float m_local = max(max(RS[fq][fk][k * 2 + 0], RS[fq][fk][k * 2 + 1]),
                                max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
        m_temp = max(m_temp, m_local);
      }

      if constexpr (!fuse_scale)
      {
        if constexpr (exp_offset)
        {
          m_temp = fmaf(m_temp, sm_scale, -S_FP8_OFFSET);
        }
        else
        {
          m_temp *= sm_scale;
        }
      }
      else if constexpr (exp_offset)
      {        
        m_temp += (-S_FP8_OFFSET);        
      }

      // exchange element with the 4 threads in the row
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1)); // 0 exchange with 1, 2 exchange with 3
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2)); // 0 exchange with 2, 1 exchange with 3

      m[fq][k] = max(m[fq][k], m_temp);

      float o_scale = math::ptx_exp2(m_prev - m[fq][k]);

      // update denominator
      d[fq][k] *= o_scale;

      half2 o_scale2;
      if constexpr (use_half_o_scale)
      {  
        o_scale2 = __floats2half2_rn(o_scale, o_scale);
      }

      // update RO
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value)
        {
          RO[fq][fv][k * 2 + 0] *= o_scale;
          RO[fq][fv][k * 2 + 1] *= o_scale;
          RO[fq][fv][k * 2 + 4] *= o_scale;
          RO[fq][fv][k * 2 + 5] *= o_scale;
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          if constexpr (use_half_o_scale)
          {
            ((half2*)RO[fq][fv])[k] = __hmul2(((half2*)RO[fq][fv])[k], o_scale2);
            ((half2*)RO[fq][fv])[k + 2] = __hmul2(((half2*)RO[fq][fv])[k + 2], o_scale2);
          }
          else
          {
            RO[fq][fv][k * 2 + 0] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 0]) * o_scale);
            RO[fq][fv][k * 2 + 1] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 1]) * o_scale);
            RO[fq][fv][k * 2 + 4] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 4]) * o_scale);
            RO[fq][fv][k * 2 + 5] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 5]) * o_scale);
          }
        }
      }

      // raise RS to exponent
      float negative_m = -m[fq][k];
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        if constexpr (fuse_scale)
        {
          RS[fq][fk][k * 2 + 0] = math::ptx_exp2(RS[fq][fk][k * 2 + 0] + negative_m);
          RS[fq][fk][k * 2 + 1] = math::ptx_exp2(RS[fq][fk][k * 2 + 1] + negative_m);
          RS[fq][fk][k * 2 + 4] = math::ptx_exp2(RS[fq][fk][k * 2 + 4] + negative_m);
          RS[fq][fk][k * 2 + 5] = math::ptx_exp2(RS[fq][fk][k * 2 + 5] + negative_m);
        }
        else
        {
          RS[fq][fk][k * 2 + 0] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 0], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 1] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 1], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 4] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 4], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 5] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 5], sm_scale, negative_m));
        }
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename T>
__device__ __forceinline__ void RS_32_to_16(T RS[][num_tiles_k][8], uint32_t RS_16[][num_tiles_k][4])
{
  static_assert(sizeof(T) == 4);
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      ((half2*)RS_16[fq][fk])[0] = __float22half2_rn(((float2*)RS[fq][fk])[0]);
      ((half2*)RS_16[fq][fk])[1] = __float22half2_rn(((float2*)RS[fq][fk])[1]);
      ((half2*)RS_16[fq][fk])[2] = __float22half2_rn(((float2*)RS[fq][fk])[2]);
      ((half2*)RS_16[fq][fk])[3] = __float22half2_rn(((float2*)RS[fq][fk])[3]);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_32_to_8(float RS[][num_tiles_k][8], uint32_t RS_8[][num_tiles_k / 2][4])
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
    {
      floatx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 4);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 2, RS[fq][fk * 2 + 0] + 6);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 4);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 2, RS[fq][fk * 2 + 1] + 6);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_16_to_8(uint32_t RS[][num_tiles_k][4], uint32_t RS_8[][num_tiles_k / 2][4])
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
    {
      halfx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_8_to_16(uint32_t RS_8[][num_tiles_k / 2][4], uint32_t RS[][num_tiles_k][4])
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
    {
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2, RS_8[fq][fk]);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3, RS_8[fq][fk] + 1);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2, RS_8[fq][fk] + 2);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3, RS_8[fq][fk] + 3);
    }
  }
}

template<uint32_t num_tiles_q, uint32_t num_tiles_v,
       ComputeUnit compute_unit = ComputeUnit::kTensorCore, // compute unit for accumulate_d
       typename DTypeQKAccum, typename DTypeSVAccum>
__device__ __forceinline__ void normalize_d(DTypeSVAccum RO[][num_tiles_v][8], DTypeQKAccum m[][2], float d[][2])
{
  if constexpr (compute_unit == ComputeUnit::kCudaCore)
  { 
    // accumulate_d performs partial accumulation with cuda core
    // aggregate d
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 2; k++)
      {
        d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x1); // sum 0 and 1, 2 and 3
        d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x2); // sum 0 and 2, 1 and 3
      }
    }
  }

  // divide O by d
  float d_rcp[num_tiles_q][2];
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      // TODO: check m to prevent nan
      d_rcp[fq][k] = math::ptx_rcp(d[fq][k]);
    }
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value)
        {
          RO[fq][fv][k] *= d_rcp[fq][(k % 4) / 2];
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          RO[fq][fv][k] = __float2half_rn(__half2float(RO[fq][fv][k]) * d_rcp[fq][(k % 4) / 2]);
        }
      }
    }
  }
}



// dispatch_utils.h
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported head dim: " << int(head_dim);       \
    throw std::invalid_argument(err_msg.str());                 \
  }

// add new support to HEAD_DIM = 192 for deepseek
#define DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else if (head_dim == 192) {                                 \
    constexpr int HEAD_DIM = 192;                               \
    __VA_ARGS__                                                 \
  } else if (head_dim == 256) {                                 \
    constexpr int HEAD_DIM = 256;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported head dim: " << int(head_dim);       \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)              \
  if (is_causal == 1) {                                         \
    constexpr bool IS_CAUSAL = true;                            \
    __VA_ARGS__                                                 \
  } else if (is_causal == 0) {                                  \
    constexpr bool IS_CAUSAL = false;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported causal mode: " << int(is_causal);   \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, ...)              \
  if (qk_quant_gran == 2) {                                         \
    constexpr int QK_QUANT_GRAN = 2;                            \
    __VA_ARGS__                                                 \
  } else if (qk_quant_gran == 3) {                                  \
    constexpr int QK_QUANT_GRAN = 3;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported qk_quant_gran: " << int(qk_quant_gran);   \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, ...)             \
  if (return_lse == 1) {                                         \
    constexpr bool RETURN_LSE = true;                            \
    __VA_ARGS__                                                  \
  } else if (return_lse == 0) {                                  \
    constexpr bool RETURN_LSE = false;                           \
    __VA_ARGS__                                                  \
  }  else {                                                      \
    std::ostringstream err_msg;                                  \
    err_msg << "Unsupported causal mode: " << int(return_lse);   \
    throw std::invalid_argument(err_msg.str());                  \
  }

// DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16
// here we will use paddle's DataType
#define DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(paddle_dtype, c_type, ...)                \
  if (paddle_dtype == paddle::DataType::FLOAT16) {                                          \
    using c_type = half;                                                                \
    __VA_ARGS__                                                                         \
  } else if (paddle_dtype == paddle::DataType::BFLOAT16) {                               \
    using c_type = nv_bfloat16;                                                         \
    __VA_ARGS__                                                                         \
  } else {                                                                              \
    std::ostringstream oss;                                                             \
    oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << paddle_dtype;    \
    PD_CHECK(false, oss.str());                                                      \
  }

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...)        \
  if (block_size == 64) {                                       \
    constexpr int BLOCK_SIZE = 64;                              \
    __VA_ARGS__                                                 \
  } else if (block_size == 128) {                               \
    constexpr int BLOCK_SIZE = 128;                             \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported block_size " << int(block_size);    \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, ...)  \
  if (warp_block_size == 16) {                                           \
    constexpr int WARP_BLOCK_SIZE = 16;                                  \
    __VA_ARGS__                                                          \
  } else if (warp_block_size == 32) {                                    \
    constexpr int WARP_BLOCK_SIZE = 32;                                  \
    __VA_ARGS__                                                          \
  }  else {                                                              \
    std::ostringstream err_msg;                                          \
    err_msg << "Unsupported warp_block_size " << int(warp_block_size);   \
    throw std::invalid_argument(err_msg.str());                          \
  }

// define the macro for necessary checks, originally in `utils.cuh`
#define CHECK_CUDA(x) \
  PD_CHECK(x.is_gpu(), "Tensor " #x " must be on CUDA") // shift to paddle API: is_gpu()

// CHECK_DTYPE aims at testing the tensor datatype, use paddle::DataType
#define CHECK_DTYPE(x, true_dtype)     \
  PD_CHECK(x.dtype() == true_dtype, \
              "Tensor " #x " must have dtype (" #true_dtype ")")  // DataType dtype() const;
#define CHECK_DIMS(x, true_dim)    \
  PD_CHECK(x.dims().size() == true_dim, \
              "Tensor " #x " must have dimension number (" #true_dim ")") // paddle API: .dims().size()
#define CHECK_NUMEL(x, minimum)     \
  PD_CHECK(x.numel() >= minimum, \
              "Tensor " #x " must have at last " #minimum " elements")
#define CHECK_SHAPE(x, ...)                                   \
  PD_CHECK(x.dims() == common::DDim({__VA_ARGS__}), \
              "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  PD_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")   // TODO: check if valid
#define CHECK_LASTDIM_CONTIGUOUS(x) \
  PD_CHECK(x.strides().at(x.strides().size() - 1) == 1,    \
              "Tensor " #x " must be contiguous at the last dimension")


namespace sageattn {

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T) (0.0f);
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockAllReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (lane < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[i][lane] : (T) (0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T) 0.0f;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}
/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;  // in-warp idx
    int                 wid  = threadIdx.x >> 5;    // warp idx
    val = warpReduceMax(val);  // get maxx in each warp
    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;
    __syncthreads();
    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);
    return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockAllReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5;    // warp idx

    val = warpReduceMax(val);      // get maxx in each warp

    if (lane == 0)                 // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMin(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = min(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}
/* Calculate the minimum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMin(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;  // in-warp idx
    int                 wid  = threadIdx.x >> 5;    // warp idx
    val = warpReduceMin(val);  // get minx in each warp
    if (lane == 0)  // record in-warp minx by warp Idx
        shared[wid] = val;
    __syncthreads();
    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : 1e20f;
    val = warpReduceMin(val);
    return val;
}

} // namespace sageattn

