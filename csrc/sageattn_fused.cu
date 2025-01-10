#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "sageattn_utils.cuh"
#include "paddle/extension.h"

enum class QuantType
{
  kInt8,
  kInt4,
};

template <typename T>
__device__ __forceinline__ float convert_to_float(T val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __half2float(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __bfloat162float(val);
  }
}

template <typename T>
__device__ __forceinline__ T convert_from_float(float val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __float2half_rn(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __float2bfloat16_rn(val);
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, bool has_sm_scale = false, bool sub_mean = false, typename T>
__global__ void QuantInt8Kernel(T *__restrict__ input, T *__restrict__ mean, int8_t *__restrict__ output, float *__restrict__ scale, float sm_scale, const uint32_t num_tokens, 
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T x_val[num_pack_per_thread][8];
  // T mean_val[8];
  float x_val_float[num_pack_per_thread][8];
  float mean_val_float[8];

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;
  float *scale_ptr_base = scale + batch_id * stride_bz_scale + head_id * stride_h_scale + bx;

  if constexpr (sub_mean)
  {
    // *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);
// #pragma unroll
//     for (int ii = 0; ii < 8; ii++) {
//       mean_val[ii] = *(mean_ptr_base[ii]);
//     }

#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      mean_val_float[j] = convert_to_float(mean_ptr_base[j]);
    }
  }

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  // load the data
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      // *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (int ii = 0; ii < 8; ii++) {
        x_val[i][ii] = *(input_ptr_base + i * iter_stride * stride_seq_input + ii);
      }
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = convert_to_float(x_val[i][j]);
      }

      if constexpr (sub_mean)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] -= mean_val_float[j];
        }
      }

      if constexpr (has_sm_scale)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] *= sm_scale;
        }
      }
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = 0.0f;
      }
    }
  }

  float amax_val = 0.0000001f; // prevent from dividing by zero

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
    }
  }

  __shared__ float s_amax;
  const float block_amax_val = sageattn::blockReduceMax(amax_val);
  if (thread_id == 0)
  {
    s_amax = block_amax_val;
    scale_ptr_base[0] = s_amax / 127.0f;
  }

  __syncthreads();

  float tmp_scale = 127.0f / s_amax;

  char4 o_val[num_pack_per_thread][2];

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 2; j += 1)
    {
      o_val[i][j] = make_char4(
        float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale)
      );
    }
  }

  // int8 result
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float2*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float2*>(&o_val[i][0]);
    }
  }
}

void quant_per_block_int8_fuse_sub_mean_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& mean,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(mean);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, paddle::DataType::INT8);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.shape()[0];
  const int head_dim = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.shape()[1];
    num_heads = input.shape()[2];
    stride_seq_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_seq_output = output.strides()[1];
    stride_h_output = output.strides()[2];
  }
  else
  {
    num_tokens = input.shape()[2];
    num_heads = input.shape()[1];
    stride_seq_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_seq_output = output.strides()[2];
    stride_h_output = output.strides()[1];
  }

  auto input_dtype = input.dtype();
  auto mean_dtype = mean.dtype();

  PD_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
        CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, true, c_type><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data()),
          reinterpret_cast<c_type*>(mean.data()),
          output.data<int8_t>(),
          reinterpret_cast<float*>(scale.data()),
          0.0f,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          mean.strides()[0], mean.strides()[1],
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.strides()[0], scale.strides()[1]
        );
      });
    });
  });
}

void quant_per_warp_int8_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, paddle::DataType::INT8);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.shape()[0];
  const int head_dim = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.shape()[1];
    num_heads = input.shape()[2];
    stride_seq_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_seq_output = output.strides()[1];
    stride_h_output = output.strides()[2];
  }
  else
  {
    num_tokens = input.shape()[2];
    num_heads = input.shape()[1];
    stride_seq_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_seq_output = output.strides()[2];
    stride_h_output = output.strides()[1];
  }

  auto input_dtype = input.dtype();

  constexpr int BLOCK_SIZE = 128;
  constexpr int WARP_BLOCK_SIZE = 32;

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

      CHECK_SHAPE(output, input.shape()[0], input.shape()[1], 
                input.shape()[2], input.shape()[3]);
      CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE));

      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);

      constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

      dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

      QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<grid, block>>>(
        reinterpret_cast<c_type*>(input.data()),
        nullptr,
        output.data<int8_t>(),
        reinterpret_cast<float*>(scale.data()),
        0.0,
        num_tokens,
        stride_bz_input, stride_seq_input, stride_h_input,
        0, 0,
        stride_bz_output, stride_seq_output, stride_h_output,
        scale.strides()[0], scale.strides()[1]
      );
    });
  });
}