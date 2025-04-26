#include "sageattn_utils.cuh"
#include <cuda_fp16.h>   // for __half and __half2 (float16) intrinsics
#include <cuda_bf16.h>   // for __nv_bfloat16 and __nv_bfloat162 (bfloat16)
#include <cub/cub.cuh>   // for CUB utilities (optional, e.g., for reductions)

#define WARP_SIZE 32

// grid: (num_head, batch_size)
// block: (head_dim)
template<typename T, uint32_t NUM_HTREADS, uint32_t CHUNK_SIZE, uint32_t ITEMS_PER_THREAD, uint32_t HEAD_DIM>
__global__ void NaiveSegmentMeanKernel(T* __restrict__ input,
                                        T* __restrict__ output,
                                        uint32_t* __restrict__ cu_seqlens,
                                        int stride_i_seqlen,      // num_head x head_dim
                                        int stride_i_h,           //            head_dim
                                        int stride_o_seqlen,      // num_head x head_dim
                                        int stride_o_h,           //            head_dim
                                        int batch_size) 
{
    const int head_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int dim_id = threadIdx.x;

    const int seqlen_this_time = cu_seqlens[batch_id + 1] - cu_seqlens[batch_id];

    T sum = T(0);
    for (int i = 0; i < seqlen_this_time; i ++) {
        T* input_idx = input + (cu_seqlens[batch_id] + i) * stride_i_seqlen + head_id * stride_i_h + dim_id;
            
        sum += *input_idx;
    }
    T mean = sum / T(seqlen_this_time);

    // write back
    T* output_idx = output + batch_id * stride_o_seqlen + head_id * stride_o_h + dim_id;
    *output_idx = mean;

}

// CUDA kernel to compute segment-wise mean for input of type T (float16 or bfloat16).
// `input` :       [total_seqlen, num_head, head_dim] (row-major layout).
// `output_accum`: [batch_size,   num_head, head_dim] float buffer for accumulating means.
// `cu_seqlens`: prefix-sum array of sequence lengths (batch_size+1).

// each block will process one chunk of 64 head_dim, and 4 num_head.
template<typename T, uint32_t NUM_HTREADS, uint32_t BLOCK_SIZE_SEQ, uint32_t BLOCK_SIZE_HEAD, uint32_t BLOCK_SIZE_DIM, uint32_t HEAD_DIM>
__global__ void SegmentMeanKernel(T* __restrict__ input,
                                  T* __restrict__ output,
                                  uint32_t* __restrict__ cu_seqlens,
                                  int stride_i_seqlen,      // num_head x head_dim
                                  int stride_i_h,           //            head_dim
                                  int stride_o_bsz,         // num_head x head_dim
                                  int stride_o_h,           //            head_dim
                                  int batch_size) 
{
    
}

std::vector<paddle::Tensor> chunked_segment_mean_fwd(paddle::Tensor& input,         // [total_seqlen, num_head, head_dim]
                                                     paddle::Tensor& cu_seqlens,    // [batch_size + 1], prefix-sum array of sequence lengths
                                                     const int max_seqlen) 
{
    CHECK_CONTIGUOUS(input);

    CHECK_DIMS(input, 3);

    const int batch_size = cu_seqlens.shape()[0] - 1;
    const int num_head = input.shape()[1];
    const int head_dim = input.shape()[2];

    // transpose operator invoke
    paddle::Tensor output = paddle::zeros({batch_size, input.shape()[1], input.shape()[2]}, input.dtype(), paddle::GPUPlace());

    DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, {
        DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM , {
            constexpr int NUM_THREADS = 128;

            constexpr int BLOCK_SIZE_SEQ = 256;
            constexpr int BLOCK_SIZE_HEAD = 4;
            constexpr int BLOCK_SIZE_DIM = 64;

            // for [131, 24, 128], the grid is: (1, 6, 2)
            dim3 grid(batch_size, div_ceil(num_head, BLOCK_SIZE_HEAD), div_ceil(HEAD_DIM, BLOCK_SIZE_DIM));
            dim3 block(NUM_THREADS);

            SegmentMeanKernel<c_type, NUM_THREADS, BLOCK_SIZE_SEQ, BLOCK_SIZE_HEAD, BLOCK_SIZE_DIM, HEAD_DIM><<<grid, block>>>(
                reinterpret_cast<c_type*>(input.data()),                // [total_seqlen, num_head, head_dim]
                reinterpret_cast<c_type*>(output.data()),               // [batch_size, num_head, head_dim]
                reinterpret_cast<uint32_t*>(cu_seqlens.data()),
                input.strides()[0], input.strides()[1], 
                output.strides()[0], output.strides()[1],
                batch_size
            );
        });
    });

    return {output};
}

std::vector<std::vector<int64_t>> chunked_segment_mean_InferShape(
  const std::vector<int64_t> input_shape, 
  const std::vector<int64_t> cu_seqlen_shape) {
    return {{static_cast<long>(cu_seqlen_shape.size() - 1), input_shape[1], input_shape[2]}};
}

std::vector<paddle::DataType> chunked_segment_mean_InferDtype(
  const paddle::DataType A_dtype,
  const paddle::DataType B_dtype) {
  return {A_dtype};
}

PD_BUILD_OP(chunked_segment_mean)
    .Inputs({"input", "cu_seqlen"})
    .Outputs({"output"})
    .Attrs({"max_seqlen: int"})
    .SetKernelFn(PD_KERNEL(chunked_segment_mean_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(chunked_segment_mean_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(chunked_segment_mean_InferDtype));