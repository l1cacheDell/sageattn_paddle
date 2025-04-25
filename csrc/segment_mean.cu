#include "sageattn_utils.cuh"
#include <cuda_fp16.h>   // for __half and __half2 (float16) intrinsics
#include <cuda_bf16.h>   // for __nv_bfloat16 and __nv_bfloat162 (bfloat16)
#include <cub/cub.cuh>   // for CUB utilities (optional, e.g., for reductions)

// CUDA kernel to compute segment-wise mean for input of type T (float16 or bfloat16).
// `input` :       [num_head, head_dim, total_seqlen] (row-major layout).
// `output_accum`: [batch_size, num_head,  head_dim] float buffer for accumulating means.
// `cu_seqlens`: prefix-sum array of sequence lengths (batch_size+1).
// equasion: CHUNK_SIZE = NUM_THREADS * ITEMS_PER_THREAD

#define WARP_SIZE 32

// each block will process one chunk of all head_dim, in one num_head.
template<typename T, uint32_t NUM_HTREADS, uint32_t CHUNK_SIZE, uint32_t ITEMS_PER_THREAD, uint32_t HEAD_DIM>
__global__ void SegmentMeanKernel(T* __restrict__ input,
                                  T* __restrict__ output,
                                  uint32_t* __restrict__ cu_seqlens,
                                  int stride_i_h,           // head_dim x total_seqlen
                                  int stride_i_d,           //            total_seqlen
                                  int stride_o_seqlen,      // num_head x head_dim
                                  int stride_o_h,           //            head_dim
                                  int batch_size) 
{
    const int bx = blockIdx.x;      // chunk_id
    const int head_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    const int tid = threadIdx.x;
    const uint32_t seqlen_this_time = cu_seqlens[batch_id + 1] - cu_seqlens[batch_id];

    const int start_pos = bx * CHUNK_SIZE;
    if (start_pos >= seqlen_this_time) return;

    const int block_num_this_seq = (seqlen_this_time + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    const int chunk_len_this_time = min(CHUNK_SIZE, seqlen_this_time - start_pos);

    // the head_dim can either be 64 or 128.
    __shared__ T mean_save[HEAD_DIM];     // why not store them in a warp-primitives?

    // every block, will process one chunk of each dim.
    for (int dim_id = 0; dim_id < HEAD_DIM; dim_id++) {
        using BlockLoad = cub::BlockLoad<T, NUM_HTREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
        __shared__ typename BlockLoad::TempStorage temp_storage_load;

        T loaded_data[ITEMS_PER_THREAD];

        T* input_idx = input + head_id * stride_i_h + dim_id * stride_i_d + cu_seqlens[batch_id] + start_pos;
        BlockLoad(temp_storage_load).Load(input_idx, loaded_data, chunk_len_this_time);    // valid items to load -> chunk_len_this_time

        // reduce sum, due to the `chunk_len_this_time` limitation, one thread compute one element
        // see https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html#_CPPv4N3cub11BlockReduce3SumE1Ti for details
        using BlockReduce = cub::BlockReduce<T, NUM_HTREADS>;
        __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

        T sum = T(0);
        for (int item_id = 0; item_id < ITEMS_PER_THREAD; item_id++) {
            T thread_data = ITEMS_PER_THREAD * tid < chunk_len_this_time ? loaded_data[item_id] : T(0);
            T sum_epoch = BlockReduce(temp_storage_reduce).Sum(thread_data, chunk_len_this_time);
            if (tid == 0) sum += sum_epoch;
        }
            
        if (threadIdx.x == 0) {
            mean_save[dim_id] = __hdiv(__hdiv(sum, T(chunk_len_this_time)), T(block_num_this_seq));   // online: div by block_num_this_seq in advance.
            __syncthreads();
        }
    }
    
    // store the mean value to output. This operation can be online.
    constexpr int items_per_thread = HEAD_DIM / WARP_SIZE;   // can either be 2 or 4
    int warp_id = tid / WARP_SIZE;
    if (warp_id == 0) {
        T temp_store[items_per_thread];
        T* output_idx = output + batch_id * stride_o_seqlen + head_id * stride_o_h + tid * items_per_thread;
        if constexpr (items_per_thread == 2) {
            ((float*)(temp_store))[0] = ((float*)(output_idx))[0]; // read
            temp_store[0] += mean_save[tid * 2];
            temp_store[1] += mean_save[tid * 2 + 1];
            ((float*)(output_idx))[0] = ((float*)(temp_store))[0];  // store
        } else {
            ((float2*)(temp_store))[0] = ((float2*)(output_idx))[0]; // read
            temp_store[0] += mean_save[tid * 4];
            temp_store[1] += mean_save[tid * 4 + 1];
            temp_store[2] += mean_save[tid * 4 + 2];
            temp_store[3] += mean_save[tid * 4 + 3];
            ((float2*)(output_idx))[0] = ((float2*)(temp_store))[0];  // store
        }
    }
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
    paddle::Tensor input_transposed = paddle::experimental::transpose(input, {1, 2, 0});    // transpose to: [num_head, head_dim, total_seqlen]

    paddle::Tensor output = paddle::zeros({batch_size, input.shape()[1], input.shape()[2]}, input.dtype(), paddle::GPUPlace());

    DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, {
        DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM ,{
            constexpr int NUM_THREADS = 1024;
            constexpr int ITEMS_PER_THREAD = 4;
            constexpr int CHUNK_SIZE = NUM_THREADS * ITEMS_PER_THREAD;

            dim3 grid((max_seqlen + CHUNK_SIZE - 1) / CHUNK_SIZE, num_head, batch_size);
            dim3 block(NUM_THREADS);

            SegmentMeanKernel<c_type, NUM_THREADS, CHUNK_SIZE, ITEMS_PER_THREAD, HEAD_DIM><<<grid, block>>>(
                reinterpret_cast<c_type*>(input_transposed.data()),
                reinterpret_cast<c_type*>(output.data()),
                reinterpret_cast<uint32_t*>(output.data()),
                input_transposed.strides()[0], input_transposed.strides()[1], 
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