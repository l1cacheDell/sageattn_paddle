#include "sageattn_utils.cuh"
#include <cuda_fp16.h>   // for __half and __half2 (float16) intrinsics
#include <cuda_bf16.h>   // for __nv_bfloat16 and __nv_bfloat162 (bfloat16)
#include <cub/cub.cuh>   // for CUB utilities (optional, e.g., for reductions)

#define WARP_SIZE 32

#define BLOCK_SIZE_DIM 64

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

// THD means `thread head dim`
template<typename T>
struct THD {
    T data[BLOCK_SIZE_DIM];
};

template<typename T>
__device__ __forceinline__ THD<T> reduction_sum(THD<T> a, THD<T> b) {
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_DIM; i++) {
        a.data[i] = a.data[i] + b.data[i];
    }
    return a;
}

// each block will process one chunk of 64 head_dim, and 4 num_head.
template<typename T, uint32_t NUM_THREADS, uint32_t BLOCK_SIZE_SEQ, uint32_t BLOCK_SIZE_HEAD, uint32_t HEAD_DIM=128>
__global__ void SegmentMeanKernel(T* __restrict__ input,
                                  T* __restrict__ output,
                                  uint32_t* __restrict__ cu_seqlens,
                                  int stride_i_seqlen,      // num_head x head_dim
                                  int stride_i_h,           //            head_dim
                                  int stride_o_bsz,         // num_head x head_dim
                                  int stride_o_h,           //            head_dim
                                  int num_head,
                                  int batch_size) 
{
    const int batch_id = blockIdx.x;
    const int head_block_id = blockIdx.y;
    const int dim_block_id = blockIdx.z;

    const int tid = threadIdx.x;
    const int seqlen_this_batch = cu_seqlens[batch_id + 1] - cu_seqlens[batch_id];

    const int NUM_SEQ_PER_THREAD = (seqlen_this_batch + NUM_THREADS - 1) / NUM_THREADS;    // (131 + 127) / 128 = 2

    THD<T> thread_data;    // 64

    // one block, computes one seq, 4 head, 64 head_dim
    __shared__ T sum_result[BLOCK_SIZE_HEAD][BLOCK_SIZE_DIM];

    using BlockReduce = cub::BlockReduce<THD<T>, NUM_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    // this loop will be executed in 128 threads, thus add NUM_THREADS
    for (int seq_id = tid * NUM_SEQ_PER_THREAD; seq_id < seqlen_this_batch; seq_id++) {

        #pragma unroll
        for (int head_id = 0; head_id < BLOCK_SIZE_HEAD; head_id++) {
            int seq_idx = cu_seqlens[batch_id] + seq_id;
            int head_idx = head_block_id * BLOCK_SIZE_HEAD + head_id;
            T* input_idx = input + seq_idx * stride_i_seqlen + head_idx * stride_i_h + dim_block_id * BLOCK_SIZE_DIM;
            
            // cover 64 head_dim
            *(float4*)(&thread_data.data[0])  = *(float4*)(input_idx);          // load 8 data
            *(float4*)(&thread_data.data[8])  = *(float4*)(input_idx + 8);
            *(float4*)(&thread_data.data[16]) = *(float4*)(input_idx + 16);
            *(float4*)(&thread_data.data[24]) = *(float4*)(input_idx + 24);
            *(float4*)(&thread_data.data[32]) = *(float4*)(input_idx + 32);
            *(float4*)(&thread_data.data[40]) = *(float4*)(input_idx + 40);
            *(float4*)(&thread_data.data[48]) = *(float4*)(input_idx + 48);
            *(float4*)(&thread_data.data[56]) = *(float4*)(input_idx + 56);

            THD<T> res = BlockReduce(temp_storage_reduce).Reduce(thread_data, reduction_sum<T>);

            if (tid == 0) {
                if (seq_id == 0) {
                    // load to shared memory
                    *(float4*)(&sum_result[head_id][0]) = *(float4*)(&res.data[0]);
                    *(float4*)(&sum_result[head_id][8])  = *(float4*)(&res.data[8]);
                    *(float4*)(&sum_result[head_id][16]) = *(float4*)(&res.data[16]);
                    *(float4*)(&sum_result[head_id][24]) = *(float4*)(&res.data[24]);
                    *(float4*)(&sum_result[head_id][32]) = *(float4*)(&res.data[32]);
                    *(float4*)(&sum_result[head_id][40]) = *(float4*)(&res.data[40]);
                    *(float4*)(&sum_result[head_id][48]) = *(float4*)(&res.data[48]);
                    *(float4*)(&sum_result[head_id][56]) = *(float4*)(&res.data[56]);
                } else {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_SIZE_DIM; i++)
                        sum_result[head_id][i] += res.data[i];
                }
            }

            __syncthreads();
        }
    }

    // store to global memory
    if (tid == 0) {
        for (int head_id = 0; head_id < BLOCK_SIZE_HEAD; head_id++) {
            int head_idx = head_block_id * BLOCK_SIZE_HEAD + head_id;
            if (head_idx < num_head) {
                T* output_idx = output + batch_id * stride_o_bsz + head_idx * stride_o_h + dim_block_id * BLOCK_SIZE_DIM;

                *(float4*)output_idx        = *(float4*)(&sum_result[head_id][0]);
                *(float4*)(output_idx + 8)  = *(float4*)(&sum_result[head_id][8]);
                *(float4*)(output_idx + 16) = *(float4*)(&sum_result[head_id][16]);
                *(float4*)(output_idx + 24) = *(float4*)(&sum_result[head_id][24]);
                *(float4*)(output_idx + 32) = *(float4*)(&sum_result[head_id][32]);
                *(float4*)(output_idx + 40) = *(float4*)(&sum_result[head_id][40]);
                *(float4*)(output_idx + 48) = *(float4*)(&sum_result[head_id][48]);
                *(float4*)(output_idx + 56) = *(float4*)(&sum_result[head_id][56]);
            }
        }
    }

}

#include <cub/cub.cuh>

template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y, int CHUNK_SIZE, int head_dim>
__global__ void OptimizedSegmentMeanKernel(
    T* __restrict__ input,
    T* __restrict__ output,
    const uint32_t* __restrict__ cu_seqlens,
    int stride_i_seqlen,
    int stride_i_h,
    int stride_o_seqlen,
    int stride_o_h,
    int num_head, 
    int batch_size
) {
    // 一个block 单次计算 chunk_size * head_dim个数据
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared_data = reinterpret_cast<T*>(shared_mem);

    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.x;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int tid = tid_y * BLOCK_DIM_X + tid_x;

    const int seq_start = cu_seqlens[batch_id];
    const int seq_end = cu_seqlens[batch_id + 1];
    const int seq_len = seq_end - seq_start;

    // 每个线程负责的维度范围
    // 一个线程计算 4 个 head_dim, 32 个线程计算一整个 head_dim。
    // 一共有 128个线程。所以可以一下子算 4 个seqlen的数据。
    const int dim_start = tid_x * (head_dim / BLOCK_DIM_X);     // BLOCK_DIM_X = 32, BLOCK_DIM_Y = 4, head_dim / BLOCK_DIM_X = 4
    const int dim_end = (tid_x + 1) * (head_dim / BLOCK_DIM_X);

    T sum[head_dim / BLOCK_DIM_X] = {0};    // 4, 一个thread计算 4 个 head_dim的sum

    // 分块处理序列, 他这个chunk开得小，就128一个chunk
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        const int chunk_end = min(chunk_start + CHUNK_SIZE, seq_len);

        // 协作加载一个 Chunk 到共享内存
        // 数据load的瓶颈，不在于load多少数据，而在于访存。你之前那种block load，它就是慢。像这样老老实实开一个循环，算得就快。
        // 先load到shared mem里面来, load
        for (int s = chunk_start; s < chunk_end; s += BLOCK_DIM_Y) { // loop 32次
            const int seq_idx = s + tid_y;      // 0, 1, 2, 3
            if (seq_idx < chunk_end) {
                const T* src = input + (seq_start + seq_idx) * stride_i_seqlen + head_id * stride_i_h;
                #pragma unroll
                for (int d = dim_start; d < dim_end; ++d) { // loop 4 次
                    shared_data[(seq_idx - chunk_start) * head_dim + d] = src[d];
                }
            }
        }
        __syncthreads();

        // 局部归约
        // 再从shared mem里面提取，做sum
        // 每一个thread x 读取4个head_dim, 32个thread计算 128个head_dim
        // 但是其实0-3, 4-7, ... 每4个thread计算的结果是一样的

            // 确定一下bottleneck


        // #pragma unroll
        // for (int s = 0; s < chunk_end - chunk_start; ++s) {
        //     #pragma unroll
        //     for (int d = dim_start; d < dim_end; ++d) {
        //         sum[d - dim_start] += shared_data[s * head_dim + d];
        //     }
        // }
        // __syncthreads();
    }

    // 直到现在，每一个thread都计算好了**当前num_head**，4个head_dim的sum
    // 0-3, 4-7, ... 每4个thread计算的结果是一样的。一共32组，每一组持有4个head_dim的sum

    // Block 内归约（使用 CUB）
    // typedef cub::BlockReduce<T, BLOCK_DIM_X, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, BLOCK_DIM_Y> BlockReduce;
    // __shared__ typename BlockReduce::TempStorage temp_storage;

    // T block_sum[head_dim / BLOCK_DIM_X];    // 
    // #pragma unroll
    // for (int d = 0; d < head_dim / BLOCK_DIM_X; ++d) {
    //     block_sum[d] = BlockReduce(temp_storage).Sum(sum[d]);
    // }

    // 写回结果
    if (tid_y == 0) {
        T* dst = output + batch_id * stride_o_seqlen + head_id * stride_o_h;
        #pragma unroll
        for (int d = dim_start; d < dim_end; ++d) {
            dst[d] = sum[d - dim_start] / static_cast<T>(seq_len);
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
    paddle::Tensor output = paddle::zeros({batch_size, input.shape()[1], input.shape()[2]}, input.dtype(), paddle::GPUPlace());

    PD_CHECK(input.dtype() == paddle::DataType::FLOAT16 || input.dtype() == paddle::DataType::BFLOAT16, "Only float16 and bfloat16 are supported");

    DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, {
        DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM, {
            constexpr int NUM_THREADS = 256;

            constexpr int BLOCK_SIZE_SEQ = 256;
            constexpr int BLOCK_SIZE_HEAD = 4;

            // for [131, 24, 128], the grid is: (1, 6, 2)
            
            constexpr int BLOCK_DIM_X = 32;
            constexpr int BLOCK_DIM_Y = 4;
            constexpr int CHUNK_SIZE = 128;
            size_t sMemSize = CHUNK_SIZE * HEAD_DIM * 2; // 16 bits - 2 byte

            dim3 grid(num_head, batch_size);
            dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);   // 32 * 4

            // SegmentMeanKernel<c_type, NUM_THREADS, BLOCK_SIZE_SEQ, BLOCK_SIZE_HEAD, HEAD_DIM><<<grid, block>>>(
            OptimizedSegmentMeanKernel<c_type, BLOCK_DIM_X, BLOCK_DIM_Y, CHUNK_SIZE, HEAD_DIM><<<grid, block, sMemSize>>>(
                reinterpret_cast<c_type*>(input.data()),                // [total_seqlen, num_head, head_dim]
                reinterpret_cast<c_type*>(output.data()),               // [batch_size, num_head, head_dim]
                reinterpret_cast<uint32_t*>(cu_seqlens.data()),
                input.strides()[0], input.strides()[1], 
                output.strides()[0], output.strides()[1],
                num_head,
                batch_size);
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