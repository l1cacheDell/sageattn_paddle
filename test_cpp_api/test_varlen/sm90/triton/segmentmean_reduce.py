import torch
import triton
import triton.language as tl

import nvtx

@triton.jit
def segmented_mean_reduce_kernel(
    input_ptr, 
    output_ptr,
    cu_seqlen_ptr, 
    num_batches, 
    max_seqlen,
    num_heads, 
    head_dim,
    input_stride_seq, input_stride_head, input_stride_dim,
    output_stride_batch, output_stride_head, output_stride_dim,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_offset = tl.program_id(1) * BLOCK_SIZE_HEAD
    dim_offset = tl.program_id(2) * BLOCK_SIZE_DIM

    if batch_idx >= num_batches:
        return

    # 获取当前 segment 的 range
    seq_start = tl.load(cu_seqlen_ptr + batch_idx)
    seq_end = tl.load(cu_seqlen_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start

    # head 和 dim 的实际索引（block中相对位置）
    head_idx = head_offset + tl.arange(0, BLOCK_SIZE_HEAD)
    dim_idx = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
    mask_head = head_idx < num_heads
    mask_dim = dim_idx < head_dim

    # 初始化累加器（float32 精度）
    acc = tl.zeros((BLOCK_SIZE_HEAD, BLOCK_SIZE_DIM), dtype=tl.float32)

    for seq_offset in range(0, seq_len, BLOCK_SIZE_SEQ):
        local_seq_idx = tl.arange(0, BLOCK_SIZE_SEQ)
        mask_seq = local_seq_idx < (seq_len - seq_offset)
        global_seq = seq_start + seq_offset + local_seq_idx
        # shape: [BLOCK_SIZE_SEQ, BLOCK_SIZE_HEAD, BLOCK_SIZE_DIM]
        input_ptrs = (
            input_ptr +
            global_seq[:, None, None] * input_stride_seq +
            head_idx[None, :, None] * input_stride_head +
            dim_idx[None, None, :] * input_stride_dim
        )

        # 加载输入，注意输入 dtype 指明 float16 以避免不必要转换
        x = tl.load(input_ptrs, 
                    mask=mask_seq[:, None, None] & mask_head[None, :, None] & mask_dim[None, None, :],
                    other=0.0).to(tl.float32)

        acc += tl.sum(x, axis=0)  # reduce over seq axis

    mean = acc / tl.maximum(seq_len, 1)

    # 构造输出地址
    output_ptrs = (
        output_ptr +
        batch_idx * output_stride_batch +
        head_idx[:, None] * output_stride_head +
        dim_idx[None, :] * output_stride_dim
    )

    tl.store(output_ptrs, mean.to(input_ptr.dtype.element_ty), mask=mask_head[:, None] & mask_dim[None, :])

def segmented_mean_reduce_triton(
    input: torch.Tensor,  # [total_seqlen, num_heads, head_dim]
    cu_seqlen: torch.Tensor,  # [batch_size + 1]
    BLOCK_SIZE_SEQ=128,
    BLOCK_SIZE_HEAD=4,
    BLOCK_SIZE_DIM=64
):
    assert cu_seqlen.is_cuda and input.is_cuda
    assert cu_seqlen.dim() == 1
    
    num_batches = cu_seqlen.shape[0] - 1
    num_heads = input.shape[1]
    head_dim = input.shape[2]
    
    # 分配输出内存 [batch_size, num_heads, head_dim]
    output = torch.empty(
        (num_batches, num_heads, head_dim),
        dtype=input.dtype,
        device=input.device
    )
    
    # 计算每个batch的最大序列长度
    max_seqlen = (cu_seqlen[1:] - cu_seqlen[:-1]).max().item()
    
    # 确定kernel配置
    grid = (
        num_batches, 
        triton.cdiv(num_heads, BLOCK_SIZE_HEAD), 
        triton.cdiv(head_dim, BLOCK_SIZE_DIM)
    )
    
    # 调用kernel
    segmented_mean_reduce_kernel[grid](
        input, output,
        cu_seqlen,
        num_batches,
        max_seqlen,
        num_heads,
        head_dim,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )
    
    return output

def test_segmented_mean_reduce():
    # 构造测试数据
    batch_sizes = [1024 * 64 - 1, 1024 * 64 + 1, 1024 * 64 + 5]
    num_batches = len(batch_sizes)
    num_heads = 24
    head_dim = 128
    dtype = torch.float16
    
    # 生成累计序列长度
    cu_seqlen = torch.cumsum(
        torch.tensor([0] + batch_sizes, device="cuda"),
        dim=0
    )
    
    # 生成随机输入数据
    total_seqlen = sum(batch_sizes)
    input = torch.randn(
        total_seqlen, num_heads, head_dim,
        dtype=dtype, device="cuda"
    )

    # 57ms
    BLOCK_SIZE_SEQ=128
    BLOCK_SIZE_HEAD=4
    BLOCK_SIZE_DIM=64

    # 47 ms
    BLOCK_SIZE_SEQ=256
    BLOCK_SIZE_HEAD=4
    BLOCK_SIZE_DIM=64
    
    # Triton实现
    triton_output = segmented_mean_reduce_triton(
        input, cu_seqlen,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM
    )

    # profile
    for i in range(100):
        torch.cuda.synchronize()
        start = nvtx.start_range(message='Segmented Mean', color='red')
        triton_output = segmented_mean_reduce_triton(
            input, cu_seqlen,
            BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
            BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD,
            BLOCK_SIZE_DIM=BLOCK_SIZE_DIM
        )
        torch.cuda.synchronize()
        nvtx.end_range(start)
    
    # PyTorch原生实现用于验证
    torch_output = []
    for i in range(num_batches):
        start = cu_seqlen[i]
        end = cu_seqlen[i+1]
        segment = input[start:end]
        torch_output.append(segment.mean(dim=0))
    torch_output = torch.stack(torch_output)
    
    # 计算误差
    cos_sim = torch.cosine_similarity(
        triton_output.flatten(),
        torch_output.flatten(),
        dim=0
    )
    max_diff = (triton_output - torch_output).abs().max()
    print(f"Cosine Similarity: {cos_sim.item():.6f}")
    print(f"Max Difference: {max_diff.item():.6f}")

if __name__ == "__main__":
    test_segmented_mean_reduce()