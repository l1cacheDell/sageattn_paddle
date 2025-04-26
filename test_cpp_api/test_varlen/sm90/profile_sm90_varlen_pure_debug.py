# To run this script you need to install torch and flash attn 3
import paddle
import sageattn_custom_ops
import numpy as np
import nvtx
import random
import argparse

def precision_cmp_paddle(t1: paddle.Tensor, t2: paddle.Tensor):
    
    x, xx = paddle.cast(t1, dtype='float32'), paddle.cast(t2, dtype='float32')
    # 重塑张量并计算余弦相似度
    x_reshaped = paddle.reshape(x, [1, -1])
    xx_reshaped = paddle.reshape(xx, [1, -1])
    sim = paddle.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # 计算 L1 误差
    l1 = (paddle.abs(x - xx).sum() / paddle.abs(xx).sum()).item()
    max_diff = paddle.max(x - xx)
    
    return sim, l1, max_diff

def pad_sequences_to_aligned_chunks(v, cu_seqlens_v, align_size=128):
    """
    将变长序列填充到指定对齐大小
    
    参数:
        v: 输入张量，形状为[seq_len, num_heads, head_dim]
        cu_seqlens_v: 累积序列长度，如[0, 246, 394, seq_len]
        align_size: 对齐大小(默认为128)
    
    返回:
        填充后的张量
        新的cu_seqlens_v(包含填充后的位置)
    """
    # 1. 计算每个序列的实际长度和需要填充的长度
    batch_size = len(cu_seqlens_v) - 1
    seq_lengths = cu_seqlens_v[1:] - cu_seqlens_v[:-1]
    
    # 2. 计算每个序列需要填充到的长度
    padded_lengths = ((seq_lengths + align_size - 1) // align_size) * align_size
    total_padded_length = paddle.sum(padded_lengths).item()
    
    # 3. 创建结果张量(初始化为0)
    padded_v = paddle.zeros(
        [total_padded_length, v.shape[1], v.shape[2]],
        dtype=v.dtype
    )
    
    # 4. 计算新的cu_seqlens_v
    new_cu_seqlens = paddle.zeros_like(cu_seqlens_v)
    new_cu_seqlens[0] = 0
    for i in range(1, batch_size + 1):
        new_cu_seqlens[i] = new_cu_seqlens[i-1] + padded_lengths[i-1]
    
    # 5. 将原始数据复制到填充后的张量中
    for i in range(batch_size):
        start = cu_seqlens_v[i].item()
        end = cu_seqlens_v[i+1].item()
        chunk_length = end - start
        
        # 计算填充后的起始位置
        padded_start = new_cu_seqlens[i].item()
        
        # 复制数据
        padded_v[padded_start:padded_start+chunk_length] = v[start:end]
    
    return padded_v, new_cu_seqlens

head_dim = 128
sm_scale = head_dim ** -0.5
is_causal=True

runtime_dtype = paddle.bfloat16
# runtime_dtype = paddle.float16

# q = paddle.load("./sa_inputs/q.pdparams").astype(runtime_dtype)
# k = paddle.load("./sa_inputs/k.pdparams").astype(runtime_dtype)
# v_padded = paddle.load("./sa_inputs/padded_v.pdparams").astype(runtime_dtype)
# cu_seqlen = paddle.load("./sa_inputs/cu_seqlen.pdparams").astype(paddle.int32)
# cu_seqlen_v_padded = paddle.load("./sa_inputs/cu_seqlen_v_padded.pdparams").astype(paddle.int32)
# km = paddle.load("./sa_inputs/km.pdparams").astype(runtime_dtype)

q = paddle.load("./inputs_2/q.pdparams").astype(paddle.bfloat16)
k = paddle.load("./inputs_2/k.pdparams").astype(paddle.bfloat16)
v_padded = paddle.load("./inputs_2/padded_v.pdparams").astype(paddle.bfloat16)
cu_seqlen_q = paddle.load("./inputs_2/cu_seqlen_q.pdparams").astype(paddle.int32)
cu_seqlen_k = paddle.load("./inputs_2/cu_seqlen_k.pdparams").astype(paddle.int32)
cu_seqlen_v_padded = paddle.load("./inputs_2/cu_seqlen_v_padded.pdparams").astype(paddle.int32)
km = paddle.load("./inputs_2/km.pdparams").astype(paddle.bfloat16)

# q = paddle.randn(q.shape, dtype=paddle.float16)
# k = paddle.randn(k.shape, dtype=paddle.float16)
# v_padded = paddle.randn(v_padded.shape, dtype=paddle.float16)

# =====================================================
# if randomly generated tensor, then the code can run

# q = paddle.randn([131, 12, 128], paddle.float16)
# k = paddle.randn([131, 2, 128], paddle.float16)
# v = paddle.randn([131, 2, 128], paddle.float16)

# cu_seqlen = paddle.to_tensor([0, 131], dtype=paddle.int32)

# v_padded, cu_seqlen_v_padded = pad_sequences_to_aligned_chunks(v, cu_seqlen, align_size=128)

# =====================================================

print(cu_seqlen_q)
print(cu_seqlen_v_padded)
print(cu_seqlen_q[-1].item())
print(km.shape, km.dtype)
print(q.shape, q.dtype)
print(k.shape, k.dtype)
print(v_padded.shape, v_padded.dtype)

for i in range(15):
    print(f"epoch: {i}")
    o1, q_int8, k_int8, vfp8_fused, v_transposed_fused = sageattn_custom_ops.sage_attention_varlen2(q, 
                                                k, 
                                                v_padded, 
                                                cu_seqlen_q,
                                                cu_seqlen_k,
                                                cu_seqlen_v_padded,
                                                km,
                                                None,
                                                131,
                                                131,
                                                256,
                                                head_dim**-0.5,
                                                "per_warp",
                                                "fp16",
                                                tensor_layout=0,
                                                is_causal=is_causal,
                                                smooth_k=True, 
                                                smooth_v=False, 
                                                return_lse=False)
    paddle.device.synchronize()

    nan_mask = paddle.isnan(q_int8.astype(paddle.float16))
    nan_indices = paddle.nonzero(nan_mask)
    print(f"q_int8 nan indices: {nan_indices}")

    nan_mask = paddle.isnan(k_int8.astype(paddle.float16))
    nan_indices = paddle.nonzero(nan_mask)
    print(f"k_int8 nan indices: {nan_indices}")

    nan_mask = paddle.isnan(o1)
    nan_indices = paddle.nonzero(nan_mask)
    print(f"o1 nan indices: {nan_indices}")

    nan_mask = paddle.isnan(vfp8_fused.astype(paddle.float16))
    nan_indices = paddle.nonzero(nan_mask)
    print(f"vfp8_fused nan indices: {nan_indices}")

    nan_mask = paddle.isnan(v_transposed_fused.astype(paddle.float16))
    nan_indices = paddle.nonzero(nan_mask)
    print(f"v_transposed_fused nan indices: {nan_indices}")


paddle.device.synchronize()

# for i in range(100):
#     paddle.device.synchronize()
#     paddle_nvtx = nvtx.start_range(message="paddle", color="green")
#     km = triton_ops.segment_mean(k, cu_seqlens)
#     o1, vfp8_fused, v_transposed_fused = sageattn_custom_ops.sage_attention_varlen(q, 
#                                                 k, 
#                                                 padded_v, 
#                                                 cu_seqlens,
#                                                 cu_seqlens,
#                                                 new_cu_seqlen_v,
#                                                 km,
#                                                 None,
#                                                 max_seqlen,
#                                                 max_seqlen,
#                                                 new_cu_seqlen_v[-1],
#                                                 head_dim**-0.5,
#                                                 "per_warp",
#                                                 "fp16",
#                                                 tensor_layout=0,
#                                                 is_causal=is_causal,
#                                                 smooth_k=True, 
#                                                 smooth_v=False, 
#                                                 return_lse=False)
#     paddle.device.synchronize()
#     nvtx.end_range(paddle_nvtx)

# =========== Compare diff zone ============
# sim, l1, max_diff = precision_cmp_paddle(o, o1)
# print(f"Total sim: {sim}, l1: {l1}, max_diff: {max_diff}")

# nan_mask = paddle.isnan((o - o1).astype(paddle.float32))
# nan_indices = paddle.nonzero(nan_mask)
# print(nan_indices)
# # 转为 NumPy
# nan_indices_np = nan_indices.numpy()

# nan_mask = paddle.isnan(o)
# nan_indices = paddle.nonzero(nan_mask)
# print(nan_indices)
# # 转为 NumPy
# nan_indices_np = nan_indices.numpy()

# nan_mask = paddle.isnan(o1)
# nan_indices = paddle.nonzero(nan_mask)
# print(nan_indices)
# # 转为 NumPy
# nan_indices_np = nan_indices.numpy()

# # 保存为 txt 文件（整数格式）
# np.savetxt("nan_indices.txt", nan_indices_np, fmt="%d")

# # 打印保存的路径
# print("已保存 nan_indices 到 nan_indices.txt")