# To run this script you need to install paddle and flash attn 3
import paddle
import sageattn_custom_ops
import numpy as np
import nvtx
import random

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

# paddle varlen: FA3
slices = [1028, 1024 * 5]
bsz = 2
total_seqlen: int = sum(slices)



max_seqlen: int = 1024 * 5
cu_seqlens: list = [0, 1028, 1024 * 5 + 1028]
cu_seqlens_list = cu_seqlens

num_head = 24
head_dim = 128
sm_scale = head_dim ** -0.5
is_causal=True

cu_seqlens = paddle.to_tensor(cu_seqlens, dtype=paddle.int32)

# sdpa
q1 = paddle.randn([1, 1028, num_head, head_dim], dtype=paddle.float16)
k1 = paddle.randn([1, 1028, num_head, head_dim], dtype=paddle.float16)
v1 = paddle.randn([1, 1028, num_head, head_dim], dtype=paddle.float16)
km1 = paddle.mean(k1, 1)
# o_paddle1 = paddle.nn.functional.scaled_dot_product_attention(q1, k1, v1, is_causal=is_causal)
o_paddle1, q_int81, k_int81, v_fp81 = sageattn_custom_ops.sage_attention(q1, k1, v1, km1, None, 
                                               head_dim**-0.5,
                                                "per_warp",
                                                "fp32+fp32",
                                                tensor_layout=0,
                                                is_causal=is_causal,
                                                smooth_k=True, 
                                                smooth_v=False, 
                                                return_lse=False)

q2 = paddle.randn([1, 1024 * 5, num_head, head_dim], dtype=paddle.float16)
k2 = paddle.randn([1, 1024 * 5, num_head, head_dim], dtype=paddle.float16)
v2 = paddle.randn([1, 1024 * 5, num_head, head_dim], dtype=paddle.float16)
km2 = paddle.mean(k2, 1)
o_paddle2, q_int82, k_int82, v_fp82 = sageattn_custom_ops.sage_attention(q2, k2, v2, km2, None, 
                                               head_dim**-0.5,
                                                "per_warp",
                                                "fp32+fp32",
                                                tensor_layout=0,
                                                is_causal=is_causal,
                                                smooth_k=True, 
                                                smooth_v=False, 
                                                return_lse=False)

o_paddle = paddle.concat([o_paddle1.squeeze(0), o_paddle2.squeeze(0)], axis=0)
print("the original cu_seqlens: ", cu_seqlens)

q = paddle.concat([q1.squeeze(0), q2.squeeze(0)], axis=0)
k = paddle.concat([k1.squeeze(0), k2.squeeze(0)], axis=0)
v = paddle.concat([v1.squeeze(0), v2.squeeze(0)], axis=0)

segment_lengths = paddle.concat([cu_seqlens[:1], cu_seqlens[1:] - cu_seqlens[:-1]])[1:]
segment_ids = paddle.concat([paddle.full([length], i, dtype='int32') for i, length in enumerate(segment_lengths)])
# print(segment_ids)

# prepare the padded v input
padded_v, new_cu_seqlen_v = pad_sequences_to_aligned_chunks(v, cu_seqlens, 64)
print("the padded cu-seqlens: ", new_cu_seqlen_v)

km1 = paddle.mean(k1, 1)
km2 = paddle.mean(k2, 1)
km = paddle.concat([km1, km2], axis=0)
print(padded_v.shape)

for i in range(1):
    sa_nvtx = nvtx.start_range(message="SA_NVTX_2", color='red')
    o1, q_int8, k_int8, v_fp8 = sageattn_custom_ops.sage_attention_varlen(q, 
                                                k, 
                                                padded_v, 
                                                cu_seqlens,
                                                cu_seqlens,
                                                new_cu_seqlen_v,
                                                km,
                                                None,
                                                max_seqlen,
                                                max_seqlen,
                                                new_cu_seqlen_v[-1],
                                                head_dim**-0.5,
                                                "per_warp",
                                                "fp32",
                                                tensor_layout=0,
                                                is_causal=is_causal,
                                                smooth_k=True, 
                                                smooth_v=False, 
                                                return_lse=False)
    paddle.device.synchronize()
    nvtx.end_range(sa_nvtx)

paddle.device.synchronize()
# print(o.shape, o1.shape)

# nan_mask = paddle.isnan(o1)
# nan_indices = paddle.nonzero(nan_mask)
# print(nan_indices)


print(o1.shape, o_paddle.shape)
sim, l1, max_diff = precision_cmp_paddle(o_paddle.squeeze(0), o1)
print(f"Total sim: {sim}, l1: {l1}, max_diff: {max_diff}")

q_int8_original = paddle.concat([q_int81.squeeze(), q_int82.squeeze()], axis=0)
k_int8_original = paddle.concat([k_int81.squeeze(), k_int82.squeeze()], axis=0)

v_fp8_original = paddle.concat([v_fp81.squeeze().astype(paddle.float32), v_fp82.squeeze().astype(paddle.float32)], axis=-1)

sim, l1, max_diff = precision_cmp_paddle(q_int8, q_int8_original)
print(f"Total sim: {sim}, l1: {l1}, max_diff: {max_diff}")

sim, l1, max_diff = precision_cmp_paddle(k_int8, k_int8_original)
print(f"Total sim: {sim}, l1: {l1}, max_diff: {max_diff}")

sim, l1, max_diff = precision_cmp_paddle(v_fp8, v_fp8_original)
print(f"Total sim: {sim}, l1: {l1}, max_diff: {max_diff}")