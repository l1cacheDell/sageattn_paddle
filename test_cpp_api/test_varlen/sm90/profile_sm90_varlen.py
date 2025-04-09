# To run this script you need to install torch and flash attn 3
import torch
from flash_attn_interface import flash_attn_varlen_func
import paddle
import sageattn_custom_ops
import numpy as np
import nvtx
import random
from paddlemix import triton_ops

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

# torch varlen: FA3
seqlen = 1024
bsz = 4
total_seqlens = [seqlen - random.randint(-10, i + 10) for i in range(bsz)]    # 例如 [1023, 1024, 1026, 1025]
# total_seqlens = [1027, 1018, 1014, 1014]

# bsz = 2
# total_seqlens = [1027, 1018]
total_seqlen: int = sum(total_seqlens)
print("the seqlens of 4-segs: ", total_seqlens)
max_seqlen: int = max(total_seqlens)
cu_seqlens: list = [0] + [sum(total_seqlens[:i+1]) for i in range(bsz)]
cu_seqlens_list = cu_seqlens

num_head = 24
head_dim = 128
sm_scale = head_dim ** -0.5
is_causal=True

q = torch.randn(total_seqlen, num_head, head_dim, dtype=torch.float16).cuda()
k = torch.randn(total_seqlen, num_head, head_dim, dtype=torch.float16).cuda()
v = torch.randn(total_seqlen, num_head, head_dim, dtype=torch.float16).cuda()
cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).cuda()

o_torch, _ = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, sm_scale, is_causal)
torch.cuda.synchronize()

# prepare tensor
q_npy = q.cpu().numpy()
k_npy = k.cpu().numpy()
v_npy = v.cpu().numpy()
o_npy = o_torch.cpu().numpy()
cu_seqlens_npy = cu_seqlens.cpu().numpy()

q = paddle.to_tensor(q_npy, dtype=paddle.float16)
k = paddle.to_tensor(k_npy, dtype=paddle.float16)
v = paddle.to_tensor(v_npy, dtype=paddle.float16)
o = paddle.to_tensor(o_npy, dtype=paddle.float16)
cu_seqlens = paddle.to_tensor(cu_seqlens_npy, dtype=paddle.int32)
print("the original cu_seqlens: ", cu_seqlens)

segment_lengths = paddle.concat([cu_seqlens[:1], cu_seqlens[1:] - cu_seqlens[:-1]])[1:]
segment_ids = paddle.concat([paddle.full([length], i, dtype='int32') for i, length in enumerate(segment_lengths)])
# print(segment_ids)

# prepare the padded v input
padded_v, new_cu_seqlen_v = pad_sequences_to_aligned_chunks(v, cu_seqlens, 128)
print("the padded cu-seqlens: ", new_cu_seqlen_v)
taltal_seqlens_padded_v = [new_cu_seqlen_v[i]-new_cu_seqlen_v[i-1] for i in range(1, len(new_cu_seqlen_v))]

for i in range(100):
    km = triton_ops.segment_mean(k, cu_seqlens)
    o1, vfp8_fused, v_transposed_fused = sageattn_custom_ops.sage_attention_varlen(q, 
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
                                                    "fp16",
                                                    tensor_layout=0,
                                                    is_causal=is_causal,
                                                    smooth_k=True, 
                                                    smooth_v=False, 
                                                    return_lse=False)

paddle.device.synchronize()
print(o.shape, o1.shape)

nan_mask = paddle.isnan(o1)
nan_indices = paddle.nonzero(nan_mask)
print(nan_indices)

# ========================================
q1, q2, q3, q4 = paddle.split(q, total_seqlens, axis=0)
k1, k2, k3, k4 = paddle.split(k, total_seqlens, axis=0)
v1, v2, v3, v4 = paddle.split(padded_v, taltal_seqlens_padded_v, axis=0)

q1 = paddle.unsqueeze(q1, axis=0)
k1 = paddle.unsqueeze(k1, axis=0)
v1 = paddle.unsqueeze(v1, axis=0)

q2 = paddle.unsqueeze(q2, axis=0)
k2 = paddle.unsqueeze(k2, axis=0)
v2 = paddle.unsqueeze(v2, axis=0)

q3 = paddle.unsqueeze(q3, axis=0)
k3 = paddle.unsqueeze(k3, axis=0)
v3 = paddle.unsqueeze(v3, axis=0)

q4 = paddle.unsqueeze(q4, axis=0)
k4 = paddle.unsqueeze(k4, axis=0)
v4 = paddle.unsqueeze(v4, axis=0)


km1 = paddle.mean(k1, axis=1, keepdim=True)
km1 = km1.squeeze(1) 

km2 = paddle.mean(k2, axis=1, keepdim=True)
km2 = km2.squeeze(1) 

km3 = paddle.mean(k3, axis=1, keepdim=True)
km3 = km3.squeeze(1) 

km4 = paddle.mean(k4, axis=1, keepdim=True)
km4 = km4.squeeze(1) 

o_set_1, vfp8_1, v_tm_1 = sageattn_custom_ops.sage_attention(q1, k1, v1, km1, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=False)
o_set_2, vfp8_2, v_tm_2 = sageattn_custom_ops.sage_attention(q2, k2, v2, km2, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=False)
o_set_3, vfp8_3, v_tm_3 = sageattn_custom_ops.sage_attention(q3, k3, v3, km3, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=False)
o_set_4, vfp8_4, v_tm_4 = sageattn_custom_ops.sage_attention(q4, k4, v4, km4, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=False)

# print(o1)
sim, l1, max_diff = precision_cmp_paddle(o, o1)
print(f"Total sim: {sim}, l1: {l1}, max_diff: {max_diff}")

o_seg_1, o_seg_2, o_seg_3, o_seg_4 = paddle.split(o, total_seqlens, axis=0)     # FA3-varlen

o_seg_1, o_seg_2, o_seg_3, o_seg_4 = o_set_1, o_set_2, o_set_3, o_set_4
o1_seg_1, o1_seg_2, o1_seg_3, o1_seg_4 = paddle.split(o1, total_seqlens, axis=0)
sim, l1, mdiff = precision_cmp_paddle(o_seg_1, o1_seg_1)
print(f"seg_1 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

sim, l1, mdiff = precision_cmp_paddle(o_seg_2, o1_seg_2)
print(f"seg_2 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

sim, l1, mdiff = precision_cmp_paddle(o_seg_3, o1_seg_3)
print(f"seg_3 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

sim, l1, mdiff = precision_cmp_paddle(o_seg_4, o1_seg_4)
print(f"seg_3 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# fp8 v
# vfp8_f1, vfp8_f2, vfp8_f3, vfp8_f4 = paddle.split(vfp8_fused.astype(paddle.float32), taltal_seqlens_padded_v, axis=-1)

# sim, l1, mdiff = precision_cmp_paddle(vfp8_f1, vfp8_1)
# print(f"vfp8 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(vfp8_f2, vfp8_2)
# print(f"vfp8 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(vfp8_f3, vfp8_3)
# print(f"vfp8 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(vfp8_f4, vfp8_4)
# print(f"vfp8 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# fp8 v tm
vfp8_f1, vfp8_f2, vfp8_f3, vfp8_f4 = paddle.split(v_transposed_fused.astype(paddle.float32), taltal_seqlens_padded_v, axis=-1)

sim, l1, mdiff = precision_cmp_paddle(vfp8_f1, v_tm_1)
print(f"v tm 1 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

sim, l1, mdiff = precision_cmp_paddle(vfp8_f2, v_tm_2)
print(f"v tm 2 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

sim, l1, mdiff = precision_cmp_paddle(vfp8_f3, v_tm_3)
print(f"v tm 3 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

sim, l1, mdiff = precision_cmp_paddle(vfp8_f4, v_tm_4)
print(f"v tm 4 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# index = paddle.argmax((vfp8_f1 - v_tm_1).astype(paddle.float32)).item()
# print(index)
# print(v_tm_1.shape)
# token_id = index % 1152
# head_id = index % (24 * 1152) if index > 24 * 1152 else 0
# head_dim_id = index % (24 * 1152 * 128) if index > 24 * 1152 * 128 else 0
# print(f"dim: {head_dim_id}, head: {head_id}, token: {token_id}")

nan_indices = paddle.nonzero((vfp8_f1 - v_tm_1.squeeze(0)).astype(paddle.float32))
print(nan_indices)
# 转为 NumPy
nan_indices_np = nan_indices.numpy()

# 保存为 txt 文件（整数格式）
np.savetxt("nan_indices.txt", nan_indices_np, fmt="%d")

# 打印保存的路径
print("已保存 nan_indices 到 nan_indices.txt")


# analyze the first seg
# print("\n================\n")
# o_seg_1 = o_seg_1.squeeze(0)
# o_seg_1, o_seg_2, o_seg_3, o_seg_4 = paddle.split(o_seg_1, [250, 250, 250, 1027-750], axis=0)
# o1_seg_1, o1_seg_2, o1_seg_3, o1_seg_4 = paddle.split(o1_seg_1, [250, 250, 250, 1027-750], axis=0)
# sim, l1, mdiff = precision_cmp_paddle(o_seg_1, o1_seg_1)
# print(f"seg_1 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(o_seg_2, o1_seg_2)
# print(f"seg_2 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(o_seg_3, o1_seg_3)
# print(f"seg_3 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(o_seg_4, o1_seg_4)
# print(f"seg_3 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# print("\n================\n")
# o_seg_1, o_seg_2, o_seg_3, o_seg_4 = paddle.split(o_seg_1, [10, 20, 30, 250-60], axis=0)
# o1_seg_1, o1_seg_2, o1_seg_3, o1_seg_4 = paddle.split(o1_seg_1, [10, 20, 30, 250-60], axis=0)
# sim, l1, mdiff = precision_cmp_paddle(o_seg_1, o1_seg_1)
# print(f"seg_1 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(o_seg_2, o1_seg_2)
# print(f"seg_2 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(o_seg_3, o1_seg_3)
# print(f"seg_3 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(o_seg_4, o1_seg_4)
# print(f"seg_3 sim: {sim}, l1: {l1}, max_diff: {mdiff}")