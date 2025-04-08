import sageattn_custom_ops
import paddle
import numpy as np

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

bsz = 2
seq_len = 1408
num_heads = 24
head_dim = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

q = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)
k = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)
v = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)

cu_seqlens = paddle.to_tensor([0, 256, 384, seq_len], dtype=paddle.int32)

segment_lengths = paddle.concat([cu_seqlens[:1], cu_seqlens[1:] - cu_seqlens[:-1]])[1:]
segment_ids = paddle.concat([paddle.full([length], i, dtype='int32') for i, length in enumerate(segment_lengths)])
max_seqlen = seq_len - 384

cu_seqlens_v = paddle.to_tensor([0, 256, 384, seq_len], dtype=paddle.int32)

padded_v, new_cu_seqlen_v = pad_sequences_to_aligned_chunks(v, cu_seqlens_v, 128)

# sm90 kernel
o1, vfp8_fused, v_transposed_fused = sageattn_custom_ops.sage_attention_varlen(q, 
                                                k, 
                                                padded_v, 
                                                cu_seqlens,
                                                cu_seqlens_v,
                                                new_cu_seqlen_v,
                                                segment_ids,
                                                None,
                                                max_seqlen,
                                                max_seqlen,
                                                max_seqlen,
                                                seq_len,
                                                head_dim**-0.5,
                                                "per_warp",
                                                "fp16",
                                                tensor_layout=0,
                                                is_causal=is_causal,
                                                smooth_k=True, 
                                                smooth_v=False, 
                                                return_lse=return_lse)

paddle.device.synchronize()

# =======================================================================
q1, q2, q3 = paddle.split(q, [256 - 0, 384 - 256, seq_len - 384], axis=0)
k1, k2, k3 = paddle.split(k, [256 - 0, 384 - 256, seq_len - 384], axis=0)
v1, v2, v3 = paddle.split(v, [256 - 0, 384 - 256, seq_len - 384], axis=0)

q1 = paddle.unsqueeze(q1, axis=0)
k1 = paddle.unsqueeze(k1, axis=0)
v1 = paddle.unsqueeze(v1, axis=0)

q2 = paddle.unsqueeze(q2, axis=0)
k2 = paddle.unsqueeze(k2, axis=0)
v2 = paddle.unsqueeze(v2, axis=0)

q3 = paddle.unsqueeze(q3, axis=0)
k3 = paddle.unsqueeze(k3, axis=0)
v3 = paddle.unsqueeze(v3, axis=0)

km1 = paddle.mean(k1, axis=1, keepdim=True)
km1 = km1.squeeze(1) if tensor_layout == "NHD" else km1.squeeze(2)

km2 = paddle.mean(k2, axis=1, keepdim=True)
km2 = km2.squeeze(1) if tensor_layout == "NHD" else km2.squeeze(2)

km3 = paddle.mean(k3, axis=1, keepdim=True)
km3 = km3.squeeze(1) if tensor_layout == "NHD" else km3.squeeze(2)

o_set_1, vfp8_1, v_tm_1 = sageattn_custom_ops.sage_attention(q1, k1, v1, km1, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)
o_set_2, vfp8_2, v_tm_2 = sageattn_custom_ops.sage_attention(q2, k2, v2, km2, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)
o_set_3, vfp8_3, v_tm_3 = sageattn_custom_ops.sage_attention(q3, k3, v3, km3, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)

km_total = paddle.concat([km1, km2, km3], axis=0)
# print(vfp8_1.shape)
# print(vfp8_2.shape)
# print(vfp8_3.shape)

print("\n====== Compare v quant =======\n")

vfp8_varlen_1, vfp8_varlen_2, vfp8_varlen_3 = paddle.split(v_transposed_fused.astype(paddle.float32), [256 - 0, 384 - 256, seq_len - 384], axis=-1)
sim, l1, md = precision_cmp_paddle(v_tm_1, vfp8_varlen_1)
print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
sim, l1, md = precision_cmp_paddle(v_tm_2, vfp8_varlen_2)
print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
sim, l1, md = precision_cmp_paddle(v_tm_3, vfp8_varlen_3)
print(f"sim: {sim}, l1: {l1}, max_diff: {md}")

# 就是vfp8有问题，transposed都没问题
vfp8_varlen_1, vfp8_varlen_2, vfp8_varlen_3 = paddle.split(vfp8_fused.astype(paddle.float32), [256 - 0, 384 - 256, seq_len - 384], axis=-1)
sim, l1, md = precision_cmp_paddle(vfp8_1, vfp8_varlen_1)
print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
sim, l1, md = precision_cmp_paddle(vfp8_2, vfp8_varlen_2)
print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
sim, l1, md = precision_cmp_paddle(vfp8_3, vfp8_varlen_3)
print(f"sim: {sim}, l1: {l1}, max_diff: {md}")


# print("\n=================================\n")
# print(vfp8_1.shape)

# sim, l1, md = precision_cmp_paddle(vfp8_1[:, :, :, :32].astype(paddle.float32), vfp8_varlen_1[:, :, :32].astype(paddle.float32))
# print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
# sim, l1, md = precision_cmp_paddle(vfp8_1[:, :, :, 32:64].astype(paddle.float32), vfp8_varlen_1[:, :, 32:64].astype(paddle.float32))
# print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
# sim, l1, md = precision_cmp_paddle(vfp8_1[:, :, :, 64:96].astype(paddle.float32), vfp8_varlen_1[:, :, 64:96].astype(paddle.float32))
# print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
# sim, l1, md = precision_cmp_paddle(vfp8_1[:, :, :, 96:128].astype(paddle.float32), vfp8_varlen_1[:, :, 96:128].astype(paddle.float32))
# print(f"sim: {sim}, l1: {l1}, max_diff: {md}")


# 这个居然是没问题的
# vfp8_varlen_1, vfp8_varlen_2, vfp8_varlen_3 = paddle.split(v_transposed_fused.astype(paddle.float32), [256 - 0, 384 - 256, seq_len - 384], axis=-1)
# sim, l1, md = precision_cmp_paddle(v_tm_1, vfp8_varlen_1)
# print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
# sim, l1, md = precision_cmp_paddle(v_tm_2, vfp8_varlen_2)
# print(f"sim: {sim}, l1: {l1}, max_diff: {md}")
# sim, l1, md = precision_cmp_paddle(v_tm_3, vfp8_varlen_3)
# print(f"sim: {sim}, l1: {l1}, max_diff: {md}")

# o_set_1 = paddle.nn.functional.scaled_dot_product_attention(q1, k1, v1, None, 0.0, True, False)
# o_set_2 = paddle.nn.functional.scaled_dot_product_attention(q2, k2, v2, None, 0.0, True, False)
# o_set_3 = paddle.nn.functional.scaled_dot_product_attention(q3, k3, v3, None, 0.0, True, False)

o2 = paddle.concat([o_set_1, o_set_2, o_set_3], axis=1).squeeze(0)

print(o2.shape)
print(o1.shape)
# print(o1)
# print(q_int8)

sim, l1, max_diff = precision_cmp_paddle(o1, o2)
print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}")

# print("\n========================== Output ==========================\n")

# # compare three segment each
# o1_seg_1, o1_seg_2, o1_seg_3 = paddle.split(o1, [256 - 0, 384 - 256, seq_len - 384], axis=0)
# sim, l1, mdiff = precision_cmp_paddle(o1_seg_1.unsqueeze(0), o_set_1)
# print(f"seg_1 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(o1_seg_2.unsqueeze(0), o_set_2)
# print(f"seg_2 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# sim, l1, mdiff = precision_cmp_paddle(o1_seg_3.unsqueeze(0), o_set_3)
# print(f"seg_3 sim: {sim}, l1: {l1}, max_diff: {mdiff}")

# # 沿着num_head维度切分一下
# sim, l1, max_diff = precision_cmp_paddle(o1[:, :8, :], o2[:, :8, :])
# print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}") # 0.593
# sim, l1, max_diff = precision_cmp_paddle(o1[:, 8:16, :], o2[:, 8:16, :])
# print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}") # 0.637
# sim, l1, max_diff = precision_cmp_paddle(o1[:, 16:, :], o2[:, 16:, :])
# print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}") # 0.693
# print()

# sim, l1, max_diff = precision_cmp_paddle(o1[:, 16:18, :], o2[:, 16:18, :])
# print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}") # 0.693
# sim, l1, max_diff = precision_cmp_paddle(o1[:, 22:, :], o2[:, 22:, :])
# print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}") # 0.693