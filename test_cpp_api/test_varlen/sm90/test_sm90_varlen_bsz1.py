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

seq_len = 1024
num_heads = 8
head_dim = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

q = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)
k = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)
v = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)

cu_seqlens = paddle.to_tensor([0, seq_len], dtype=paddle.int32)

segment_lengths = paddle.concat([cu_seqlens[:1], cu_seqlens[1:] - cu_seqlens[:-1]])[1:]
segment_ids = paddle.concat([paddle.full([length], i, dtype='int32') for i, length in enumerate(segment_lengths)])
max_seqlen = seq_len

cu_seqlens_v = paddle.to_tensor([0, seq_len], dtype=paddle.int32)

padded_v, new_cu_seqlen_v = pad_sequences_to_aligned_chunks(v, cu_seqlens_v, 128)

# sm90 kernel
o1, _, _ = sageattn_custom_ops.sage_attention_varlen(q, 
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

q = q.unsqueeze(0)
k = k.unsqueeze(0)
v = v.unsqueeze(0)

km = paddle.mean(k, axis=1, keepdim=True)
km = km.squeeze(1) if tensor_layout == "NHD" else km.squeeze(2)


o_set_1, q_int8_1, k_int8_1 = sageattn_custom_ops.sage_attention(q, k, v, km, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)

o_set_1 = paddle.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, True, False)

o2 = o_set_1.squeeze(0)

print(o2.shape)
print(o1.shape)

sim, l1, max_diff = precision_cmp_paddle(o1, o2)
print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}")

# nan_mask = paddle.isnan(o1)
# nan_indices = paddle.nonzero(nan_mask)
# print(nan_indices)

# print(o1)