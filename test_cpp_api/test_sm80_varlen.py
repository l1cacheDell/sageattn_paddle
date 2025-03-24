import sageattn_custom_ops
import paddle

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

# def create_tensor(bsz: int):
#     tensors = []
#     num_heads = 24
#     head_dim = 128

#     cu_seqlens = [0]

#     for i in range(bsz):
#         tensors.append(paddle.randn([i + 1, i + 1], dtype=paddle.float32))
#     return paddle.stack(tensors)

bsz = 2
seq_len = 1048
num_heads = 24
head_dim = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

q = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)
k = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)
v = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)

cu_seqlens = paddle.to_tensor([0, 246, 394, 1048], dtype=paddle.int32)

segment_lengths = paddle.concat([cu_seqlens[:1], cu_seqlens[1:] - cu_seqlens[:-1]])[1:]
segment_ids = paddle.concat([paddle.full([length], i, dtype='int32') for i, length in enumerate(segment_lengths)])
max_seqlen = 1048 - 394

# sm80 kernel
o1, q_int8 = sageattn_custom_ops.sage_attention_varlen(q, 
                                                k, 
                                                v, 
                                                cu_seqlens,
                                                segment_ids,
                                                None,
                                                max_seqlen,
                                                max_seqlen,
                                                head_dim**-0.5,
                                                "per_warp",
                                                "fp32",
                                                tensor_layout=0,
                                                is_causal=is_causal,
                                                smooth_k=True, 
                                                smooth_v=False, 
                                                return_lse=return_lse)

# =======================================================================
q1, q2, q3 = paddle.split(q, [246 - 0, 394 - 246, seq_len - 394], axis=0)
k1, k2, k3 = paddle.split(k, [246 - 0, 394 - 246, seq_len - 394], axis=0)
v1, v2, v3 = paddle.split(v, [246 - 0, 394 - 246, seq_len - 394], axis=0)

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

o_set_1 = sageattn_custom_ops.sage_attention(q1, k1, v1, km1, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)
o_set_2 = sageattn_custom_ops.sage_attention(q2, k2, v2, km2, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)
o_set_3 = sageattn_custom_ops.sage_attention(q3, k3, v3, km3, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)

o2 = paddle.concat([o_set_1, o_set_2, o_set_3], axis=1).squeeze(0)

print(o2.shape)
print(o1.shape)
# print(o1)
print(q_int8)
print(paddle.max(o1-o2))

# o2 = paddle.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

# sim, l1, max_diff = precision_cmp_paddle(o1, o2)
# print(f"sim: {sim}, l1: {l1}, max_diff: {max_diff}")