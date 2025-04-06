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


seq_len = 1027
num_heads = 24
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
max_seqlen = seq_len - 0

# sm80 kernel
o1, q_int8, k_int8, km_out = sageattn_custom_ops.sage_attention_varlen(q, 
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
paddle.device.synchronize()
# =======================================================================

q = q.unsqueeze(0)
k = k.unsqueeze(0)
v = v.unsqueeze(0)

km = paddle.mean(k, axis=1, keepdim=True)
km = km.squeeze(1) if tensor_layout == "NHD" else km.squeeze(2)


o_set_1, q_int8_1, k_int8_1 = sageattn_custom_ops.sage_attention(q, k, v, km, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)

# compare kmean
km_total = paddle.concat([km], axis=0)
print("max km: ", paddle.max(km_out - km_total))

# o_set_1 = paddle.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, True, False)

o2 = o_set_1.squeeze(0)

# print(o2.shape)
# print(o1.shape)

# compare quant results
sim_q_int8_1, _, max_diff_q_int8_1 = precision_cmp_paddle(q_int8_1.squeeze(0), q_int8)

print(f"sim_q_int8_1: {sim_q_int8_1}, max_diff_q_int8_1: {max_diff_q_int8_1}")

sim_k_int8_1, _, max_diff_k_int8_1 = precision_cmp_paddle(k_int8_1.squeeze(0), k_int8)

print(f"sim_k_int8_1: {sim_k_int8_1}, max_diff_k_int8_1: {max_diff_k_int8_1}")

diff_mat = q_int8_1.squeeze(0).astype("int32") - q_int8.astype(paddle.int32)

non_zero_indices = paddle.nonzero(diff_mat != 0)  # 形状为 [N, rank]，N 是非零元素数量
print(non_zero_indices.shape)
np.savetxt("matq.txt", non_zero_indices.cpu().numpy(), fmt='%d')

diff_mat = k_int8_1.squeeze(0).astype("int32") - k_int8.astype(paddle.int32)

non_zero_indices = paddle.nonzero(diff_mat != 0)  # 形状为 [N, rank]，N 是非零元素数量
print(non_zero_indices.shape)
np.savetxt("matk.txt", non_zero_indices.cpu().numpy(), fmt='%d')


sim, l1, max_diff = precision_cmp_paddle(o1, o2)
print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}")

nan_mask = paddle.isnan(o1)
nan_indices = paddle.nonzero(nan_mask)
print(nan_indices)

# print(o1)