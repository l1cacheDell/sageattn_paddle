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

# def create_tensor(bsz: int):
#     tensors = []
#     num_heads = 24
#     head_dim = 128

#     cu_seqlens = [0]

#     for i in range(bsz):
#         tensors.append(paddle.randn([i + 1, i + 1], dtype=paddle.float32))
#     return paddle.stack(tensors)

bsz = 2
seq_len = 1025
num_heads = 24
head_dim = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

q = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)
k = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)
v = paddle.randn(shape=(seq_len, num_heads, head_dim), dtype=paddle.float16)

cu_seqlens = paddle.to_tensor([0, 246, 394, seq_len], dtype=paddle.int32)

segment_lengths = paddle.concat([cu_seqlens[:1], cu_seqlens[1:] - cu_seqlens[:-1]])[1:]
segment_ids = paddle.concat([paddle.full([length], i, dtype='int32') for i, length in enumerate(segment_lengths)])
max_seqlen = seq_len - 394

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

o_set_1, q_int8_1, k_int8_1 = sageattn_custom_ops.sage_attention(q1, k1, v1, km1, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)
o_set_2, q_int8_2, k_int8_2 = sageattn_custom_ops.sage_attention(q2, k2, v2, km2, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)
o_set_3, q_int8_3, k_int8_3 = sageattn_custom_ops.sage_attention(q3, k3, v3, km3, None, head_dim**-0.5, "per_warp", "fp32", tensor_layout=0, is_causal=is_causal, smooth_k=True, smooth_v=False, return_lse=return_lse)

km_total = paddle.concat([km1, km2, km3], axis=0)
print("max km: ", paddle.max(km_out - km_total))

o_set_1 = paddle.nn.functional.scaled_dot_product_attention(q1, k1, v1, None, 0.0, True, False)
o_set_2 = paddle.nn.functional.scaled_dot_product_attention(q2, k2, v2, None, 0.0, True, False)
o_set_3 = paddle.nn.functional.scaled_dot_product_attention(q3, k3, v3, None, 0.0, True, False)

o2 = paddle.concat([o_set_1, o_set_2, o_set_3], axis=1).squeeze(0)

print(o2.shape)
print(o1.shape)
# print(o1)
# print(q_int8)

# compare quant results
q_int8_varlen_1, q_int8_varlen_2, q_int8_varlen_3 = paddle.split(q_int8, [246 - 0, 394 - 246, seq_len - 394], axis=0)
sim_q_int8_1, _, max_diff_q_int8_1 = precision_cmp_paddle(q_int8_1.squeeze(0), q_int8_varlen_1)
sim_q_int8_2, _, max_diff_q_int8_2 = precision_cmp_paddle(q_int8_2.squeeze(0), q_int8_varlen_2)
sim_q_int8_3, _, max_diff_q_int8_3 = precision_cmp_paddle(q_int8_3.squeeze(0), q_int8_varlen_3)
print(f"sim_q_int8_1: {sim_q_int8_1}, max_diff_q_int8_1: {max_diff_q_int8_1}")
print(f"sim_q_int8_2: {sim_q_int8_2}, max_diff_q_int8_2: {max_diff_q_int8_2}")
print(f"sim_q_int8_3: {sim_q_int8_3}, max_diff_q_int8_3: {max_diff_q_int8_3}")


k_int8_varlen_1, k_int8_varlen_2, k_int8_varlen_3 = paddle.split(k_int8, [246 - 0, 394 - 246, seq_len - 394], axis=0)
sim_k_int8_1, _, max_diff_k_int8_1 = precision_cmp_paddle(k_int8_1.squeeze(0), k_int8_varlen_1)
sim_k_int8_2, _, max_diff_k_int8_2 = precision_cmp_paddle(k_int8_2.squeeze(0), k_int8_varlen_2)
sim_k_int8_3, _, max_diff_k_int8_3 = precision_cmp_paddle(k_int8_3.squeeze(0), k_int8_varlen_3)

print(f"sim_k_int8_1: {sim_k_int8_1}, max_diff_k_int8_1: {max_diff_k_int8_1}")
print(f"sim_k_int8_2: {sim_k_int8_2}, max_diff_k_int8_2: {max_diff_k_int8_2}")
print(f"sim_k_int8_3: {sim_k_int8_3}, max_diff_k_int8_3: {max_diff_k_int8_3}")

# print(k_int8_1.squeeze(0).place, k_int8_varlen_1.place)
diff_mat = k_int8_1.squeeze(0).astype("int32") - k_int8_varlen_1.astype(paddle.int32)
# idx = paddle.argmax(diff_mat).item()
# print(f"The seq: {idx // (head_dim * num_heads)} The Head: {(idx % (head_dim * num_heads)) // head_dim}, The dim: {idx % head_dim}")
non_zero_indices = paddle.nonzero(diff_mat != 0)  # 形状为 [N, rank]，N 是非零元素数量
print(non_zero_indices.shape)
# np.savetxt("mat1.txt", non_zero_indices.cpu().numpy(), fmt='%.1f')
# np.savetxt("mat1_val.txt", diff_mat[non_zero_indices].reshape([-1,]).cpu().numpy(), fmt='%.1f')

diff_mat2 = k_int8_2.squeeze(0).astype("int32") - k_int8_varlen_2.astype(paddle.int32)
# idx = paddle.argmax(diff_mat2).item()
# print(f"The seq: {idx // (head_dim * num_heads)} The Head: {(idx % (head_dim * num_heads)) // head_dim}, The dim: {idx % head_dim}")
non_zero_indices = paddle.nonzero(diff_mat2 != 0)  # 形状为 [N, rank]，N 是非零元素数量
print(non_zero_indices.shape)
# np.savetxt("mat2.txt", non_zero_indices.cpu().numpy(), fmt='%.1f')

diff_mat3 = k_int8_3.squeeze(0).astype("int32") - k_int8_varlen_3.astype(paddle.int32)
# idx = paddle.argmax(diff_mat3).item()
# print(f"The seq: {idx // (head_dim * num_heads)} The Head: {(idx % (head_dim * num_heads)) // head_dim}, The dim: {idx % head_dim}")
non_zero_indices = paddle.nonzero(diff_mat3 != 0)  # 形状为 [N, rank]，N 是非零元素数量
print(non_zero_indices.shape)
# np.savetxt("mat3.txt", non_zero_indices.cpu().numpy(), fmt='%.1f')
# print(k_int8_1.squeeze(0).astype("int32") - k_int8_varlen_1.astype(paddle.int32))

# o2 = paddle.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

sim, l1, max_diff = precision_cmp_paddle(o1, o2)
print(f"result sim: {sim}, l1: {l1}, max_diff: {max_diff}")