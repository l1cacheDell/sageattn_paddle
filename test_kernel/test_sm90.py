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

bsz = 2
seq_len = 1024
num_heads = 128
head_dim = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

q = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)
k = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)
v = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)

km = paddle.mean(k, axis=1, keepdim=True)
km = km.squeeze(1) if tensor_layout == "NHD" else km.squeeze(2)

# remember do padding to v!
v_pad_len = 128 - (seq_len % 128) if seq_len % 128 != 0 else 0
if v_pad_len > 0:
    if tensor_layout == "HND":
        v = paddle.concat([v, paddle.zeros(v.shape[0], v.shape[1], v_pad_len, v.shape[3], dtype=v.dtype, device=v.device)], dim=2)
    else:
        v = paddle.concat([v, paddle.zeros(v.shape[0], v_pad_len, v.shape[2], v.shape[3], dtype=v.dtype, device=v.device)], dim=1)

o1 = sageattn_custom_ops.sage_attention(q, 
                                        k, 
                                        v, 
                                        km, 
                                        None,
                                        head_dim**-0.5,
                                        "per_warp",
                                        "fp16",
                                        tensor_layout=0, 
                                        is_causal=is_causal, 
                                        smooth_k=True, 
                                        smooth_v=False, 
                                        return_lse=return_lse)
o2 = paddle.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

sim, l1, max_diff = precision_cmp_paddle(o1, o2)
print(f"sim: {sim}, l1: {l1}, max_diff: {max_diff}")