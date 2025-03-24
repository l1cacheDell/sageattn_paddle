import paddle
from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle

from torch.nn.functional import scaled_dot_product_attention as sdpa
import sageattn_custom_ops

import torch
import paddle
import numpy as np
import os
import argparse
import nvtx

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
head_dim_qk = 128 + 64
head_dim_v = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

torch.backends.cuda.enable_flash_sdp(True)

# prepare input for torch
q = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
k = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
v = torch.randn((bsz, seq_len, num_heads, head_dim_v), dtype=torch.float16).cuda()

# permute for sdpa
q = q.transpose(2, 1)
k = k.transpose(2, 1)
v = v.transpose(2, 1)

o_torch_fa2 = sdpa(q, k, v, is_causal=is_causal)
torch.cuda.synchronize()

torch.backends.cuda.enable_flash_sdp(False)
o_torch_sdpa = sdpa(q, k, v, is_causal=is_causal)
torch.cuda.synchronize()

# try sage attn
q = q.transpose(2, 1)
k = k.transpose(2, 1)
v = v.transpose(2, 1)

q_npy = q.cpu().numpy()
k_npy = k.cpu().numpy()
v_npy = v.cpu().numpy()

o_npy = o_torch_sdpa.cpu().numpy()

q_paddle = paddle.to_tensor(q_npy, dtype=paddle.float16, place=paddle.CUDAPlace(0))
k_paddle = paddle.to_tensor(k_npy, dtype=paddle.float16, place=paddle.CUDAPlace(0))
v_paddle = paddle.to_tensor(v_npy, dtype=paddle.float16, place=paddle.CUDAPlace(0))
o_paddle = paddle.to_tensor(o_npy, dtype=paddle.float16)
o_paddle = paddle.transpose(o_paddle, [0, 2, 1, 3])

head_dim_og = head_dim_qk
sm_scale = head_dim_og**-0.5

tensor_layout = "NHD"
is_causal = True
return_lse = False

q = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_qk), dtype=paddle.float16)
k = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_qk), dtype=paddle.float16)
v = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_v), dtype=paddle.float16)

o2 = paddle.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

# o3 = sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle(
#     q, k, v, "NHD", is_causal
# )

q = paddle.nn.functional.pad(q_paddle, (0, 256 - head_dim_qk))
k = paddle.nn.functional.pad(k_paddle, (0, 256 - head_dim_qk))
v = v_paddle

km = paddle.mean(k, axis=1, keepdim=True)
km = km.squeeze(1) if tensor_layout == "NHD" else km.squeeze(2)

# remember do padding to v!
v_pad_len = 128 - (seq_len % 128) if seq_len % 128 != 0 else 0
if v_pad_len > 0:
    if tensor_layout == "HND":
        v = paddle.concat([v, paddle.zeros(v.shape[0], v.shape[1], v_pad_len, v.shape[3], dtype=v.dtype, device=v.device)], dim=2)
    else:
        v = paddle.concat([v, paddle.zeros(v.shape[0], v_pad_len, v.shape[2], v.shape[3], dtype=v.dtype, device=v.device)], dim=1)

# sm90 kernel
o1 = sageattn_custom_ops.sage_attention_dsk(q, 
                                        k, 
                                        v, 
                                        km, 
                                        None,
                                        head_dim_qk**-0.5,
                                        "per_warp",
                                        "Whatever",
                                        tensor_layout=0, 
                                        is_causal=is_causal, 
                                        smooth_k=True, 
                                        smooth_v=False, 
                                        return_lse=return_lse)



sim, l1, max_diff = precision_cmp_paddle(o_paddle, o1)
print(f"sim: {sim}, l1: {l1}, max_diff: {max_diff}")