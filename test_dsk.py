import paddle
from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle

from torch.nn.functional import scaled_dot_product_attention as sdpa

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

for i in range(2):
    transformer_nvtx = nvtx.start_range(message='paddle', color='green')
    o_paddle_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle(q_paddle, k_paddle, v_paddle,
                    tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    paddle.device.synchronize()
    nvtx.end_range(transformer_nvtx)
    
sim, l1, max_diff = precision_cmp_paddle(o_paddle, o_paddle_sa)
print(f"paddle sa sim: {sim}, diff: {max_diff}")
