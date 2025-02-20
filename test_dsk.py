import paddle
from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle

from torch.nn.functional import scaled_dot_product_attention as sdpa
from sageattention import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a

from utils import precision_cmp_torch, precision_cmp, precision_cmp_paddle, precision_cmp_s

import torch
import paddle
import numpy as np
import os
import argparse
# import nvtx

bsz = 2
seq_len = 1026
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
o_torch_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
torch.cuda.synchronize()

sim, l1, max_diff = precision_cmp_torch(o_torch_fa2, o_torch_sa.transpose(2, 1))
# print((o_torch_fa2 - o_torch_sa.transpose(2, 1))[0, 0, 0, :50])
print(f"Torch SA & torch sdpa: {sim}, {max_diff}")

q_npy = q.cpu().numpy()
k_npy = k.cpu().numpy()
v_npy = v.cpu().numpy()

o_npy = o_torch_sdpa.cpu().numpy()
o_torch_sa_npy = o_torch_sa.cpu().numpy()

q_paddle = paddle.to_tensor(q_npy, dtype=paddle.float16)
# q_paddle = paddle.transpose(q_paddle, [0, 2, 1, 3])
k_paddle = paddle.to_tensor(k_npy, dtype=paddle.float16)
# k_paddle = paddle.transpose(k_paddle, [0, 2, 1, 3])
v_paddle = paddle.to_tensor(v_npy, dtype=paddle.float16)
# v_paddle = paddle.transpose(v_paddle, [0, 2, 1, 3])
o_paddle = paddle.to_tensor(o_npy, dtype=paddle.float16)
o_paddle = paddle.transpose(o_paddle, [0, 2, 1, 3])
o_torch_sa_paddle = paddle.to_tensor(o_torch_sa_npy, dtype=paddle.float16)

head_dim_og = head_dim_qk
sm_scale = head_dim_og**-0.5

o_paddle_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle(q_paddle, k_paddle, v_paddle, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
paddle.device.synchronize()

# q_paddle = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_v), dtype=paddle.float16)
# k_paddle = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_v), dtype=paddle.float16)
# v_paddle = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_v), dtype=paddle.float16)
# o_from_paddle_sdpa = paddle.nn.functional.scaled_dot_product_attention(q_paddle, k_paddle, v_paddle, attn_mask=None, dropout_p=0.0, is_causal=is_causal, training=False)
# paddle.device.synchronize()

sim, l1, max_diff = precision_cmp_paddle(o_paddle, o_paddle_sa)
print(f"paddle sa {sim}, {max_diff}")

print("==== diff quant")