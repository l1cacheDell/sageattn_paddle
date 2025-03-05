import paddle
import torch
from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle
from flash_attn_interface import flash_attn_func as flash_attn_func_v3
from torch.nn.functional import scaled_dot_product_attention as sdpa
from sageattention import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a

from utils import precision_cmp_torch, precision_cmp, precision_cmp_paddle, precision_cmp_s


import paddle
import numpy as np
import os
import argparse
import nvtx

bsz = 2

seq_len = 1024 * 16

num_heads = 16
head_dim_qk = 128
head_dim_v = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False



WARMUP_NUM = 5
REPEAT_NUM = 100

# =========================================================================================
torch.backends.cuda.enable_flash_sdp(True)

# prepare input for torch
q = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
k = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
v = torch.randn((bsz, seq_len, num_heads, head_dim_v), dtype=torch.float16).cuda()

# # permute for sdpa
# q = q.transpose(2, 1)
# k = k.transpose(2, 1)
# v = v.transpose(2, 1)

# o_torch_fa2 = sdpa(q, k, v, is_causal=is_causal)
# torch.cuda.synchronize()

# torch.backends.cuda.enable_flash_sdp(False)
# o_torch_sdpa = sdpa(q, k, v, is_causal=is_causal)
# torch.cuda.synchronize()

# # try sage attn
# q = q.transpose(2, 1)
# k = k.transpose(2, 1)
# v = v.transpose(2, 1)

# =========================================================================================

# for i in range(100):
#     transformer_nvtx = nvtx.start_range(message='torch', color='red')
    
#     o_torch_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
#     torch.cuda.synchronize()
#     nvtx.end_range(transformer_nvtx)

# sim, l1, max_diff = precision_cmp_torch(o_torch_fa2, o_torch_sa.transpose(2, 1))
# # print((o_torch_fa2 - o_torch_sa.transpose(2, 1))[0, 0, 0, :50])
# print(f"Torch SA & torch sdpa: {sim}, {max_diff}")

print("======= Warm up: flash attn v3 torch =======")
for i in range(WARMUP_NUM): 
    o_torch_fa3, _ = flash_attn_func_v3(q, k, v, causal=is_causal)

print("======= Bench: flash attn v3 torch =======")
for i in range(REPEAT_NUM): 
    # if i == REPEAT_NUM - 1:
    transformer_nvtx = nvtx.start_range(message="FA3_torch", color="red")
    o_torch_fa3, _ = flash_attn_func_v3(q, k, v, causal=is_causal)
    torch.cuda.synchronize()
    # if i == REPEAT_NUM - 1:
    nvtx.end_range(transformer_nvtx)

torch.cuda.synchronize()

print("======= Warm up flash attn v3 FP8 torch =======")
q2 = q.to(dtype=torch.float8_e4m3fn)
k2 = k.to(dtype=torch.float8_e4m3fn)
v2 = v.to(dtype=torch.float8_e4m3fn)
descale_q = torch.tensor([1.0], dtype=torch.float32, device="cuda")
descale_k = torch.tensor([1.0], dtype=torch.float32, device="cuda")
descale_v = torch.tensor([1.0], dtype=torch.float32, device="cuda")
for i in range(WARMUP_NUM): o_torch_fa3_fp8, _ = flash_attn_func_v3(q2, k2, v2, 1 / head_dim_qk**0.5, causal=is_causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)

torch.cuda.synchronize()

print("======= Bench: flash attn v3 FP8 torch =======")
for i in range(REPEAT_NUM): 
    # if i == REPEAT_NUM - 1:
    transformer_nvtx = nvtx.start_range(message="FA3_FP8_torch", color="blue")
    o_torch_fa3_fp8, _ = flash_attn_func_v3(q, k, v, 1 / head_dim_qk**0.5, causal=is_causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)
    torch.cuda.synchronize()
    # if i == REPEAT_NUM - 1:
    nvtx.end_range(transformer_nvtx)

torch.cuda.synchronize()


bsz = 2
num_heads = 16
head_dim_qk = 256
head_dim_v = 256

tensor_layout = "NHD"
is_causal = True
return_lse = False



WARMUP_NUM = 5
REPEAT_NUM = 100

# =========================================================================================

# prepare input for torch
q = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
k = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
v = torch.randn((bsz, seq_len, num_heads, head_dim_v), dtype=torch.float16).cuda()

# # permute for sdpa
# q = q.transpose(2, 1)
# k = k.transpose(2, 1)
# v = v.transpose(2, 1)

# o_torch_fa2 = sdpa(q, k, v, is_causal=is_causal)
# torch.cuda.synchronize()

# torch.backends.cuda.enable_flash_sdp(False)
# o_torch_sdpa = sdpa(q, k, v, is_causal=is_causal)
# torch.cuda.synchronize()

# # try sage attn
# q = q.transpose(2, 1)
# k = k.transpose(2, 1)
# v = v.transpose(2, 1)

# =========================================================================================

# for i in range(100):
#     transformer_nvtx = nvtx.start_range(message='torch', color='red')
    
#     o_torch_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
#     torch.cuda.synchronize()
#     nvtx.end_range(transformer_nvtx)

# sim, l1, max_diff = precision_cmp_torch(o_torch_fa2, o_torch_sa.transpose(2, 1))
# # print((o_torch_fa2 - o_torch_sa.transpose(2, 1))[0, 0, 0, :50])
# print(f"Torch SA & torch sdpa: {sim}, {max_diff}")

print("======= Warm up: flash attn v3 torch =======")
for i in range(WARMUP_NUM): 
    o_torch_fa3, _ = flash_attn_func_v3(q, k, v, causal=is_causal)

print("======= Bench: flash attn v3 torch =======")
for i in range(REPEAT_NUM): 
    # if i == REPEAT_NUM - 1:
    transformer_nvtx = nvtx.start_range(message="FA3_torch", color="red")
    o_torch_fa3, _ = flash_attn_func_v3(q, k, v, causal=is_causal)
    torch.cuda.synchronize()
    # if i == REPEAT_NUM - 1:
    nvtx.end_range(transformer_nvtx)

torch.cuda.synchronize()

print("======= Warm up flash attn v3 FP8 torch =======")
q2 = q.to(dtype=torch.float8_e4m3fn)
k2 = k.to(dtype=torch.float8_e4m3fn)
v2 = v.to(dtype=torch.float8_e4m3fn)
descale_q = torch.tensor([1.0], dtype=torch.float32, device="cuda")
descale_k = torch.tensor([1.0], dtype=torch.float32, device="cuda")
descale_v = torch.tensor([1.0], dtype=torch.float32, device="cuda")
for i in range(WARMUP_NUM): o_torch_fa3_fp8, _ = flash_attn_func_v3(q2, k2, v2, 1 / head_dim_qk**0.5, causal=is_causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)

torch.cuda.synchronize()

print("======= Bench: flash attn v3 FP8 torch =======")
for i in range(REPEAT_NUM): 
    # if i == REPEAT_NUM - 1:
    transformer_nvtx = nvtx.start_range(message="FA3_FP8_torch", color="blue")
    o_torch_fa3_fp8, _ = flash_attn_func_v3(q, k, v, 1 / head_dim_qk**0.5, causal=is_causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)
    torch.cuda.synchronize()
    # if i == REPEAT_NUM - 1:
    nvtx.end_range(transformer_nvtx)

torch.cuda.synchronize()
