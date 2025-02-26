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

parser = argparse.ArgumentParser(description='Bench mark for FA3')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
args = parser.parse_args()

bsz = 2
seq_len = args.seq_len
num_heads = 128
head_dim_qk = 128 + 64
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
    transformer_nvtx = nvtx.start_range(message="FA3_torch", color="blue")
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

q_npy = q.cpu().numpy()
k_npy = k.cpu().numpy()
v_npy = v.cpu().numpy()

o_npy = o_torch_sdpa.cpu().numpy()


q_paddle = paddle.to_tensor(q_npy, dtype=paddle.float16, place=paddle.CUDAPlace(0))
# q_paddle = paddle.transpose(q_paddle, [0, 2, 1, 3])
k_paddle = paddle.to_tensor(k_npy, dtype=paddle.float16, place=paddle.CUDAPlace(0))
# k_paddle = paddle.transpose(k_paddle, [0, 2, 1, 3])
v_paddle = paddle.to_tensor(v_npy, dtype=paddle.float16, place=paddle.CUDAPlace(0))
# v_paddle = paddle.transpose(v_paddle, [0, 2, 1, 3])
o_paddle = paddle.to_tensor(o_npy, dtype=paddle.float16)
o_paddle = paddle.transpose(o_paddle, [0, 2, 1, 3])


head_dim_og = head_dim_qk
sm_scale = head_dim_og**-0.5

for i in range(REPEAT_NUM):
    transformer_nvtx = nvtx.start_range(message='paddle', color='green')
    o_paddle_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle(q_paddle, k_paddle, v_paddle, 
                    tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    paddle.device.synchronize()
    nvtx.end_range(transformer_nvtx)
    # print(6)

# q_paddle = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_v), dtype=paddle.float16)
# k_paddle = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_v), dtype=paddle.float16)
# v_paddle = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim_v), dtype=paddle.float16)
# o_from_paddle_sdpa = paddle.nn.functional.scaled_dot_product_attention(q_paddle, k_paddle, v_paddle, attn_mask=None, dropout_p=0.0, is_causal=is_causal, training=False)
# paddle.device.synchronize()

sim, l1, max_diff = precision_cmp_paddle(o_paddle, o_paddle_sa)
print(f"paddle sa {sim}, {max_diff}")

