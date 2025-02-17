from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_torch
from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle
from flash_attn_interface import flash_attn_func as flash_attn_func_v3

from utils import precision_cmp_torch, precision_cmp

import torch
import paddle
import numpy as np
import os
import argparse
import nvtx

from loguru import logger

tensor_layout = "NHD" # NHD
is_causal = True
return_lse = False

parser = argparse.ArgumentParser(description='Bench mark for FA3')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
args = parser.parse_args()

WARMUP_NUM = 5
REPEAT_NUM = 100

if __name__ == '__main__':
    bsz = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    seq_len = args.seq_len

    print("======= Generating qkv & o =======")
    q = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16, device='cuda').to(torch.float8_e4m3fn)
    k = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16, device='cuda').to(torch.float8_e4m3fn)
    v = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16, device='cuda').to(torch.float8_e4m3fn)
    descale_q = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    descale_k = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    descale_v = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    head_dim_og = q.size(-1)
    sm_scale = head_dim_og**-0.5

    # phase 2: test FA3 official
    print("======= Warm up flash attn v3 torch =======")
    for i in range(WARMUP_NUM): o_torch_fa3, _ = flash_attn_func_v3(q, k, v, 1 / head_dim**0.5, causal=is_causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)

    torch.cuda.synchronize()

    print("======= Bench: flash attn v3 torch =======")
    for i in range(REPEAT_NUM): 
        # if i == REPEAT_NUM - 1:
        transformer_nvtx = nvtx.start_range(message="FA3_FP8_torch", color="blue")
        o_torch_fa3, _ = flash_attn_func_v3(q, k, v, 1 / head_dim**0.5, causal=is_causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)
        torch.cuda.synchronize()
        # if i == REPEAT_NUM - 1:
        nvtx.end_range(transformer_nvtx)

    torch.cuda.synchronize()