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
    q = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
    k = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
    v = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()

    head_dim_og = q.size(-1)
    sm_scale = head_dim_og**-0.5

    # phase 1: test SA official as baseline
    print("======= Warm up sage attn torch =======")
    for i in range(WARMUP_NUM): o_torch_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_torch(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")

    torch.cuda.synchronize()

    print("======= Bench: sage attn torch =======")
    for i in range(REPEAT_NUM): 
        # if i == REPEAT_NUM - 1:
        transformer_nvtx = nvtx.start_range(message="SA_torch", color="red")
        o_torch_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_torch(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
        torch.cuda.synchronize()
        # if i == REPEAT_NUM - 1:
        nvtx.end_range(transformer_nvtx)

    torch.cuda.synchronize()

    # phase 2: test FA3 official
    print("======= Warm up flash attn v3 torch =======")
    for i in range(WARMUP_NUM): o_torch_fa3, _ = flash_attn_func_v3(q, k, v, causal=is_causal)

    torch.cuda.synchronize()

    print("======= Bench: flash attn v3 torch =======")
    for i in range(REPEAT_NUM): 
        # if i == REPEAT_NUM - 1:
        transformer_nvtx = nvtx.start_range(message="FA3_torch", color="blue")
        o_torch_fa3, _ = flash_attn_func_v3(q, k, v, causal=is_causal)
        torch.cuda.synchronize()
        # if i == REPEAT_NUM - 1:
        nvtx.end_range(transformer_nvtx)

    torch.cuda.synchronize()

    sim_from_paddle_sa, l1_from_paddle_sa, max_diff = precision_cmp_torch(o_torch_sa, o_torch_fa3)
    logger.debug(f"Torch SA Cos sim: {sim_from_paddle_sa}, L1: {l1_from_paddle_sa}, Max Diff: {max_diff.item()}")

    # save for paddle, torch fp16
    q_npy = q.cpu().numpy()
    k_npy = k.cpu().numpy()
    v_npy = v.cpu().numpy()

    o_npy = o_torch_sa.cpu().numpy()

    # phase 3: test FA3 FP8 official
    print("======= Warm up flash attn v3 FP8 torch =======")
    q = q.to(dtype=torch.float8_e4m3fn)
    k = k.to(dtype=torch.float8_e4m3fn)
    v = v.to(dtype=torch.float8_e4m3fn)
    descale_q = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    descale_k = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    descale_v = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    for i in range(WARMUP_NUM): o_torch_fa3_fp8, _ = flash_attn_func_v3(q, k, v, 1 / head_dim**0.5, causal=is_causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)

    torch.cuda.synchronize()

    print("======= Bench: flash attn v3 FP8 torch =======")
    for i in range(REPEAT_NUM): 
        # if i == REPEAT_NUM - 1:
        transformer_nvtx = nvtx.start_range(message="FA3_FP8_torch", color="blue")
        o_torch_fa3_fp8, _ = flash_attn_func_v3(q, k, v, 1 / head_dim**0.5, causal=is_causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)
        torch.cuda.synchronize()
        # if i == REPEAT_NUM - 1:
        nvtx.end_range(transformer_nvtx)

    torch.cuda.synchronize()

    # phase 4: test SA paddle impl


    q_paddle = paddle.to_tensor(q_npy, dtype=paddle.float16)
    k_paddle = paddle.to_tensor(k_npy, dtype=paddle.float16)
    v_paddle = paddle.to_tensor(v_npy, dtype=paddle.float16)
    o_paddle = paddle.to_tensor(o_npy, dtype=paddle.float16)

    print("======= Warm up sage attn paddle =======")
    for i in range(WARMUP_NUM): o_paddle_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle(q_paddle, k_paddle, v_paddle, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")

    paddle.device.synchronize()

    print("======= Bench: sage attn paddle =======")
    for i in range(REPEAT_NUM): 
        # if i == REPEAT_NUM - 1:
        transformer_nvtx = nvtx.start_range(message="SA_paddle", color="green")
        o_paddle_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle(q_paddle, k_paddle, v_paddle, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
        paddle.device.synchronize()
        # if i == REPEAT_NUM - 1:
        nvtx.end_range(transformer_nvtx)

    paddle.device.synchronize()

    sim_from_paddle_sa, l1_from_paddle_sa, max_diff = precision_cmp(o_torch_fa3, o_paddle_sa)
    logger.debug(f"Paddle SA Cos sim: {sim_from_paddle_sa}, L1: {l1_from_paddle_sa}, Max Diff: {max_diff.item()}")

    # save it for phase 4: append_attn
    input_dir = './inputs'

    o_npy = o_torch_fa3.cpu().numpy()

    np.save(os.path.join(input_dir, "q.npy"), q_npy.astype(np.float16))
    np.save(os.path.join(input_dir, "k.npy"), k_npy.astype(np.float16))
    np.save(os.path.join(input_dir, "v.npy"), v_npy.astype(np.float16))
    np.save(os.path.join(input_dir, "o.npy"), o_npy.astype(np.float16))