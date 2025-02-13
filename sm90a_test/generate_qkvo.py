from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_torch
from utils import precision_cmp_torch

import torch
import numpy as np
import os
import argparse

tensor_layout = "NHD" # NHD
is_casual = True
return_lse = False

parser = argparse.ArgumentParser(description='Generate qkvo')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
args = parser.parse_args()

if __name__ == '__main__':
    bsz = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    seq_len = args.seq_len

    q = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
    k = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
    v = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()

    head_dim_og = q.size(-1)
    sm_scale = head_dim_og**-0.5

    o = sageattn_qk_int8_pv_fp8_cuda_sm90a_torch(q, k, v, tensor_layout=tensor_layout, is_casual=is_casual, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")

    torch.cuda.synchronize()

    print("======= Generating qkv & o =======")
    for i in range(100): o2 = sageattn_qk_int8_pv_fp8_cuda_sm90a_torch(q, k, v, tensor_layout=tensor_layout, is_casual=is_casual, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")

    torch.cuda.synchronize()

    sim, l1 = precision_cmp_torch(o, o2)
    print(f"sim: {sim}, l1: {l1}")

    input_dir = './inputs'

    q_npy = q.cpu().numpy()
    k_npy = k.cpu().numpy()
    v_npy = v.cpu().numpy()
    o_npy = o2.cpu().numpy()

    np.save(os.path.join(input_dir, "q.npy"), q_npy.astype(np.float16))
    np.save(os.path.join(input_dir, "k.npy"), k_npy.astype(np.float16))
    np.save(os.path.join(input_dir, "v.npy"), v_npy.astype(np.float16))
    np.save(os.path.join(input_dir, "o.npy"), o_npy.astype(np.float16))