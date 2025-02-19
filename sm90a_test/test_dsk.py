import paddle
from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle
from flash_attn_interface import flash_attn_func as flash_attn_func_v3

from utils import precision_cmp_torch, precision_cmp

import torch
import paddle
import numpy as np
import os
import argparse
import nvtx

bsz = 2
seq_len = 1026
num_heads = 128
head_dim_qk = 128 + 64
head_dim_v = 128

tensor_layout = "NHD"
is_casual = True
return_lse = False

q = paddle.randn([bsz, seq_len, num_heads, head_dim_qk], dtype=paddle.float16)
k = paddle.randn([bsz, seq_len, num_heads, head_dim_qk], dtype=paddle.float16)
v = paddle.randn([bsz, seq_len, num_heads, head_dim_v], dtype=paddle.float16)

head_dim_og = head_dim_qk
sm_scale = head_dim_og**-0.5

o_paddle_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle(q, k, v, tensor_layout=tensor_layout, is_causal=is_casual, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
paddle.device.synchronize()