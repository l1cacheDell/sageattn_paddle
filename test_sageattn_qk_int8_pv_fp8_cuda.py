import paddle
import torch
import numpy as np

from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda as sageattn_qk_int8_pv_fp8_cuda_paddle
from sageattention import sageattn_qk_int8_pv_fp8_cuda as sageattn_qk_int8_pv_fp8_cuda_torch

from utils import precision_cmp

bsz = 2
seq_len = 1375
head_num = 24
head_dim = 64

BLKQ=128
BLKK=64
WARPQ=32
WARPK=64
_qk_quant_gran = 3
tensor_layout = "NHD" # NHD
is_casual = False
return_lse = False

# torch
q_t = torch.randn((bsz, seq_len, head_num, head_dim), dtype=torch.float16).cuda()
k_t = torch.randn((bsz, seq_len, head_num, head_dim), dtype=torch.float16).cuda()
v_t = torch.randn((bsz, seq_len, head_num, head_dim), dtype=torch.float16).cuda()

head_dim_og = q_t.size(-1)
sm_scale = head_dim_og**-0.5

o_t = sageattn_qk_int8_pv_fp8_cuda_torch(q_t, k_t, v_t, tensor_layout=tensor_layout, is_casual=is_casual, return_lse=return_lse, pv_accum_dtype="fp32+fp32")

torch.cuda.synchronize()

# paddle
q_npy = q_t.cpu().numpy()
k_npy = k_t.cpu().numpy()
v_npy = v_t.cpu().numpy()

# q_p = paddle.to_tensor(paddle.randint(-95, 95, shape=(bsz, seq_len, head_num, head_dim), dtype='int32'), dtype='int8')
# k_p = paddle.to_tensor(paddle.randint(-95, 95, shape=(bsz, seq_len, head_num, head_dim), dtype='int32'), dtype='int8')
# v_p = paddle.empty(shape=(bsz, seq_len, head_num, head_dim), dtype='float16')
# o_p = paddle.empty(shape=(bsz, seq_len, head_num, head_dim), dtype='float16')
# q_scale_p = paddle.empty(shape=(bsz, head_num, (seq_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), dtype='float32')
# k_scale_p = paddle.empty(shape=(bsz, head_num, (seq_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), dtype='float32')

q_p = paddle.to_tensor(q_npy, dtype='float16')
k_p = paddle.to_tensor(k_npy, dtype='float16')
v_p = paddle.to_tensor(v_npy, dtype='float16')

o_p = sageattn_qk_int8_pv_fp8_cuda_paddle(q_p, k_p, v_p, tensor_layout=tensor_layout, is_casual=is_casual, return_lse=return_lse, pv_accum_dtype="fp32+fp32")


paddle.device.synchronize()

sim, l1 = precision_cmp(o_t, o_p)

print(sim, l1)
