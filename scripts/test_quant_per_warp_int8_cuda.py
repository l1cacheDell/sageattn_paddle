import paddle
import torch
import numpy as np
from sageattn_custom_ops import quant_per_warp_int8_cuda as quant_per_warp_int8_cuda_paddle
from phi_sageattention import quant_per_warp_int8_cuda as quant_per_warp_int8_cuda_torch

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
tensor_layout = 0 # NHD
is_casual = 0
return_lse = 0

# torch
q_t = torch.randn(size=(bsz, seq_len, head_num, head_dim), dtype=torch.float16).cuda()
q_int8_t = torch.empty(q_t.shape, dtype=torch.int8, device=q_t.device)
q_scale = torch.empty((bsz, head_num, ((seq_len + 127) // 128) * (128 // 32)), device=q_t.device, dtype=torch.float32)


head_dim_og = q_t.size(-1)
sm_scale = head_dim_og**-0.5

quant_per_warp_int8_cuda_torch(q_t, q_int8_t, q_scale, tensor_layout)

torch.cuda.synchronize()

# paddle
q_npy = q_t.cpu().numpy()
# q_int8_t_npy = q_int8_t.cpu().numpy()
# q_scale_npy = q_scale.cpu().numpy()

# q_p = paddle.to_tensor(paddle.randint(-95, 95, shape=(bsz, seq_len, head_num, head_dim), dtype='int32'), dtype='int8')
# k_p = paddle.to_tensor(paddle.randint(-95, 95, shape=(bsz, seq_len, head_num, head_dim), dtype='int32'), dtype='int8')
# v_p = paddle.empty(shape=(bsz, seq_len, head_num, head_dim), dtype='float16')
# o_p = paddle.empty(shape=(bsz, seq_len, head_num, head_dim), dtype='float16')
# q_scale_p = paddle.empty(shape=(bsz, head_num, (seq_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), dtype='float32')
# k_scale_p = paddle.empty(shape=(bsz, head_num, (seq_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), dtype='float32')

q_p = paddle.to_tensor(q_npy, dtype='float16')
# q_int8_t_p = paddle.to_tensor(q_int8_t_npy, dtype='int8')
# q_scale_p = paddle.to_tensor(q_scale_npy, dtype='float32')
q_int8_t_p = paddle.to_tensor(paddle.empty(shape=q_p.shape, dtype='int32'), dtype='int8')
q_scale_p = paddle.empty(shape=(bsz, head_num, ((seq_len + 127) // 128) * (128 // 32)), dtype='float32')
# print(q_scale_p.dim())

lse_p = quant_per_warp_int8_cuda_paddle(q_p, q_int8_t_p, q_scale_p, tensor_layout)


paddle.device.synchronize()

sim, l1 = precision_cmp(q_int8_t, q_int8_t_p)

print(sim, l1)
