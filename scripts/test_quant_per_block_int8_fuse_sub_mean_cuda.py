import paddle
import torch
import numpy as np
from sageattn_custom_ops import quant_per_block_int8_fuse_sub_mean_cuda as quant_per_block_int8_fuse_sub_mean_cuda_paddle
from phi_sageattention import quant_per_block_int8_fuse_sub_mean_cuda as quant_per_block_int8_fuse_sub_mean_cuda_torch

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
k_t = torch.randn(size=(bsz, seq_len, head_num, head_dim), dtype=torch.float16).cuda()
k_int8_t = torch.empty(k_t.shape, dtype=torch.int8, device=k_t.device)
k_scale = torch.empty((bsz, head_num, ((seq_len + 63) // 64)), device=k_t.device, dtype=torch.float32)
seq_dim = 1
km = k_t.mean(dim=seq_dim, keepdim=True)
km = km.squeeze(1) if tensor_layout == 0 else km.squeeze(2)

head_dim_og = k_t.size(-1)
sm_scale = head_dim_og**-0.5

quant_per_block_int8_fuse_sub_mean_cuda_torch(k_t, km, k_int8_t, k_scale, 64, tensor_layout)

torch.cuda.synchronize()

# paddle
k_npy = k_t.cpu().numpy()
# q_int8_t_npy = q_int8_t.cpu().numpy()
# q_scale_npy = q_scale.cpu().numpy()

# q_p = paddle.to_tensor(paddle.randint(-95, 95, shape=(bsz, seq_len, head_num, head_dim), dtype='int32'), dtype='int8')
# k_p = paddle.to_tensor(paddle.randint(-95, 95, shape=(bsz, seq_len, head_num, head_dim), dtype='int32'), dtype='int8')
# v_p = paddle.empty(shape=(bsz, seq_len, head_num, head_dim), dtype='float16')
# o_p = paddle.empty(shape=(bsz, seq_len, head_num, head_dim), dtype='float16')
# q_scale_p = paddle.empty(shape=(bsz, head_num, (seq_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), dtype='float32')
# k_scale_p = paddle.empty(shape=(bsz, head_num, (seq_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), dtype='float32')

k_p = paddle.to_tensor(k_npy, dtype='float16')
# q_int8_t_p = paddle.to_tensor(q_int8_t_npy, dtype='int8')
# q_scale_p = paddle.to_tensor(q_scale_npy, dtype='float32')
km_t = paddle.mean(k_p, axis=seq_dim, keepdim=True)
km_t = km_t.squeeze(1)
# print(km_t.shape)
k_int8_t_p = paddle.to_tensor(paddle.empty(shape=k_p.shape, dtype='int32'), dtype='int8')
k_scale_p = paddle.empty(shape=(bsz, head_num, ((seq_len + 63) // 64)), dtype='float32')
# print(q_scale_p.dim())

quant_per_block_int8_fuse_sub_mean_cuda_paddle(k_p, km_t, k_int8_t_p, k_scale_p, 64, tensor_layout)


paddle.device.synchronize()

sim, l1 = precision_cmp(km, km_t)

print(sim, l1)

sim, l1 = precision_cmp(k_int8_t, k_int8_t_p)

print(sim, l1)

sim, l1 = precision_cmp(k_scale, k_scale_p)

print(sim, l1)
