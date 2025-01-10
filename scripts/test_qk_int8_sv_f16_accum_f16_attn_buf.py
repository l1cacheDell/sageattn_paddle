import paddle
import torch
import numpy as np
from sageattn_custom_ops import qk_int8_sv_f16_accum_f16_attn_buf as qk_int8_sv_f16_accum_f16_attn_buf_paddle
from phi_sageattention import qk_int8_sv_f16_accum_f16_attn_buf as qk_int8_sv_f16_accum_f16_attn_buf_torch

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
q_t = torch.randint(-95, 95, (bsz, seq_len, head_num, head_dim), dtype=torch.int8).cuda()
k_t = torch.randint(-95, 95, (bsz, seq_len, head_num, head_dim), dtype=torch.int8).cuda()
v_t = torch.randn((bsz, seq_len, head_num, head_dim), dtype=torch.float16).cuda()
o_t = torch.empty((bsz, seq_len, head_num, head_dim), dtype=torch.float16).cuda()
q_scale_t = torch.randn((bsz, head_num, seq_len // WARPQ * 8), device=q_t.device, dtype=torch.float32)
k_scale_t = torch.randn((bsz, head_num, seq_len // WARPK * 4), device=k_t.device, dtype=torch.float32)

print(f"original o: {o_t[0, 0, 0, :]}")

head_dim_og = q_t.size(-1)
sm_scale = head_dim_og**-0.5

lse_t = qk_int8_sv_f16_accum_f16_attn_buf_torch(q_t, k_t, v_t, o_t, q_scale_t, k_scale_t, tensor_layout, is_casual, _qk_quant_gran, sm_scale, return_lse)

torch.cuda.synchronize()

# paddle
q_npy = q_t.cpu().numpy()
k_npy = k_t.cpu().numpy()
v_npy = v_t.cpu().numpy()
o_npy = o_t.cpu().numpy()
q_scale_npy = q_scale_t.cpu().numpy()
k_scale_npy = k_scale_t.cpu().numpy()

# q_p = paddle.to_tensor(paddle.randint(-95, 95, shape=(bsz, seq_len, head_num, head_dim), dtype='int32'), dtype='int8')
# k_p = paddle.to_tensor(paddle.randint(-95, 95, shape=(bsz, seq_len, head_num, head_dim), dtype='int32'), dtype='int8')
# v_p = paddle.empty(shape=(bsz, seq_len, head_num, head_dim), dtype='float16')
# o_p = paddle.empty(shape=(bsz, seq_len, head_num, head_dim), dtype='float16')
# q_scale_p = paddle.empty(shape=(bsz, head_num, (seq_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), dtype='float32')
# k_scale_p = paddle.empty(shape=(bsz, head_num, (seq_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), dtype='float32')

q_p = paddle.to_tensor(q_npy, dtype='int8')
k_p = paddle.to_tensor(k_npy, dtype='int8')
v_p = paddle.to_tensor(v_npy, dtype='float16')
o_p = paddle.to_tensor(o_npy, dtype='float16')
q_scale_p = paddle.to_tensor(q_scale_npy, dtype='float32')
k_scale_p = paddle.to_tensor(k_scale_npy, dtype='float32')
print(q_scale_p.dim())

lse_p = qk_int8_sv_f16_accum_f16_attn_buf_paddle(q_p, k_p, v_p, o_p, q_scale_p, k_scale_p, tensor_layout, is_casual, _qk_quant_gran, sm_scale, return_lse)


paddle.device.synchronize()

sim, l1 = precision_cmp(o_t, o_p)

print(sim, l1)
