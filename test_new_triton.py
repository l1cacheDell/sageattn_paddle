import paddle
from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda_sm90 as sageattn_qk_int8_pv_fp8_cuda_paddle
import nvtx
from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90 as sageattn_qk_int8_pv_fp8_cuda_torch
import numpy as np
import torch

def precision_cmp_paddle(t1: paddle.Tensor, t2: paddle.Tensor):
    
    x, xx = paddle.cast(t1, dtype='float32'), paddle.cast(t2, dtype='float32')
    # 重塑张量并计算余弦相似度
    x_reshaped = paddle.reshape(x, [1, -1])
    xx_reshaped = paddle.reshape(xx, [1, -1])
    sim = paddle.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # 计算 L1 误差
    l1 = (paddle.abs(x - xx).sum() / paddle.abs(xx).sum()).item()
    
    max_diff = paddle.max(x - xx)
    
    return sim, l1, max_diff

def precision_cmp_torch(t1: torch.Tensor, t2: torch.Tensor):
    x, xx = t1.to(dtype=torch.float32), t2.to(dtype=torch.float32)
    # 重塑张量并计算余弦相似度
    x_reshaped = torch.reshape(x, [1, -1])
    xx_reshaped = torch.reshape(xx, [1, -1])
    sim = torch.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # 计算 L1 误差
    l1 = (torch.abs(x - xx).sum() / torch.abs(xx).sum()).item()

    max_diff = torch.max(x - xx)
    # print("Max Diff: ", max_diff.item())
    
    return sim, l1, max_diff

bsz = 2
batch_size = bsz
seq_len = 1024 * 32
head_num = 24
num_heads=head_num
head_dim = 64

REPEAT_NUM=100

BLKQ=128
BLKK=64
WARPQ=32
WARPK=64
_qk_quant_gran = 3
tensor_layout = "NHD" # NHD
is_causal = False
return_lse = False

q = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float16, device="cuda")
k = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float16, device="cuda")
v = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float16, device="cuda")

o_0 = sageattn_qk_int8_pv_fp8_cuda_torch(q, k, v, is_causal=is_causal, tensor_layout=tensor_layout)
o_1 = sageattn_qk_int8_pv_fp8_cuda_torch(q, k, v, is_causal=is_causal, qk_quant_gran="per_warp", tensor_layout=tensor_layout)

sim, l1, diff = precision_cmp_torch(o_0, o_1)
print(f"torch impl sim: {sim}, diff: {diff}")

q_npy = q.cpu().numpy()
k_npy = k.cpu().numpy()
v_npy = v.cpu().numpy()

# paddle
q_p = paddle.to_tensor(q_npy)
k_p = paddle.to_tensor(k_npy)
v_p = paddle.to_tensor(v_npy)


o_p0 = paddle.nn.functional.scaled_dot_product_attention(q_p, k_p, v_p, is_causal=is_causal)

for i in range(REPEAT_NUM):
    transformer_nvtx = nvtx.start_range(message="cuda_kernel", color="red")
    o_p = sageattn_qk_int8_pv_fp8_cuda_paddle(q_p, k_p, v_p, qk_quant_gran="per_warp", tensor_layout=tensor_layout, is_causal=is_causal)
    paddle.device.synchronize()
    nvtx.end_range(transformer_nvtx)

for i in range(REPEAT_NUM):
    transformer_nvtx = nvtx.start_range(message="triton_kernel", color="red")
    o_p2 = sageattn_qk_int8_pv_fp8_cuda_paddle(q_p, k_p, v_p, qk_quant_gran="per_thread", tensor_layout=tensor_layout, is_causal=is_causal)
    paddle.device.synchronize()
    nvtx.end_range(transformer_nvtx)

paddle.device.synchronize()



sim, l1, diff = precision_cmp_paddle(o_p, o_p0)
print(f"sim: {sim}, diff: {diff}")

sim, l1, diff = precision_cmp_paddle(o_p2, o_p0)
print(f"sim: {sim}, diff: {diff}")