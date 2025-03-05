import paddle
from torch.nn.functional import scaled_dot_product_attention as sdpa

import torch
import paddle
import numpy as np
import nvtx

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

bsz = 2
seq_len = 1024 * 64
num_heads = 16
head_dim_qk = 128 + 64
head_dim_v = 128

REPEAT_NUM = 100

tensor_layout = "NHD"
is_causal = True
return_lse = False

torch.backends.cuda.enable_flash_sdp(True)

# prepare input for torch
q = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
k = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
v = torch.randn((bsz, seq_len, num_heads, head_dim_v), dtype=torch.float16).cuda()
v = torch.nn.functional.pad(v, (0, head_dim_qk - head_dim_v))

# permute for sdpa
q = q.transpose(2, 1)
k = k.transpose(2, 1)
v = v.transpose(2, 1)

for i in range(REPEAT_NUM):
    transformer_nvtx = nvtx.start_range(message='FA2', color='red')
    o_torch_fa2 = sdpa(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    nvtx.end_range(transformer_nvtx)
