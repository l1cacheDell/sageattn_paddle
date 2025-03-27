import torch
from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90

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
seq_len = 1024
num_heads = 128
head_dim = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

q = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
k = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
v = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()

q = q.transpose_(1, 2)
k = k.transpose_(1, 2)
v = v.transpose_(1, 2)

o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)

q = q.transpose_(1, 2)
k = k.transpose_(1, 2)
v = v.transpose_(1, 2)

o2 = sageattn_qk_int8_pv_fp8_cuda_sm90(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, return_lse=return_lse, pv_accum_dtype="fp32+fp32")

sim, l1, max_diff = precision_cmp_torch(o.transpose_(1, 2), o2)
print(f"Sim: {sim}, L1: {l1}, MaxDiff: {max_diff}")