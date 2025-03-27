from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90
import torch

import nvtx

bsz = 2
seq_lens = [1024, 4096, 8192, 1024*16, 1024*32, 1024*64]
num_heads = 16
head_dim = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

for seq_len in seq_lens:
    q = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
    k = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
    v = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()

    for i in range(100):
        transformer_nvtx = nvtx.start_range(message=f'FA_{seq_len}', color='red')
        o1 = sageattn_qk_int8_pv_fp8_cuda_sm90(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
        torch.cuda.synchronize()
        nvtx.end_range(transformer_nvtx)