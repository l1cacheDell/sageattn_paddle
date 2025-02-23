import paddle
from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle

bsz = 2
seq_len = 1026
num_heads = 128
head_dim_qk = 128 + 64
head_dim_v = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

q = paddle.randn([bsz, seq_len, num_heads, head_dim_qk], dtype="float16")
k = paddle.randn([bsz, seq_len, num_heads, head_dim_qk], dtype="float16")
v = paddle.randn([bsz, seq_len, num_heads, head_dim_v], dtype="float16")

o_paddle_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a_paddle(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
paddle.device.synchronize()

