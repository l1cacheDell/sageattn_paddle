import paddle
import argparse
import nvtx

from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda

parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP16 CUDA')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
args = parser.parse_args()

num_heads = args.num_heads
bsz = args.batch_size
head_dim = args.head_dim
seq_len = args.seq_len

tensor_layout = "NHD" # bsz, seq_len, num_head, head_dim
return_lse = False
pv_accum_dtype ="fp32+fp32"

q = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)
k = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)
v = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)

# warm up FA2
is_casual = False

for i in range(5):
    o = paddle.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_casual, training=False)

paddle.device.synchronize()

# runing
for i in range(100):
    transformer_nvtx = nvtx.start_range(message="FA2_casual_false", color="red")
    # code
    o = paddle.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_casual, training=False)
    
    paddle.device.synchronize()
    nvtx.end_range(transformer_nvtx)    

# warm up SA
for i in range(5):
    o = sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_casual, return_lse=return_lse, pv_accum_dtype="fp32+fp32")

# runing
for i in range(100):
    sageatt_nvtx = nvtx.start_range(message="Sageattn_casual_false", color="green")
    # code
    o = sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_casual, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    
    paddle.device.synchronize()
    nvtx.end_range(sageatt_nvtx)
    
flops = 4 * num_heads * bsz * head_dim * seq_len * seq_len / (2 if is_casual else 1)
print(f"is casual: {is_casual}, FLOPS: {flops}")

################################################################################################################################################################

is_casual = True

for i in range(5):
    o = paddle.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_casual, training=False)

paddle.device.synchronize()

# runing
for i in range(100):
    transformer_nvtx = nvtx.start_range(message="FA2_casual_true", color="red")
    o = paddle.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_casual, training=False)
    paddle.device.synchronize()
    nvtx.end_range(transformer_nvtx)    

# warm up SA
for i in range(5):
    o = sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_casual, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    
# runing
for i in range(100):
    sageatt_nvtx = nvtx.start_range(message="Sageattn_casual_true", color="green")
    o = sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_casual, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    paddle.device.synchronize()
    nvtx.end_range(sageatt_nvtx)
    
flops = 4 * num_heads * bsz * head_dim * seq_len * seq_len / (2 if is_casual else 1)
print(f"is casual: {is_casual}, FLOPS: {flops}")