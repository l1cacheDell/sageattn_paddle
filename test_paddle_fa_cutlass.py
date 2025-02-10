import paddle
import argparse
import nvtx
from paddle.incubate.nn.functional import variable_length_memory_efficient_attention
import math

from paddle_sageattn import sageattn_qk_int8_pv_fp8_cuda
from utils import precision_cmp_paddle

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

cutlass_seq_len = paddle.to_tensor(data=[[seq_len]], dtype=paddle.int32)
# print(cutlass_seq_len)


tensor_layout = "NHD" # bsz, seq_len, num_head, head_dim
return_lse = False
pv_accum_dtype ="fp32+fp32"

# q = paddle.randn(shape=(bsz, num_heads, seq_len, head_dim), dtype=paddle.float16)
# k = paddle.randn(shape=(bsz, 4, seq_len, head_dim), dtype=paddle.float16)
# v = paddle.randn(shape=(bsz, 4, seq_len, head_dim), dtype=paddle.float16)
q = paddle.to_tensor(paddle.load("./inputs/q.pdparams"), dtype=paddle.float16)
k = paddle.to_tensor(paddle.load("./inputs/k.pdparams"), dtype=paddle.float16)
v = paddle.to_tensor(paddle.load("./inputs/v.pdparams"), dtype=paddle.float16)
print(f"q: {q[0, 0, 0, :]}")
print(f"k: {k[0, 0, 0, :]}")
print(f"v: {v[0, 0, 0, :]}")

# print(q.shape)
# print(k.shape)
# print(v.shape)

mask = paddle.zeros([bsz, 1, 4096, 4096], dtype=paddle.float16)

# warm up cutlass
for i in range(5):
    o = variable_length_memory_efficient_attention(q, k, v, 
                                                cutlass_seq_len, 
                                                cutlass_seq_len + 0, 
                                                # mask=mask, 
                                                scale=float(head_dim**-0.5))
    # print(o[0, 0, 0, :])
    qkv_out = paddle.load("./inputs/qktv_out.pdparams")
    print((qkv_out - o)[0, 0, 0, :])
    print(paddle.max(qkv_out - o))
    break

# runing
for i in range(100):
    sageatt_nvtx = nvtx.start_range(message="cutlass_casual_false", color="green")
    # code
    o2 = variable_length_memory_efficient_attention(q, k, v, 
                                                seq_lens=cutlass_seq_len, kv_seq_lens=cutlass_seq_len, mask=mask, scale=float(head_dim**-0.5))
    
    paddle.device.synchronize()
    nvtx.end_range(sageatt_nvtx)
# print(o2[0, 0, 0, :])

# warm up FA2
is_casual = False

for i in range(5):
    o = paddle.nn.functional.scaled_dot_product_attention(q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3]), attn_mask=None, dropout_p=0.0, is_causal=is_casual, training=False)

paddle.device.synchronize()

# runing
for i in range(100):
    transformer_nvtx = nvtx.start_range(message="FA2_casual_false", color="red")
    # code
    o1 = paddle.nn.functional.scaled_dot_product_attention(q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3]), attn_mask=None, dropout_p=0.0, is_causal=is_casual, training=False)
    
    paddle.device.synchronize()
    nvtx.end_range(transformer_nvtx)    


    
# sageattn
for i in range(5):
    o = sageattn_qk_int8_pv_fp8_cuda(q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3]), is_causal=is_casual, sm_scale=float(head_dim**-0.5), tensor_layout="NHD")

paddle.device.synchronize()

# runing
for i in range(100):
    transformer_nvtx = nvtx.start_range(message="sageattn_casual_false", color="red")
    # code
    o3 = sageattn_qk_int8_pv_fp8_cuda(q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3]), is_causal=is_casual, sm_scale=float(head_dim**-0.5), tensor_layout="NHD")
    
    paddle.device.synchronize()
    nvtx.end_range(transformer_nvtx)    
    
print(o1[0, 0, 0, :])

cos, l1 = precision_cmp_paddle(o3, o2.transpose([0, 2, 1, 3]))
print(f"cos: {cos}, l1: {l1}", paddle.max(o3 -  o2.transpose([0, 2, 1, 3])))
cos, l1 = precision_cmp_paddle(o3, o1)
print(f"cos: {cos}, l1: {l1}", paddle.max(o3-o1))
