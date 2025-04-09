from paddlenlp.ops.triton_ops.segment_mean import segment_mean
import paddle
import nvtx

BLOCK_SIZE_SEQ=256
BLOCK_SIZE_HEAD=4
BLOCK_SIZE_DIM=64

# 构造测试数据
# batch_sizes = [1024 * 64 - 1, 1024 * 32 + 1, 1024 * 48 + 5, 1024 * 64 + 6, 1024 * 8 + 6]
batch_sizes = [1024 * 64 - 1]
num_batches = len(batch_sizes)
num_heads = 24
head_dim = 128
dtype = paddle.float16

# 生成累计序列长度
cu_seqlen = paddle.cumsum(
    paddle.to_tensor([0] + batch_sizes, dtype=paddle.int32),
    axis=0
)

# 生成随机输入数据
total_seqlen = sum(batch_sizes)
input = paddle.randn(
    [total_seqlen, num_heads, head_dim],
    dtype=dtype
)

output_triton = segment_mean(input, cu_seqlen)

for i in range(100):
    paddle.device.synchronize()
    start = nvtx.start_range(message="pd_triton", color="green")
    output_triton = segment_mean(input, cu_seqlen)
    paddle.device.synchronize()
    nvtx.end_range(start)

# PyTorch原生实现用于验证
torch_output = []
for i in range(num_batches):
    start = cu_seqlen[i]
    end = cu_seqlen[i+1]
    segment = input[start:end]
    torch_output.append(segment.mean(axis=0))
torch_output = paddle.stack(torch_output)

# 计算误差
cos_sim = paddle.nn.functional.cosine_similarity(
    output_triton.flatten(),
    torch_output.flatten(),
    axis=0
)
max_diff = (output_triton - torch_output).abs().max()
print(f"Cosine Similarity: {cos_sim.item():.6f}")
print(f"Max Difference: {max_diff.item():.6f}")