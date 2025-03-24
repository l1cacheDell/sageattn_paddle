import paddle

# 示例张量
total_seq_len, num_heads, head_dim = 10, 24, 64
# data = paddle.randn([total_seq_len, num_heads, head_dim], dtype='float32')

data1 = paddle.ones([3, num_heads, head_dim], dtype='float32')
data2 = paddle.ones([3, num_heads, head_dim], dtype='float32') * 2
data3 = paddle.ones([4, num_heads, head_dim], dtype='float32') * 5

data = paddle.concat([data1, data2, data3], axis=0)

# 假设 cu_seqlen 是累积的序列长度
cu_seqlen = paddle.to_tensor([0, 3, 6, 10], dtype='int32')  # 累积长度

# 计算每个子序列的长度
segment_lengths = paddle.concat([cu_seqlen[:1], cu_seqlen[1:] - cu_seqlen[:-1]])[1:]
print(segment_lengths)

# 创建 segment_ids
segment_ids = paddle.concat([paddle.full([length], i, dtype='int32') for i, length in enumerate(segment_lengths)])

# 计算每个子序列的平均值
out = paddle.incubate.segment_mean(data, segment_ids)

print(out)