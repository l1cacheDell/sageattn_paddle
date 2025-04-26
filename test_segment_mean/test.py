import paddle
import sageattn_custom_ops
from paddlenlp.ops.triton_ops.segment_mean import segment_mean
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

num_head = 24
head_dim = 128

# bsz = 1
k = paddle.randn([131, num_head, head_dim], dtype=paddle.float16)
cu_seqlen = paddle.to_tensor([0, 131], paddle.int32)

km1 = segment_mean(k, cu_seqlen)
paddle.device.synchronize()
km2 = sageattn_custom_ops.chunked_segment_mean(k, cu_seqlen, 131)
paddle.device.synchronize()

sim, l1, max_diff = precision_cmp_paddle(km1, km2)
print(f"bsz = 1, sim: {sim}, l1: {l1}, max_diff: {max_diff}")

# bsz = 2
total_seqlen = 1026
cu_seqlen = paddle.to_tensor([0, 511, total_seqlen], paddle.int32)
k = paddle.randn([total_seqlen, num_head, head_dim], dtype=paddle.float16)


km1 = segment_mean(k, cu_seqlen)
paddle.device.synchronize()
km2 = sageattn_custom_ops.chunked_segment_mean(k, cu_seqlen, 131)
paddle.device.synchronize()

sim, l1, max_diff = precision_cmp_paddle(km1, km2)
print(f"bsz = 2, sim: {sim}, l1: {l1}, max_diff: {max_diff}")

# bsz = 4
total_seqlen = 1026 * 48
cu_seqlen = paddle.to_tensor([0, 1024 * 12, 1024 * 24, 1024 * 37, total_seqlen], paddle.int32)
k = paddle.randn([total_seqlen, num_head, head_dim], dtype=paddle.float16)


km1 = segment_mean(k, cu_seqlen)
paddle.device.synchronize()
km2 = sageattn_custom_ops.chunked_segment_mean(k, cu_seqlen, 131)
paddle.device.synchronize()

sim, l1, max_diff = precision_cmp_paddle(km1, km2)
print(f"bsz = 4, sim: {sim}, l1: {l1}, max_diff: {max_diff}")


# for i in range(100):
#     paddle.device.synchronize()
#     triton_nvtx = nvtx.start_range(message="triton_nvtx", color="blue")
#     km1 = segment_mean(k, cu_seqlen)
#     paddle.device.synchronize()
#     nvtx.end_range(triton_nvtx)

#     paddle.device.synchronize()
#     custom_nvtx = nvtx.start_range(message="custom_nvtx", color="red")
#     km2 = sageattn_custom_ops.chunked_segment_mean(k, cu_seqlen, 131)
#     paddle.device.synchronize()
#     nvtx.end_range(custom_nvtx)