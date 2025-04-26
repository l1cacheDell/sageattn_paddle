import paddle
import sageattn_custom_ops
from paddlenlp.ops.triton_ops.segment_mean import segment_mean

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
km2 = sageattn_custom_ops.chunked_segment_mean(k, cu_seqlen, 131)

sim, l1, max_diff = precision_cmp_paddle(km1, km2)
print(f"sim: {sim}, l1: {l1}, max_diff: {max_diff}")