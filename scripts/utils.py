import paddle
import torch
import numpy as np

def precision_cmp(t1: torch.Tensor, t2: paddle.Tensor):
    t1_npy = t1.cpu().numpy()
    pt1 = paddle.to_tensor(t1_npy, dtype=t2.dtype, place=t2.place)
    
    x, xx = paddle.cast(pt1, dtype='float32'), paddle.cast(t2, dtype='float32')
    # 重塑张量并计算余弦相似度
    x_reshaped = paddle.reshape(x, [1, -1])
    xx_reshaped = paddle.reshape(xx, [1, -1])
    sim = paddle.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # 计算 L1 误差
    l1 = (paddle.abs(x - xx).sum() / paddle.abs(xx).sum()).item()
    
    # if sim == 0:    
    # print(pt1 - t2)
    # print(pt1[0, 0, :, 0])
    # print(t2[0, 0, :, 0])
    
    return sim, l1