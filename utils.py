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
    # sub_tensor = x - xx
    # sum_out = paddle.sum(sub_tensor)
    # print("sum out: ", sum_out.numpy())
    # 计算 L1 误差
    l1 = (paddle.abs(x - xx).sum() / paddle.abs(xx).sum()).item()

    max_diff = paddle.max(x - xx)
    
    # if sim == 0:    
    # print(pt1 - t2)
    # print(pt1[0, 0, :, 0])
    # print(t2[0, 0, :, 0])
    
    return sim, l1, max_diff

def precision_cmp_s(t1: torch.Tensor, t2: paddle.Tensor):
    t1_npy = t1.cpu().numpy()
    pt1 = paddle.to_tensor(t1_npy, dtype=t2.dtype, place=t2.place)
    
    x, xx = paddle.cast(pt1, dtype='float32'), paddle.cast(t2, dtype='float32')
    # 重塑张量并计算余弦相似度
    x_reshaped = paddle.reshape(x, [1, -1])
    xx_reshaped = paddle.reshape(xx, [1, -1])
    sim = paddle.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # with open("log.txt", "w") as f:
    #     tmp_tensor = x_reshaped - xx_reshaped
    #     print(tmp_tensor.shape)
    #     for i in range(9388928, 9388928+449):
    #         f.write(str(tmp_tensor[0, i].numpy()) + "\n")

    # 计算 L1 误差
    l1 = (paddle.abs(x - xx).sum() / paddle.abs(xx).sum()).item()

    max_diff = paddle.max(x - xx)
    argmax = paddle.argmax((x - xx).astype(paddle.float16))
    
    # if sim == 0:    
    # print(pt1 - t2)
    # print(pt1[0, 0, :, 0])
    # print(t2[0, 0, :, 0])
    
    return sim, l1, max_diff, argmax


def precision_cmp_paddle(t1: paddle.Tensor, t2: paddle.Tensor):
    
    x, xx = paddle.cast(t1, dtype='float32'), paddle.cast(t2, dtype='float32')
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
    max_diff = paddle.max(x - xx)
    
    return sim, l1, max_diff


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