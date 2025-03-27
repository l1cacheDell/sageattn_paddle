import sageattn_custom_ops
import paddle

import nvtx

bsz = 2
seq_lens = [1024, 4096, 8192, 1024*16, 1024*32, 1024*64]
num_heads = 16
head_dim = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

for seq_len in seq_lens:
    q = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)
    k = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)
    v = paddle.randn(shape=(bsz, seq_len, num_heads, head_dim), dtype=paddle.float16)

    for i in range(100):
        km = paddle.mean(k, axis=1, keepdim=True)
        km = km.squeeze(1) if tensor_layout == "NHD" else km.squeeze(2)
        transformer_nvtx = nvtx.start_range(message=f'SA_{seq_len}', color='green')
        o1 = sageattn_custom_ops.sage_attention(q, 
                                                k, 
                                                v, 
                                                km, 
                                                None,
                                                head_dim**-0.5,
                                                "per_warp",
                                                "fp16",
                                                tensor_layout=0, 
                                                is_causal=is_causal, 
                                                smooth_k=True, 
                                                smooth_v=False, 
                                                return_lse=return_lse)
        paddle.device.synchronize()
        nvtx.end_range(transformer_nvtx)
