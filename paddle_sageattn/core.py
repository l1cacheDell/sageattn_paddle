import paddle
import sageattn_custom_ops

from typing import Optional, Any
import warnings

from .quant import per_channel_fp8
from .quant import per_warp_int8 as per_warp_int8_cuda


def sageattn_qk_int8_pv_fp8_cuda(
    q: paddle.Tensor, 
    k: paddle.Tensor, 
    v: paddle.Tensor,
    tensor_layout: str = "NHD",
    is_causal: bool = False,
    qk_quant_gran: str = "per_warp",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
):
    dtype = q.dtype
    assert dtype in [paddle.float16, paddle.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_causal = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0
    
    head_dim_og = q.shape[-1]
    
    if head_dim_og < 64:
        q = paddle.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = paddle.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = paddle.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = paddle.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = paddle.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = paddle.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    
    assert q.strides[-1] == 1 and k.strides[-1] == 1 and v.strides[-1] == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5
        
    seq_dim = 1 if _tensor_layout == 0 else 2
    
    if smooth_k:
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = paddle.squeeze(paddle.matmul(paddle.transpose(q, [0, 2, 1, 3], paddle.transpose(km, [0, 2, 3, 1]))), axis=-1)
            else:
                lse_correction = paddle.squeeze(paddle.matmul(q, paddle.transpose(km, [0, 1, 3, 2])), axis=-1)
    else:
        km = None
        
    q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout)

    o = paddle.empty(q.shape, dtype=dtype)
    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False
        
    v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=smooth_v)

    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = sageattn_custom_ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            lse = sageattn_custom_ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = sageattn_custom_ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
    

def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: paddle.Tensor, 
    k: paddle.Tensor, 
    v: paddle.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_warp",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> paddle.Tensor:
    dtype = q.dtype
    assert dtype in [paddle.float16, paddle.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_causal = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.shape[-1]

    if head_dim_og < 64:
        q = paddle.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = paddle.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = paddle.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = paddle.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = paddle.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = paddle.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    
    assert q.strides[-1] == 1 and k.strides[-1] == 1 and v.strides[-1] == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5
        
    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = paddle.squeeze(paddle.matmul(paddle.transpose(q, [0, 2, 1, 3], paddle.transpose(km, [0, 2, 3, 1]))), axis=-1)
            else:
                lse_correction = paddle.squeeze(paddle.matmul(q, paddle.transpose(km, [0, 1, 3, 2])), axis=-1)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128)

    o = paddle.empty(q.shape, dtype=dtype)

    kv_len = k.shape[seq_dim]
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = paddle.concat([v, paddle.zeros(shape=[v.shape[0], v.shape[1], v_pad_len, v.shape[3]], dtype=v.dtype)], axis=2)
        else:
            v = paddle.concat([v, paddle.zeros(shape=[v.shape[0], v_pad_len, v.shape[2], v.shape[3]], dtype=v.dtype)], axis=1)

    
    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)

    lse = sageattn_custom_ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


def sageattn_qk_int8_pv_fp8_cuda_dsk_sm90(
    q: paddle.Tensor, 
    k: paddle.Tensor, 
    v: paddle.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_warp",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> paddle.Tensor:
    dtype = q.dtype
    assert dtype in [paddle.float16, paddle.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_causal = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.shape[-1]

    if head_dim_og < 64:
        q = paddle.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = paddle.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = paddle.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = paddle.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = paddle.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = paddle.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128 and head_dim_og < 256:
        q = paddle.nn.functional.pad(q, (0, 256 - head_dim_og))
        k = paddle.nn.functional.pad(k, (0, 256 - head_dim_og))
    elif head_dim_og > 256:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    
    assert q.strides[-1] == 1 and k.strides[-1] == 1 and v.strides[-1] == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5
        
    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = paddle.squeeze(paddle.matmul(paddle.transpose(q, [0, 2, 1, 3], paddle.transpose(km, [0, 2, 3, 1]))), axis=-1)
            else:
                lse_correction = paddle.squeeze(paddle.matmul(q, paddle.transpose(km, [0, 1, 3, 2])), axis=-1)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128)

    o = paddle.empty(v.shape, dtype=dtype)

    kv_len = k.shape[seq_dim]
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = paddle.concat([v, paddle.zeros(shape=[v.shape[0], v.shape[1], v_pad_len, v.shape[3]], dtype=v.dtype)], axis=2)
        else:
            v = paddle.concat([v, paddle.zeros(shape=[v.shape[0], v_pad_len, v.shape[2], v.shape[3]], dtype=v.dtype)], axis=1)

    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)
    q_int8_nope, q_int8_pe, waste = q_int8.split([128, 64, 64], axis=-1)
    k_int8_nope, k_int8_pe, _ = k_int8.split([128, 64, 64], axis=-1)

    print(q_int8_nope.shape)
    print(k_int8_nope.shape)
    print(q_int8_pe.shape)
    print(k_int8_pe.shape)
    print(v_fp8.shape)
    print(o.shape)
    print(q_scale.shape)
    print(k_scale.shape)
    print(v_scale.shape)

    print(q_int8_nope.dtype)
    print(k_int8_nope.dtype)
    print(q_int8_pe.dtype)
    print(k_int8_pe.dtype)
    print(v_fp8.dtype)
    print(o.dtype)
    print(q_scale.dtype)
    print(k_scale.dtype)
    print(v_scale.dtype)

    lse = sageattn_custom_ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_dsk_sm90(q_int8_nope, k_int8_nope, q_int8_pe, k_int8_pe, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse)

    head_dim_og = v.shape[-1]
    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o