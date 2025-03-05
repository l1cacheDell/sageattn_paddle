#include <vector>

#include "paddle/extension.h"
#include "sageattn_func.cuh"

std::vector<paddle::Tensor> per_warp_int8_cuda(paddle::Tensor& q,
                                            paddle::Tensor& k,
                                            paddle::optional<paddle::Tensor>& km,
                                            int BLKQ,
                                            int WARPQ,
                                            int BLKK,
                                            int tensor_layout) 
{
    paddle::Tensor q_int8 = paddle::empty(q.shape, paddle::DataType::INT8);
    paddle::Tensor k_int8 = paddle::empty(k.shape, paddle::DataType::INT8);
    int b, h_qo, qo_len, head_dim, h_kv, kv_len;

    // tensor_layout == 0: NHD - [b, qo_len, h_qo, head_dim]
    // tensor_layout == 1: HND - [b, h_qo, qo_len, head_dim]
    if (tensor_layout == 0) {
        b = q.shape()[0];
        h_qo = q.shape()[2];
        qo_len = q.shape()[1];
        head_dim = q.shape()[3];

        h_kv = k.shape()[2];
        kv_len = k.shape()[1];
    } else if (tensor_layout == 1) {
        b = q.shape()[0];
        h_qo = q.shape()[1];
        qo_len = q.shape()[2];
        head_dim = q.shape()[3];

        h_kv = k.shape()[1];
        kv_len = k.shape()[2];
    } else {
        throw std::invalid_argument("tensor_layout must be 0 or 1");
    }

    paddle::Tensor q_scale = paddle::empty({b, h_qo, ((qo_len + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ)}, paddle::DataType::FLOAT32);
    paddle::Tensor k_scale = paddle::empty({b, h_kv, ((kv_len + BLKK - 1) / BLKK)}, paddle::DataType::FLOAT32);

    quant_per_warp_int8_cuda_fwd(q, q_int8, q_scale, BLKQ, WARPQ, tensor_layout);
    if (km) {
        if (tensor_layout == 0) {
            km = km.squeeze(1);
        } else {
            km = km.squeeze(2);
        }
        quant_per_block_int8_fuse_sub_mean_cuda_fwd(k, km, k_int8, k_scale, BLKK, tensor_layout);
    } else {
        quant_per_block_int8_cuda_fwd(k, k_int8, k_scale, BLKK, tensor_layout);
    }

    return {q_int8, q_scale, k_int8, k_scale};
}

std::vector<paddle::Tensor> per_channel_fp8(paddle::Tensor& v,
                                            int tensor_layout,
                                            float scale_max,
                                            bool smooth_v)
{
    int b, head_dim, h_kv, kv_len, padded_len;
    paddle::Tensor v_transposed_permutted;
    if (tensor_layout == 1) {
        b = v.shape()[0];
        h_kv = v.shape()[1];
        kv_len = v.shape()[2];
        head_dim = v.shape()[3];

        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = paddle::empty({b, h_kv, head_dim, padded_len}, v.dtype());
    } else if (tensor_layout == 0) {
        b = v.shape()[0];
        kv_len = v.shape()[1];
        h_kv = v.shape()[2];
        head_dim = v.shape()[3];

        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = paddle::empty({b, head_dim, h_kv, padded_len}, v.dtype());
    }
    transpose_pad_permute_cuda_fwd(v, v_transposed_permutted, tensor_layout);

    paddle::Tensor v_fp8 = paddle::empty(v_transposed_permutted.shape(), paddle::DataType::FLOAT8_E4M3FN);
    paddle::Tensor v_scale = paddle::empty({b, h_kv, head_dim}, paddle::DataType::FLOAT32);
    paddle::Tensor vm = paddle::empty({b, h_kv, head_dim}, paddle::DataType::FLOAT32);
    if (smooth_v) {
        mean_scale_fuse_quant_cuda_fwd(v_transposed_permutted, v_fp8, vm, v_scale, kv_len, scale_max, tensor_layout);
    } else {
        scale_fuse_quant_cuda_fwd(v_transposed_permutted, v_fp8, v_scale, kv_len, scale_max, tensor_layout);
    }

    return {v_fp8, v_scale, vm};
}

std::vector<paddle::Tensor> sub_mean(paddle::Tensor& v,
                                    int tensor_layout)
{
    int tgt_dim = 1;
    if (tensor_layout == 1) {
        tgt_dim = 2;
    }
    paddle::Tensor vm = v.mean(axis=tgt_dim);
    paddle::Tensor v_smoothed = paddle::empty(v.shape(), paddle::DataType::FLOAT16);

    sub_mean_cuda_fwd(v, vm, v_smoothed, tensor_layout);
    return {v_smoothed, vm}
}