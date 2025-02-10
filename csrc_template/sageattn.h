// this file stores all sageattn low-level operators API exposed to outside.

#include "paddle/extension.h"

// low-level operators
// see impl -> sageattn_qk_int_sv_f16_kernel.cu
std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f32_attn(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f16_attn(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f16_attn_inst_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f16_attn_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_mean,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

// see impl -> sageattn_qk_int_sv_f8_kernel.cu
std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_attn(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_scale,
                    paddle::Tensor value_mean,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_attn_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_attn_inst_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);


// fused kernel APIs
void quant_per_block_int8_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor scale,
                float sm_scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_fuse_sub_mean_cuda(
                paddle::Tensor input,
                paddle::Tensor mean,
                paddle::Tensor output,
                paddle::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_warp_int8_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor scale,
                int tensor_layout);

void sub_mean_cuda(
                paddle::Tensor input,
                paddle::Tensor mean,
                paddle::Tensor output,
                int tensor_layout);

void transpose_pad_permute_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                int tensor_layout);

void scale_fuse_quant_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

void mean_scale_fuse_quant_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor mean,
                paddle::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);