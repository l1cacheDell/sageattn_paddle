#pragma once
#include "paddle/extension.h"

//
// ========== See implement: sageattn_fused.cu ==========
//

void quant_per_block_int8_fuse_sub_mean_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& mean,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int tensor_layout);

void quant_per_warp_int8_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int warp_block_size,
                int tensor_layout);

void quant_per_block_int8_cuda_scale_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                float sm_scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int tensor_layout);

void sub_mean_cuda_fwd(paddle::Tensor& input,
                paddle::Tensor& mean,
                paddle::Tensor& output,
                int tensor_layout);

void transpose_pad_permute_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                int tensor_layout);

void scale_fuse_quant_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

void mean_scale_fuse_quant_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& mean,
                paddle::Tensor& scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

//
// ========== See implement: sageattn_qk_int_sv_f8_dsk_kernel_sm90.cu ==========
//

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_attn_inst_buf_dsk_sm90_fwd(
                  paddle::Tensor& query,
                  paddle::Tensor& key,
                  paddle::Tensor& query_pe,
                  paddle::Tensor& key_pe,
                  paddle::Tensor& value,
                  paddle::Tensor& output,
                  paddle::Tensor& query_scale,
                  paddle::Tensor& key_scale,
                  int tensor_layout,
                  int is_causal,
                  int qk_quant_gran,
                  float sm_scale,
                  int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_dsk_sm90_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& query_pe,
                    paddle::Tensor& key_pe,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

//
// ========== See implement: sageattn_qk_int_sv_f8_kernel_sm89.cu ==========
//

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_attn_fwd(paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_attn_inst_buf_fwd(paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    paddle::Tensor& value_mean,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

//
// ========== See implement: sageattn_qk_int_sv_f8_kernel_sm90.cu ==========
//

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_fwd(
                  paddle::Tensor& query,
                  paddle::Tensor& key,
                  paddle::Tensor& value,
                  paddle::Tensor& output,
                  paddle::Tensor& query_scale,
                  paddle::Tensor& key_scale,
                  int tensor_layout,
                  int is_causal,
                  int qk_quant_gran,
                  float sm_scale,
                  int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

//
// ========== See implement: sageattn_qk_int_sv_f16_kernel_sm80.cu ==========
//

std::vector<paddle::Tensor>  qk_int8_sv_f16_accum_f32_attn_fwd(paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f16_attn_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f16_attn_inst_buf_fwd(paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_fwd(paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_mean,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);