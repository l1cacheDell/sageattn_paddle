#include "paddle/extension.h"


//
//  ============== fp16 kernels registry, for sm80 arch ==============
//
// impl: sageattn_qk_int_sv_f16_kernel.cu
// attn buffer kernel
std::vector<paddle::Tensor>  qk_int8_sv_f16_accum_f16_attn_buf_fwd(
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

std::vector<std::vector<int64_t>> qk_int8_sv_f16_accum_f16_attn_buf_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f16_accum_f16_attn_buf_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f16_accum_f16_attn_buf)
    .Inputs({"query", "key", "value", "output", "query_scale", "key_scale"})
    .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "lse"})
    .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f16_accum_f16_attn_buf_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f16_accum_f16_attn_buf_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f16_accum_f16_attn_buf_InferDtype));


// attn forward kernel: sv f16 accumulator f32
std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f32_attn_fwd(
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


std::vector<std::vector<int64_t>> qk_int8_sv_f16_accum_f32_attn_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f16_accum_f32_attn_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f16_accum_f32_attn)
    .Inputs({"query", "key", "value", "output", "query_scale", "key_scale"})
    .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "lse"})
    .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f16_accum_f32_attn_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f16_accum_f32_attn_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f16_accum_f32_attn_InferDtype));

//
//  ============== fp8 kernels registry, for sm89 arch ==============
//



//
//  ============== fused kernels registry ==============
//

void quant_per_block_int8_fuse_sub_mean_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& mean,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int tensor_layout);


// quant_per_warp_int8_cuda_fwd does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(quant_per_block_int8_fuse_sub_mean_cuda)
    .Inputs({"input", "mean", "output", "scale"})
    .Outputs({"out1", "out2", "out3", "out4"})
    .SetInplaceMap({{"input", "out1"}, {"mean", "out2"}, {"output", "out3"}, {"scale", "out4"}}) // Inplace
    .Attrs({"block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_block_int8_fuse_sub_mean_cuda_fwd));

void quant_per_warp_int8_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int tensor_layout);


// quant_per_warp_int8_cuda_fwd does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(quant_per_warp_int8_cuda)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_warp_int8_cuda_fwd));