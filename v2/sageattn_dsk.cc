#include "paddle/extension.h"

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

std::vector<std::vector<int64_t>> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_dsk_sm90_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> query_pe_shape, 
  std::vector<int64_t> key_pe_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape,
  std::vector<int64_t> value_scale_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_dsk_sm90_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype,
  paddle::DataType G_dtype,
  paddle::DataType H_dtype,
  paddle::DataType I_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_dsk_sm90_v2)
    .Inputs({"query", "key", "query_pe", "key_pe", "value", "output", "query_scale", "key_scale", "value_scale"})
    .Outputs({"out", "lse"})
    .SetInplaceMap({{"output", "out"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_dsk_sm90_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_dsk_sm90_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_dsk_sm90_InferDtype));
