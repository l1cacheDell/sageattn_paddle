#include <type_traits>

#include "paddle/extension.h"

// #include "sageattn.h"
#include "sageattn_utils.cuh"
#include "sageattn_fused.cuh"

#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 16  // fp8
#define PACK_SIZE_O 8   // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp8 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 32

// qk_int_sv_f16_buffer
// when instantiating, the head dim = 64, which makes the V_STRIDE = 64, then div 16 = 4, 
// which triggered the compiling fault.
// it is the macro: PACK_SIZE_V and MMA_SV_K's problem, so we will redefine them here:
#ifdef PACK_SIZE_V
#define PACK_SIZE_V 8
#endif

#ifdef MMA_SV_K
#define MMA_SV_K 16
#endif

// inner impl
template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, SADataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        typename DTypeSVAccum = float, bool use_inst_buffer = false, typename DTypeOut = half, ComputeUnit DenominatorAccumUnit, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_mean=false>
__global__ void qk_int_sv_f16_attn_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, half *__restrict__ V, DTypeOut *__restrict__ O, float *__restrict__ Lse,
                      float *__restrict__ Q_scale, float *__restrict__ K_scale, DTypeOut *__restrict__ V_mean,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
                      const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      float sm_scale)
{
  // compile time check
  static_assert(DTypeQK == SADataType::kInt8 || DTypeQK == SADataType::kInt4, "DTypeQK must be int8 or int4");
  static_assert(Q_GRAN == QuantGranularity::kPerBlock || Q_GRAN == QuantGranularity::kPerWarp || Q_GRAN == QuantGranularity::kPerThread, "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp || K_GRAN == QuantGranularity::kPerThread, "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(std::is_same<DTypeSVAccum, float>::value || !use_inst_buffer, "use_inst_buffer only supports DTypeSVAccum as float");
  static_assert(std::is_same<DTypeSVAccum, float>::value || std::is_same<DTypeSVAccum, half>::value, "DTypeSVAccum must be float or half");
  static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value, "DTypeOut must be half or nv_bfloat16");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(!fuse_v_mean || std::is_same<DTypeSVAccum, half>::value, "fuse_v_mean only supports half");
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation

  using DTypeOut2 = typename std::conditional<std::is_same<DTypeOut, half>::value, half2, nv_bfloat162>::type;

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == SADataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

  constexpr uint32_t QK_SMEM_STRIDE = (DTypeQK == SADataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = head_dim;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>();
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) + get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + get_warp_idx_k<num_warps_q, num_warps_k>();
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) + (head_id / num_kv_groups) * (num_warp_block_k * 4) + get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock) ? 1 : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K) : (CTA_K / WARP_K) * 4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {        
          RO[fq][fv][k] = 0.0f;
        }
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          ((int32_t*)RO[fq][fv])[k] = 0;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      m[fq][k] = -5000000.0f;
      d[fq][k] = 1.0f;
    }
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 : (QK_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 32) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O = (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr = K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  half *V_lane_base_ptr = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_seq_v + (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);

  // for causal masking
  uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t V_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V;

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal
          ? min(kv_len, (bx + 1) * CTA_Q)
          : kv_len,
      CTA_K);

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
    &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  // for num_tiles_qk_inner = 1, we load all Qs in register
  uint32_t RQ[num_tiles_q][4];
  if constexpr (num_tiles_qk_inner == 1)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
      Q_smem_offset_mma = smem_Q.advance_offset_by_row<16>(Q_smem_offset_mma);
    }
  }

  // load K with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
    &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  float q_scale = Q_scale[q_scale_idx];

  float original_sm_scale = sm_scale;
  float dequant_scale = q_scale * K_scale[k_scale_idx + 0 * k_scale_advance_offset];

  sm_scale = original_sm_scale * dequant_scale;

  // load V with predicate
  load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
    &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  K_load_idx_lane_base += CTA_K;
  V_load_idx_lane_base += CTA_K;

#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    // do not apply causal mask and out of bound mask for these iterations
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(RS_f32, RO, m, d, sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    __syncthreads();

    // load K
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma); 
    }

    __syncthreads();
    // load V
    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }
  
  // second last iter, apply causal mask
  if (num_iterations > 1)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    __syncthreads();

    // load K with predicate
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();
    // load V with predicate
    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }

  // last iter, apply causal mask and out of bound mask
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
            RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // check out of bound in the last iter
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    // ensure V is ready
    cp_async::wait_group<0>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();

  }

  // TODO: thread block sync mdo state for num_warps_k > 0

  normalize_d<num_tiles_q, num_tiles_v, DenominatorAccumUnit>(RO, m, d);

  // save the result
  // if (get_warp_idx_k<num_warps_q, num_warps_k>() == 0)
  // {

  // convert half to bfloat16
  if constexpr (std::is_same<DTypeSVAccum, half>::value && std::is_same<DTypeOut, nv_bfloat16>::value)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
      {
        ((nv_bfloat162*)RO[fq][fv])[0] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[0]));
        ((nv_bfloat162*)RO[fq][fv])[1] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[1]));
        ((nv_bfloat162*)RO[fq][fv])[2] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[2]));
        ((nv_bfloat162*)RO[fq][fv])[3] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[3]));
      }
    }
  }

  // add v_mean
  if constexpr (fuse_v_mean)
  {
    DTypeOut2 v_mean[2];
    DTypeOut *V_mean_lane_ptr = V_mean + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + lane_id % 4 * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      v_mean[0] = *((DTypeOut2*)(V_mean_lane_ptr + fv * 16));
      v_mean[1] = *((DTypeOut2*)(V_mean_lane_ptr + 8 + fv * 16));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        ((DTypeOut2*)RO[fq][fv])[0] = __hadd2(((DTypeOut2*)RO[fq][fv])[0], v_mean[0]);
        ((DTypeOut2*)RO[fq][fv])[1] = __hadd2(((DTypeOut2*)RO[fq][fv])[1], v_mean[0]);
        ((DTypeOut2*)RO[fq][fv])[2] = __hadd2(((DTypeOut2*)RO[fq][fv])[2], v_mean[1]);
        ((DTypeOut2*)RO[fq][fv])[3] = __hadd2(((DTypeOut2*)RO[fq][fv])[3], v_mean[1]);
      }
    }
  }

  // save the result to shared memory
  uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        { 
          if constexpr (std::is_same<DTypeOut, half>::value)
          {
            ((half2*)RO_f16)[k] = __float22half2_rn(((float2*)RO[fq][fv])[k]);
          }
          else if constexpr (std::is_same<DTypeOut, nv_bfloat16>::value)
          {
            ((nv_bfloat162*)RO_f16)[k] = __float22bfloat162_rn(((float2*)RO[fq][fv])[k]);
          }
        }

        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

        // ! permuted, make sure you know what you are doing
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_f16[2];
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[1];

        // ! permuted, make sure you know what you are doing
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[2];
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[3]; 
      }
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o + lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / global_to_shared_line_lanes_O, lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(offset_O);
    }

    offset_O = smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

  if constexpr (return_lse)
  { 
    // ! this only works for num_tiles_q = 2
    uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    float *lse_lane_ptr = Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
    uint32_t fq = (lane_id % 4) / 2;
    uint32_t k = (lane_id % 4) % 2;

    if (lse_idx < qo_len)
    {
      lse_lane_ptr[0] = (math::ptx_log2(d[fq][k]) + m[fq][k]);
    }
  }

  // }
}

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, SADataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        typename DTypeSVAccum = float, bool use_inst_buffer = false, typename DTypeOut = half, ComputeUnit DenominatorAccumUnit, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_mean=false>
__global__ void qk_int_sv_f16_attn_varlen_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, half *__restrict__ V, 
                      DTypeOut *__restrict__ O, float *__restrict__ Lse,
                      float *__restrict__ Q_scale, float *__restrict__ K_scale, DTypeOut *__restrict__ V_mean,
                      uint32_t *__restrict__ cu_seqlen,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_seq_q, const uint32_t stride_h_q,
                      const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_seq_v, const uint32_t stride_h_v,
                      const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      float sm_scale)
{
  // compile time check
  static_assert(DTypeQK == SADataType::kInt8 || DTypeQK == SADataType::kInt4, "DTypeQK must be int8 or int4");
  static_assert(Q_GRAN == QuantGranularity::kPerBlock || Q_GRAN == QuantGranularity::kPerWarp || Q_GRAN == QuantGranularity::kPerThread, "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp || K_GRAN == QuantGranularity::kPerThread, "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(std::is_same<DTypeSVAccum, float>::value || !use_inst_buffer, "use_inst_buffer only supports DTypeSVAccum as float");
  static_assert(std::is_same<DTypeSVAccum, float>::value || std::is_same<DTypeSVAccum, half>::value, "DTypeSVAccum must be float or half");
  static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value, "DTypeOut must be half or nv_bfloat16");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(!fuse_v_mean || std::is_same<DTypeSVAccum, half>::value, "fuse_v_mean only supports half");
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation

  using DTypeOut2 = typename std::conditional<std::is_same<DTypeOut, half>::value, half2, nv_bfloat162>::type;

  const uint32_t num_threads_per_token = head_dim / PACK_SIZE_QK;

  const uint32_t thread_token_base = blockIdx.x * CTA_Q + threadIdx.x / num_threads_per_token;
  if (blockIdx.x * CTA_Q >= cu_seqlen[blockIdx.z + 1] - cu_seqlen[blockIdx.z]) return;

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == SADataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

  constexpr uint32_t QK_SMEM_STRIDE = (DTypeQK == SADataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = head_dim;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>();
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) + get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + get_warp_idx_k<num_warps_q, num_warps_k>();
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) + (head_id / num_kv_groups) * (num_warp_block_k * 4) + get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock) ? 1 : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K) : (CTA_K / WARP_K) * 4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {        
          RO[fq][fv][k] = 0.0f;
        }
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          ((int32_t*)RO[fq][fv])[k] = 0;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      m[fq][k] = -5000000.0f;
      d[fq][k] = 1.0f;
    }
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 : (QK_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 32) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O = (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr = Q + cu_seqlen[batch_id] * stride_seq_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;

  int8_t *K_lane_base_ptr = K + cu_seqlen[batch_id] * stride_seq_k + (head_id / num_kv_groups) * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;

  half   *V_lane_base_ptr = V + cu_seqlen[batch_id] * stride_seq_v + (head_id / num_kv_groups) * stride_h_v + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_seq_v + (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;

  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);

  // for causal masking
  uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t V_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V;

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal
          ? min(kv_len, (bx + 1) * CTA_Q)
          : kv_len,
      CTA_K);

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
    &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  // for num_tiles_qk_inner = 1, we load all Qs in register
  uint32_t RQ[num_tiles_q][4];
  if constexpr (num_tiles_qk_inner == 1)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
      Q_smem_offset_mma = smem_Q.advance_offset_by_row<16>(Q_smem_offset_mma);
    }
  }

  // load K with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
    &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  float q_scale = Q_scale[q_scale_idx];

  float original_sm_scale = sm_scale;
  float dequant_scale = q_scale * K_scale[k_scale_idx + 0 * k_scale_advance_offset];

  sm_scale = original_sm_scale * dequant_scale;

  // load V with predicate
  load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
    &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  K_load_idx_lane_base += CTA_K;
  V_load_idx_lane_base += CTA_K;

#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    // do not apply causal mask and out of bound mask for these iterations
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(RS_f32, RO, m, d, sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    __syncthreads();

    // load K
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma); 
    }

    __syncthreads();
    // load V
    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }
  
  // second last iter, apply causal mask
  if (num_iterations > 1)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    __syncthreads();

    // load K with predicate
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();
    // load V with predicate
    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }

  // last iter, apply causal mask and out of bound mask
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
            RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // check out of bound in the last iter
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    // ensure V is ready
    cp_async::wait_group<0>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();

  }

  // TODO: thread block sync mdo state for num_warps_k > 0

  normalize_d<num_tiles_q, num_tiles_v, DenominatorAccumUnit>(RO, m, d);

  // save the result
  // if (get_warp_idx_k<num_warps_q, num_warps_k>() == 0)
  // {

  // convert half to bfloat16
  if constexpr (std::is_same<DTypeSVAccum, half>::value && std::is_same<DTypeOut, nv_bfloat16>::value)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
      {
        ((nv_bfloat162*)RO[fq][fv])[0] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[0]));
        ((nv_bfloat162*)RO[fq][fv])[1] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[1]));
        ((nv_bfloat162*)RO[fq][fv])[2] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[2]));
        ((nv_bfloat162*)RO[fq][fv])[3] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[3]));
      }
    }
  }

  // add v_mean
  if constexpr (fuse_v_mean)
  {
    DTypeOut2 v_mean[2];
    DTypeOut *V_mean_lane_ptr = V_mean + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + lane_id % 4 * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      v_mean[0] = *((DTypeOut2*)(V_mean_lane_ptr + fv * 16));
      v_mean[1] = *((DTypeOut2*)(V_mean_lane_ptr + 8 + fv * 16));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        ((DTypeOut2*)RO[fq][fv])[0] = __hadd2(((DTypeOut2*)RO[fq][fv])[0], v_mean[0]);
        ((DTypeOut2*)RO[fq][fv])[1] = __hadd2(((DTypeOut2*)RO[fq][fv])[1], v_mean[0]);
        ((DTypeOut2*)RO[fq][fv])[2] = __hadd2(((DTypeOut2*)RO[fq][fv])[2], v_mean[1]);
        ((DTypeOut2*)RO[fq][fv])[3] = __hadd2(((DTypeOut2*)RO[fq][fv])[3], v_mean[1]);
      }
    }
  }

  // save the result to shared memory
  uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        { 
          if constexpr (std::is_same<DTypeOut, half>::value)
          {
            ((half2*)RO_f16)[k] = __float22half2_rn(((float2*)RO[fq][fv])[k]);
          }
          else if constexpr (std::is_same<DTypeOut, nv_bfloat16>::value)
          {
            ((nv_bfloat162*)RO_f16)[k] = __float22bfloat162_rn(((float2*)RO[fq][fv])[k]);
          }
        }

        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

        // ! permuted, make sure you know what you are doing
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_f16[2];
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[1];

        // ! permuted, make sure you know what you are doing
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[2];
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[3]; 
      }
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr = O + cu_seqlen[batch_id] * stride_seq_o + head_id * stride_h_o + (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o + lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / global_to_shared_line_lanes_O, lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(offset_O);
    }

    offset_O = smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

  if constexpr (return_lse)
  { 
    // ! this only works for num_tiles_q = 2
    uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    float *lse_lane_ptr = Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
    uint32_t fq = (lane_id % 4) / 2;
    uint32_t k = (lane_id % 4) % 2;

    if (lse_idx < qo_len)
    {
      lse_lane_ptr[0] = (math::ptx_log2(d[fq][k]) + m[fq][k]);
    }
  }

  // }
}

// tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D]
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
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, paddle::DataType::INT8);
  CHECK_DTYPE(key, paddle::DataType::INT8);
  CHECK_DTYPE(value, paddle::DataType::FLOAT16);    // TODO: there maybe some problem, for bf16 type
  CHECK_DTYPE(query_scale, paddle::DataType::FLOAT32);
  CHECK_DTYPE(key_scale, paddle::DataType::FLOAT32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.shape()[3];
  const int batch_size = query.shape()[0];

  int stride_bz_q = query.strides()[0];
  int stride_bz_k = key.strides()[0];
  int stride_bz_v = value.strides()[0];
  int stride_bz_o = output.strides()[0];

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.shape()[1];
    kv_len = key.shape()[1];
    num_qo_heads = query.shape()[2];
    num_kv_heads = key.shape()[2];
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

    stride_seq_q = query.strides()[1];
    stride_seq_k = key.strides()[1];
    stride_seq_v = value.strides()[1];
    stride_seq_o = output.strides()[1];

    stride_h_q = query.strides()[2];
    stride_h_k = key.strides()[2];
    stride_h_v = value.strides()[2];
    stride_h_o = output.strides()[2];
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.shape()[2];
    kv_len = key.shape()[2];
    num_qo_heads = query.shape()[1];
    num_kv_heads = key.shape()[1];
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

    stride_seq_q = query.strides()[2];
    stride_seq_k = key.strides()[2];
    stride_seq_v = value.strides()[2];
    stride_seq_o = output.strides()[2];

    stride_h_q = query.strides()[1];
    stride_h_k = key.strides()[1];
    stride_h_v = value.strides()[1];
    stride_h_o = output.strides()[1];
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  paddle::Tensor lse = paddle::empty({1});
  if (return_lse)
  {
    lse = paddle::empty({batch_size, num_qo_heads, qo_len}, paddle::DataType::FLOAT32);
  }

  auto output_dtype = output.dtype();   // in [bfloat16 or float16]

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = 32;
            constexpr int WARP_K = 64;

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
            }
            else
            {
              static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
            }

            //                                     smem_Q                                     smem_K                            smem_V                     smem_O
            size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half), CTA_Q * HEAD_DIM * sizeof(half));
            
            auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, SADataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN), float, false, DTypeOut, ComputeUnit::kTensorCore, 
                                                          mask_mode, RETURN_LSE, false>;

            cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

            kernel_func<<<grid, block, smem_max>>>(
              query.data<int8_t>(), 
              key.data<int8_t>(),
              reinterpret_cast<half*>(value.data()),    // reinterpret_cast<nv_bfloat16>(reinterpret_cast<float16*>(value.data<float16>()))
              reinterpret_cast<DTypeOut*>(output.data()),
              (RETURN_LSE) ? lse.data<float>() : nullptr,
              query_scale.data<float>(),
              key_scale.data<float>(),
              nullptr,
              qo_len,
              kv_len,
              num_kv_groups,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_v, stride_seq_v, stride_h_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              sm_scale);
          });
        });
      });
    });
  });

  return {lse};
}

// tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D]
std::vector<paddle::Tensor>  qk_int8_sv_f16_accum_f32_attn_varlen_fwd(paddle::Tensor& query,  // total_seqlen x num_heads x head_dim, int8
                    paddle::Tensor& key,    // total_seqlen x num_heads x head_dim, int8
                    paddle::Tensor& value,  // total_seqlen x num_heads x head_dim, fp16
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& cu_seqlen,
                    int max_seqlen_q,
                    int max_seqlen_k,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, paddle::DataType::INT8);
  CHECK_DTYPE(key, paddle::DataType::INT8);
  CHECK_DTYPE(value, paddle::DataType::FLOAT16);
  CHECK_DTYPE(query_scale, paddle::DataType::FLOAT32);
  CHECK_DTYPE(key_scale, paddle::DataType::FLOAT32);

  CHECK_DIMS(query, 3);
  CHECK_DIMS(key, 3);
  CHECK_DIMS(value, 3);
  CHECK_DIMS(output, 3);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.shape()[2];
  const int batch_size = cu_seqlen.shape()[0] - 1;

  int stride_bz_q = query.strides()[0];
  int stride_bz_k = key.strides()[0];
  int stride_bz_v = value.strides()[0];
  int stride_bz_o = output.strides()[0];

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  qo_len = max_seqlen_q;
  kv_len = max_seqlen_k;

  num_qo_heads = query.shape()[1];
  num_kv_heads = key.shape()[1];

  stride_seq_q = query.strides()[0];
  stride_seq_k = key.strides()[0];
  stride_seq_v = value.strides()[0];
  stride_seq_o = output.strides()[0];

  stride_h_q = query.strides()[1];
  stride_h_k = key.strides()[1];
  stride_h_v = value.strides()[1];
  stride_h_o = output.strides()[1];

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  paddle::Tensor lse = paddle::empty({1});
  if (return_lse)
  {
    lse = paddle::empty({batch_size, num_qo_heads, qo_len}, paddle::DataType::FLOAT32);
  }

  auto output_dtype = output.dtype();   // in [bfloat16 or float16]

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = 32;
            constexpr int WARP_K = 64;

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
            }
            else
            {
              static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
            }

            //                                     smem_Q                                     smem_K                            smem_V                     smem_O
            size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half), CTA_Q * HEAD_DIM * sizeof(half));
            
            auto kernel_func = qk_int_sv_f16_attn_varlen_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, SADataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN), float, false, DTypeOut, ComputeUnit::kTensorCore, mask_mode, RETURN_LSE, false>;

            cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

            kernel_func<<<grid, block, smem_max>>>(
              query.data<int8_t>(), 
              key.data<int8_t>(),
              reinterpret_cast<half*>(value.data()),    // reinterpret_cast<nv_bfloat16>(reinterpret_cast<float16*>(value.data<float16>()))
              reinterpret_cast<DTypeOut*>(output.data()),
              (RETURN_LSE) ? lse.data<float>() : nullptr,
              query_scale.data<float>(),
              key_scale.data<float>(),
              nullptr,
              reinterpret_cast<uint32_t*>(cu_seqlen.data()),
              qo_len,
              kv_len,
              num_kv_groups,
              stride_seq_q, stride_h_q,
              stride_seq_k, stride_h_k,
              stride_seq_v, stride_h_v,
              stride_seq_o, stride_h_o,
              sm_scale);
          });
        });
      });
    });
  });

  return {lse};
}

// tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D]
// impl -> see sageattn.h file
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
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, paddle::DataType::INT8);
  CHECK_DTYPE(key, paddle::DataType::INT8);
  CHECK_DTYPE(value, paddle::DataType::FLOAT16);
  CHECK_DTYPE(query_scale, paddle::DataType::FLOAT32);
  CHECK_DTYPE(key_scale, paddle::DataType::FLOAT32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.shape()[3];
  const int batch_size = query.shape()[0];

  int stride_bz_q = query.strides()[0];
  int stride_bz_k = key.strides()[0];
  int stride_bz_v = value.strides()[0];
  int stride_bz_o = output.strides()[0];

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.shape()[1];
    kv_len = key.shape()[1];
    num_qo_heads = query.shape()[2];
    num_kv_heads = key.shape()[2];
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

    stride_seq_q = query.strides()[1];
    stride_seq_k = key.strides()[1];
    stride_seq_v = value.strides()[1];
    stride_seq_o = output.strides()[1];

    stride_h_q = query.strides()[2];
    stride_h_k = key.strides()[2];
    stride_h_v = value.strides()[2];
    stride_h_o = output.strides()[2];
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.shape()[2];
    kv_len = key.shape()[2];
    num_qo_heads = query.shape()[1];
    num_kv_heads = key.shape()[1];
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

    stride_seq_q = query.strides()[2];
    stride_seq_k = key.strides()[2];
    stride_seq_v = value.strides()[2];
    stride_seq_o = output.strides()[2];

    stride_h_q = query.strides()[1];
    stride_h_k = key.strides()[1];
    stride_h_v = value.strides()[1];
    stride_h_o = output.strides()[1];
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  paddle::Tensor lse = paddle::empty({1});
  if (return_lse)
  {
    lse = paddle::empty({batch_size, num_qo_heads, qo_len}, paddle::DataType::FLOAT32);
  }

  auto output_dtype = output.dtype();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = 32;
            constexpr int WARP_K = 64;

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
            }
            else
            {
              static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
            }

            //                                     smem_Q                                     smem_K                            smem_V                     smem_O
            size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half), CTA_Q * HEAD_DIM * sizeof(half));
            
            auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, SADataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN), half, false, DTypeOut, ComputeUnit::kTensorCore, 
                                                          mask_mode, RETURN_LSE, false>;

            cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

            kernel_func<<<grid, block, smem_max>>>(
              query.data<int8_t>(), 
              key.data<int8_t>(),
              reinterpret_cast<half*>(value.data()),
              reinterpret_cast<DTypeOut*>(output.data()),
              (RETURN_LSE) ? reinterpret_cast<float*>(lse.data()) : nullptr,
              reinterpret_cast<float*>(query_scale.data()),
              reinterpret_cast<float*>(key_scale.data()),
              nullptr,
              qo_len,
              kv_len,
              num_kv_groups,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_v, stride_seq_v, stride_h_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              sm_scale);
          });
        });
      });
    });
  });

  return {lse};
}

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
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, paddle::DataType::INT8);
  CHECK_DTYPE(key, paddle::DataType::INT8);
  CHECK_DTYPE(value, paddle::DataType::FLOAT16);
  CHECK_DTYPE(query_scale, paddle::DataType::FLOAT32);
  CHECK_DTYPE(key_scale, paddle::DataType::FLOAT32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.shape()[3];
  const int batch_size = query.shape()[0];

  int stride_bz_q = query.strides()[0];
  int stride_bz_k = key.strides()[0];
  int stride_bz_v = value.strides()[0];
  int stride_bz_o = output.strides()[0];

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.shape()[1];
    kv_len = key.shape()[1];
    num_qo_heads = query.shape()[2];
    num_kv_heads = key.shape()[2];
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

    stride_seq_q = query.strides()[1];
    stride_seq_k = key.strides()[1];
    stride_seq_v = value.strides()[1];
    stride_seq_o = output.strides()[1];

    stride_h_q = query.strides()[2];
    stride_h_k = key.strides()[2];
    stride_h_v = value.strides()[2];
    stride_h_o = output.strides()[2];
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.shape()[2];
    kv_len = key.shape()[2];
    num_qo_heads = query.shape()[1];
    num_kv_heads = key.shape()[1];
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

    stride_seq_q = query.strides()[2];
    stride_seq_k = key.strides()[2];
    stride_seq_v = value.strides()[2];
    stride_seq_o = output.strides()[2];

    stride_h_q = query.strides()[1];
    stride_h_k = key.strides()[1];
    stride_h_v = value.strides()[1];
    stride_h_o = output.strides()[1];
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  paddle::Tensor lse = paddle::empty({0}, paddle::DataType::FLOAT32);
  if (return_lse)
  {
    lse = paddle::empty({batch_size, num_qo_heads, qo_len}, paddle::DataType::FLOAT32);
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  auto output_dtype = output.dtype();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
              
            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = (HEAD_DIM == 64) ? 32 : 16;
            constexpr int WARP_K = 64;

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
            }
            else
            {
              static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
            }

            //                                     smem_Q                                     smem_K                            smem_V                     smem_O
            size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half), CTA_Q * HEAD_DIM * sizeof(half));
            
            auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, SADataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN), float, true, DTypeOut, ComputeUnit::kTensorCore, 
                                                          mask_mode, RETURN_LSE, false>;

            cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

            kernel_func<<<grid, block, smem_max>>>(
              query.data<int8_t>(), 
              key.data<int8_t>(),
              reinterpret_cast<half*>(value.data()),
              reinterpret_cast<DTypeOut*>(output.data()),
              (RETURN_LSE) ? reinterpret_cast<float*>(lse.data()) : nullptr,
              reinterpret_cast<float*>(query_scale.data()),
              reinterpret_cast<float*>(key_scale.data()),
              nullptr,
              qo_len,
              kv_len,
              num_kv_groups,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_v, stride_seq_v, stride_h_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              sm_scale);
          });
        });
      });
    });
  });
  
  return {lse};
}

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
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_mean);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_mean);

  CHECK_DTYPE(query, paddle::DataType::INT8);
  CHECK_DTYPE(key, paddle::DataType::INT8);
  CHECK_DTYPE(value, paddle::DataType::FLOAT16);
  CHECK_DTYPE(query_scale, paddle::DataType::FLOAT32);
  CHECK_DTYPE(key_scale, paddle::DataType::FLOAT32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_mean, 3);

  const int head_dim = query.shape()[3];
  const int batch_size = query.shape()[0];

  int stride_bz_q = query.strides()[0];
  int stride_bz_k = key.strides()[0];
  int stride_bz_v = value.strides()[0];
  int stride_bz_o = output.strides()[0];

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.shape()[1];
    kv_len = key.shape()[1];
    num_qo_heads = query.shape()[2];
    num_kv_heads = key.shape()[2];
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

    stride_seq_q = query.strides()[1];
    stride_seq_k = key.strides()[1];
    stride_seq_v = value.strides()[1];
    stride_seq_o = output.strides()[1];

    stride_h_q = query.strides()[2];
    stride_h_k = key.strides()[2];
    stride_h_v = value.strides()[2];
    stride_h_o = output.strides()[2];
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.shape()[2];
    kv_len = key.shape()[2];
    num_qo_heads = query.shape()[1];
    num_kv_heads = key.shape()[1];
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

    stride_seq_q = query.strides()[2];
    stride_seq_k = key.strides()[2];
    stride_seq_v = value.strides()[2];
    stride_seq_o = output.strides()[2];

    stride_h_q = query.strides()[1];
    stride_h_k = key.strides()[1];
    stride_h_v = value.strides()[1];
    stride_h_o = output.strides()[1];
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  paddle::Tensor lse = paddle::empty({0}, paddle::DataType::FLOAT32);
  if (return_lse)
  {
    lse = paddle::empty({batch_size, num_qo_heads, qo_len}, paddle::DataType::FLOAT32);
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  auto output_dtype = output.dtype();
  auto value_mean_dtype = value_mean.dtype();

  PD_CHECK(value_mean_dtype == output_dtype, "value_mean and output must have the same dtype");

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
              
            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = 32;
            constexpr int WARP_K = 64;

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
            }
            else
            {
              static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
            }

            CHECK_SHAPE(value_mean, batch_size, num_kv_heads, head_dim);

            //                                     smem_Q                                     smem_K                            smem_V                     smem_O
            size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half), CTA_Q * HEAD_DIM * sizeof(half));
            
            auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, SADataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN), half, false, DTypeOut, ComputeUnit::kTensorCore, 
                                                          mask_mode, RETURN_LSE, true>;

            cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

            kernel_func<<<grid, block, smem_max>>>(
              query.data<int8_t>(), 
              key.data<int8_t>(),
              reinterpret_cast<half*>(value.data()),
              reinterpret_cast<DTypeOut*>(output.data()),
              (RETURN_LSE) ? reinterpret_cast<float*>(lse.data()) : nullptr,
              reinterpret_cast<float*>(query_scale.data()),
              reinterpret_cast<float*>(key_scale.data()),
              reinterpret_cast<DTypeOut*>(value_mean.data()),
              qo_len,
              kv_len,
              num_kv_groups,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_v, stride_seq_v, stride_h_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              sm_scale);
          });
        });
      });
    });
  });

  return {lse};
}

//
//  =========== Exposed to Outside API - ARCH: SM80 ===========
//

std::vector<paddle::Tensor> sage_attention_fwd(paddle::Tensor& q,
                                               paddle::Tensor& k,
                                               paddle::Tensor& v,
                                               paddle::Tensor& km,
                                               paddle::optional<paddle::Tensor>& vm,
                                               float sm_scale,
                                               std::string qk_quant_gran,
                                               std::string pv_accum_dtype,
                                               int tensor_layout,
                                               bool is_causal,
                                               bool smooth_k,
                                               bool smooth_v,
                                               bool return_lse)
{
  int _is_causal = int(is_causal);
  int _qk_quant_gran = (qk_quant_gran == std::string("per_thread")) ? 3 : 2;
  int _return_lse = int(return_lse);

  PD_CHECK(pv_accum_dtype == std::string("fp16+fp32") || pv_accum_dtype == std::string("fp32") || pv_accum_dtype == std::string("fp16"), 
            "pv_accum_dtype must be either fp16, fp32 or fp16+fp32");
  auto pv_accum_dtype_const = (pv_accum_dtype == std::string("fp16+fp32")) ? paddle::DataType::UNDEFINED : 
                                (pv_accum_dtype == std::string("fp16")) ? paddle::DataType::FLOAT16 : paddle::DataType::FLOAT32;

  PD_CHECK(q.shape()[3] == 64 || q.shape()[3] == 128, "head_dim must be either 64 or 128");
  PD_CHECK(q.strides()[3] == 1 && k.strides()[3] == 1 && v.strides()[3] == 1, "Last dim of qkv must be contiguous.");

  int seq_dim = (tensor_layout == 0) ? 1 : 2;

  constexpr int BLKQ = 128;
  int WARPQ = (q.shape()[3] == 128 && pv_accum_dtype_const == paddle::DataType::UNDEFINED) ? 16 : 32;
  constexpr int BLKK = 64;
  std::vector<paddle::Tensor>&& quant_results = per_warp_int8_cuda(q, k, km, BLKQ, WARPQ, BLKK, tensor_layout); // q_int8, q_scale, k_int8, k_scale
  paddle::Tensor o = paddle::empty(v.shape(), v.dtype(), paddle::GPUPlace());

  if (pv_accum_dtype_const == paddle::DataType::UNDEFINED || pv_accum_dtype_const == paddle::DataType::FLOAT32) {
    if (smooth_v) smooth_v = false;
  }

  switch (pv_accum_dtype_const) {
    case paddle::DataType::FLOAT32: {
      v = v.cast(paddle::DataType::FLOAT16);
      qk_int8_sv_f16_accum_f32_attn_fwd(quant_results[0], quant_results[2], v, o, quant_results[1], quant_results[3], tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse);
      break;
    }
    case paddle::DataType::FLOAT16: {
      if (smooth_v && vm) {
        std::vector<paddle::Tensor>&& sub_mean_results = sub_mean(v, vm.get(), tensor_layout); // smooth_v, v_mean
        qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_fwd(quant_results[0], quant_results[2], sub_mean_results[0], o, quant_results[1], quant_results[3], sub_mean_results[1], tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse);
      } else {
        v = v.cast(paddle::DataType::FLOAT16);
        qk_int8_sv_f16_accum_f16_attn_fwd(quant_results[0], quant_results[2], v, o, quant_results[1], quant_results[3], tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse);
      }
      break;
    }
    case paddle::DataType::UNDEFINED: {
      v = v.cast(paddle::DataType::FLOAT16);
      qk_int8_sv_f16_accum_f16_attn_inst_buf_fwd(quant_results[0], quant_results[2], v, o, quant_results[1], quant_results[3], 
                                                  tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse);
      break;
    }
    default: {
      throw std::runtime_error("pv_accum_dtype must be fp32, fp16 or fp16+fp32");
      break;
    }
  }

  return {o, quant_results[0], quant_results[2]};
}

std::vector<std::vector<int64_t>> sage_attention_InferShape(
  const std::vector<int64_t> query_shape, 
  const std::vector<int64_t> key_shape, 
  const std::vector<int64_t> value_shape,
  const std::vector<int64_t> km_shape,
  const paddle::optional<std::vector<int64_t>>& vm_shape) {
    return {value_shape};
}

std::vector<paddle::DataType> sage_attention_InferDtype(
  const paddle::DataType A_dtype,
  const paddle::DataType B_dtype,
  const paddle::DataType C_dtype,
  const paddle::DataType D_dtype,
  const paddle::optional<paddle::DataType>& E_dtype) {
  return {C_dtype};
}

PD_BUILD_OP(sage_attention)
    .Inputs({"q", "k", "v", "km", paddle::Optional("vm")})
    .Outputs({"o", "q_int8", "k_int8"})
    .Attrs({"sm_scale: float",
            "qk_quant_gran: std::string",
            "pv_accum_dtype: std::string",
            "tensor_layout: int",
            "is_causal: bool",
            "smooth_k: bool",
            "smooth_v: bool",
            "return_lse: bool"})
    .SetKernelFn(PD_KERNEL(sage_attention_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(sage_attention_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(sage_attention_InferDtype));


// VARLEN

std::vector<paddle::Tensor> sage_attention_varlen_fwd(paddle::Tensor& q,  // total_seqlen x num_head x head_dim
                                               paddle::Tensor& k,         // total_seqlen x num_head x head_dim
                                               paddle::Tensor& v,         // total_seqlen x num_head x head_dim
                                               paddle::Tensor& cu_seqlen_q,
                                               paddle::Tensor& segment_ids,
                                               paddle::optional<paddle::Tensor>& vm,
                                               int max_seqlen_q,
                                               int max_seqlen_k,
                                               float sm_scale,
                                               std::string qk_quant_gran,
                                               std::string pv_accum_dtype,
                                               int tensor_layout,
                                               bool is_causal,
                                               bool smooth_k,
                                               bool smooth_v,
                                               bool return_lse)
{
  int _is_causal = int(is_causal);
  int _qk_quant_gran = (qk_quant_gran == std::string("per_thread")) ? 3 : 2;
  int _return_lse = int(return_lse);

  PD_CHECK(pv_accum_dtype == std::string("fp16+fp32") || pv_accum_dtype == std::string("fp32") || pv_accum_dtype == std::string("fp16"), 
            "pv_accum_dtype must be either fp16, fp32 or fp16+fp32");
  auto pv_accum_dtype_const = (pv_accum_dtype == std::string("fp16+fp32")) ? paddle::DataType::UNDEFINED : 
                                (pv_accum_dtype == std::string("fp16")) ? paddle::DataType::FLOAT16 : paddle::DataType::FLOAT32;

  PD_CHECK(q.shape()[2] == 64 || q.shape()[2] == 128, "head_dim must be either 64 or 128");
  PD_CHECK(q.strides()[2] == 1 && k.strides()[2] == 1 && v.strides()[2] == 1, "Last dim of qkv must be contiguous.");

  int seq_dim = (tensor_layout == 0) ? 1 : 2;

  constexpr int BLKQ = 128;
  int WARPQ = (q.shape()[2] == 128 && pv_accum_dtype_const == paddle::DataType::UNDEFINED) ? 16 : 32;
  constexpr int BLKK = 64;
  std::vector<paddle::Tensor>&& quant_results = per_warp_int8_varlen_cuda_fwd(q, k, cu_seqlen_q, segment_ids, max_seqlen_q, max_seqlen_k, BLKQ, WARPQ, BLKK); // q_int8, q_scale, k_int8, k_scale
  paddle::Tensor o = paddle::empty(q.shape(), q.dtype(), paddle::GPUPlace());

  if (pv_accum_dtype_const == paddle::DataType::UNDEFINED || pv_accum_dtype_const == paddle::DataType::FLOAT32) {
    if (smooth_v) smooth_v = false;
  }

  switch (pv_accum_dtype_const) {
    case paddle::DataType::FLOAT32: {
      v = v.cast(paddle::DataType::FLOAT16);
      qk_int8_sv_f16_accum_f32_attn_varlen_fwd(quant_results[0], quant_results[2], v, o, quant_results[1], quant_results[3], 
                                                cu_seqlen_q, 
                                                max_seqlen_q, max_seqlen_k,
                                                tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse);
      break;
    }
    default: {
      throw std::runtime_error("pv_accum_dtype must be fp32, fp16 or fp16+fp32");
      break;
    }
  }

  return {o, quant_results[0], quant_results[2], quant_results[4]};
}

std::vector<std::vector<int64_t>> sage_attention_varlen_InferShape(
  const std::vector<int64_t> query_shape, 
  const std::vector<int64_t> key_shape, 
  const std::vector<int64_t> value_shape,
  const std::vector<int64_t> cu_seqlen_shape,
  const std::vector<int64_t> segment_ids_shape,
  const paddle::optional<std::vector<int64_t>>& vm_shape) {
    return {value_shape, query_shape};
}

std::vector<paddle::DataType> sage_attention_varlen_InferDtype(
  const paddle::DataType A_dtype,
  const paddle::DataType B_dtype,
  const paddle::DataType C_dtype,
  const paddle::DataType D_dtype,
  const paddle::DataType E_dtype,
  const paddle::optional<paddle::DataType>& F_dtype) {
  return {C_dtype, paddle::DataType::INT8};
}

PD_BUILD_OP(sage_attention_varlen)
    .Inputs({"q", "k", "v", "cu_seqlen", "segment_ids", paddle::Optional("vm")})
    .Outputs({"o", "q_int8", "k_int8", "km"})
    .Attrs({"max_seqlen_q: int",
            "max_seqlen_k: int",
            "sm_scale: float",
            "qk_quant_gran: std::string",
            "pv_accum_dtype: std::string",
            "tensor_layout: int",
            "is_causal: bool",
            "smooth_k: bool",
            "smooth_v: bool",
            "return_lse: bool"})
    .SetKernelFn(PD_KERNEL(sage_attention_varlen_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(sage_attention_varlen_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(sage_attention_varlen_InferDtype));