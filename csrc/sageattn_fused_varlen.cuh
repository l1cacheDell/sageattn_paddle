#pragma once
#include "paddle/extension.h"

std::vector<paddle::Tensor> per_warp_int8_varlen_cuda_fwd(paddle::Tensor& q,  // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& k,    // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& cu_seqlen_q,
                                                    paddle::Tensor& segment_ids,
                                                    int max_seq_len_q,
                                                    int max_seq_len_k,
                                                    int BLKQ,
                                                    int WARPQ,
                                                    int BLKK);

std::vector<paddle::Tensor> per_channel_varlen_fp8(paddle::Tensor& v, // total_seqlen x num_head x head_dim
                                                  paddle::Tensor& cu_seqlen_v,
                                                  paddle::Tensor& padded_seqlen,  // the same shape as cu_seq_len_v
                                                  int max_seq_len_v,
                                                  int max_seq_len_v_padded,
                                                  int tensor_layout,
                                                  float scale_max,
                                                  bool smooth_v);