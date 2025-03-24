#pragma once
#include "paddle/extension.h"

std::vector<paddle::Tensor> per_warp_int8_cuda(paddle::Tensor& q,
                                            paddle::Tensor& k,
                                            paddle::Tensor& km,
                                            int BLKQ,
                                            int WARPQ,
                                            int BLKK,
                                            int tensor_layout);
                                
std::vector<paddle::Tensor> per_warp_int8_varlen_cuda_fwd(paddle::Tensor& q,  // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& k,    // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& cu_seqlen_q,
                                                    paddle::Tensor& segment_ids,
                                                    int max_seq_len_q,
                                                    int max_seq_len_k,
                                                    int BLKQ,
                                                    int WARPQ,
                                                    int BLKK);
                    
std::vector<paddle::Tensor> per_channel_fp8(paddle::Tensor& v,
                                            int tensor_layout,
                                            float scale_max,
                                            bool smooth_v);

std::vector<paddle::Tensor> sub_mean(paddle::Tensor& v,
                                    paddle::Tensor& vm,
                                    int tensor_layout);