# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import numpy as np
import paddle
import paddlenlp_ops
np.random.seed(2024)
paddle.seed(2024)

def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time)
    cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = paddle.sum(seq_lens_this_time)
    padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i]
        cum_offset = cum_offsets[i]
        for j in range(seq_len_now):
            padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
        cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return padding_offsets, cum_offsets[:-1], cu_seqlens_q, cu_seqlens_k
run_time = 1000
warm_up = 200
block_size = 64

# normal setting
head_dim_qk = 128
head_dim_v= 128
num_q_head = 64
num_kv_head = 64

max_dec_len = 1
dtype = "float16"
use_neox_rotary_style = False
# prefille
max_length = 1026
bsz = 2
input_length = 1024

def test_append_c16_attention():
    seq_lens_enc = [
        input_length,
    ] * bsz
    seq_lens_dec = [
        0,
    ] * bsz
    seq_lens_this_time = [
        input_length,
    ] * bsz
    max_enc_len_this_time = max(seq_lens_enc)
    max_dec_len_this_time = max(seq_lens_dec)
    max_enc_len_this_time = paddle.to_tensor([max_enc_len_this_time], "int32", place=paddle.CPUPlace())
    max_dec_len_this_time = paddle.to_tensor([max_dec_len_this_time], "int32", place=paddle.CPUPlace())
    token_num = sum(seq_lens_this_time)
    block_num_per_seq = (max_length + block_size - 1) // block_size
    max_block_num = block_num_per_seq * bsz
    free_list = list(range(max_block_num - 1, -1, -1))
    block_tables = paddle.zeros(shape=(bsz, block_num_per_seq), dtype="int32") * (-1)
    for i in range(bsz):
        need_block_num = (seq_lens_dec[i] + max_dec_len + block_size - 1) // block_size
        for j in range(need_block_num):
            block_id = free_list.pop()
            block_tables[i, j] = block_id
    seq_lens_encoder = paddle.to_tensor(seq_lens_enc, "int32")
    seq_lens_this_time = paddle.to_tensor(seq_lens_this_time, "int32")
    seq_lens_decoder = paddle.to_tensor(seq_lens_dec, "int32")
    padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k = get_padding_offset(bsz, max_length, seq_lens_this_time)
    q_varlen_shape = [token_num, (num_q_head + num_kv_head) * head_dim_qk + num_kv_head * head_dim_v]
    cache_k_shape = (
        max_block_num,
        num_kv_head,
        block_size,
        head_dim_qk,
    )
    cache_v_shape = (
        max_block_num,
        num_kv_head,
        block_size,
        head_dim_v,
    )

    (
        encoder_batch_ids,
        encoder_tile_ids_per_batch,
        encoder_num_blocks,
        kv_batch_ids,
        kv_tile_ids_per_batch,
        kv_num_blocks,
        decoder_batch_ids,
        decoder_tile_ids_per_batch,
        decoder_num_blocks,
        max_len_kv,
    ) = paddlenlp_ops.get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        max_enc_len_this_time,
        max_dec_len_this_time,
        seq_lens_this_time,
        cum_offsets,
        num_q_head // num_kv_head,
        block_size,
        1,
    )

    qkv = paddle.randn(shape=q_varlen_shape).astype(dtype)
    print(qkv.shape)
    cache_k = paddle.ones(shape=cache_k_shape).astype(dtype)
    cache_v = paddle.ones(shape=cache_v_shape).astype(dtype)
    softmax_scale = head_dim_qk ** (-0.5)
    # base_out = mqa_attention(qkv, token_num, softmax_scale, max_dec_len)
   
    s_time = 0
    for i in range(run_time + warm_up):
        if i == warm_up:
            s_time = time.time()
        out = paddlenlp_ops.append_attention(
            qkv,
            cache_k,
            cache_v,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            padding_offsets,
            cum_offsets,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            max_enc_len_this_time,
            max_dec_len_this_time,
            max_len_kv,
            None,  # rotary_embs
            None,  # attn_mask
            None,  # qkv_bias
            None,  # qkv_scale
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # out_linear_shifts
            None,  # out_linear_smooths
            "bf16",  # compute_type
            "none",  # cache_quant_type
            use_neox_rotary_style,  # use_neox_rotary_style
            max_length,  # max_input_length
            softmax_scale,  # softmax_scale
            127.0,  # quant_max_bound
            -127.0,  # quant_min_bound
            0.0,  # out_linear_in_scale
            1,  # speculate_max_draft_token_num
            True,  # causal
            False,  # speculate_decoder
        )[0]
        paddle.device.synchronize()
    # print(out)
    e_time = time.time()

    print("prefill bsz:{}, num_kv_head:{}, cost_time:{}ms".format(bsz, num_kv_head, (e_time - s_time) / run_time * 1000))
if __name__ == "__main__":
    test_append_c16_attention()