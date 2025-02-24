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
import os
from utils import precision_cmp_paddle
import nvtx
from loguru import logger
# np.random.seed(2024)
# paddle.seed(2024)

paddle.set_printoptions(threshold=10000, edgeitems=10000, linewidth=10000)

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

run_time = 5
warm_up = 5
block_size = 64

# normal setting
# head_dim_qk = 128
# head_dim_v= 128
# num_q_head = 64
# num_kv_head = 64

max_dec_len = 1
dtype = "bfloat16"
use_neox_rotary_style = False

# prefille
max_length = 1026
# bsz = 2
# input_length = 1024

input_dir = "./inputs"
q_npy = np.load(os.path.join(input_dir, "q.npy"))
k_npy = np.load(os.path.join(input_dir, "k.npy"))
v_npy = np.load(os.path.join(input_dir, "v.npy"))
o_fa3_npy = np.load(os.path.join(input_dir, "o.npy"))

q_paddle = paddle.to_tensor(q_npy, dtype=paddle.float16)
k_paddle = paddle.to_tensor(k_npy, dtype=paddle.float16)
v_paddle = paddle.to_tensor(v_npy, dtype=paddle.float16)
o_fa3_paddle = paddle.to_tensor(o_fa3_npy, dtype=paddle.float16)

# for some reasons, we follow [bsz, seq_len, num_head, head_dim] pattern
# MHA
bsz = q_paddle.shape[0]
input_length = q_paddle.shape[1]
max_length = input_length + 2

head_dim_qk = q_paddle.shape[3]
head_dim_v= q_paddle.shape[3]
num_q_head = q_paddle.shape[2]
num_kv_head = q_paddle.shape[2]

def test_append_c16_attention():
    global o_fa3_paddle
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
        need_block_num = (seq_lens_enc[i] + max_dec_len + block_size - 1) // block_size
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

    # qkv = paddle.randn(shape=q_varlen_shape).astype(dtype)
    # print(qkv.shape)

    q_flat = paddle.view(q_paddle, [bsz, input_length, head_dim_qk * num_q_head])
    k_flat = paddle.view(k_paddle, [bsz, input_length, head_dim_qk * num_kv_head])
    v_flat = paddle.view(v_paddle, [bsz, input_length, head_dim_v * num_kv_head])
    qkv_combined = paddle.concat([q_flat, k_flat, v_flat], axis=-1)
    qkv = paddle.view(qkv_combined, [bsz * input_length, (num_q_head + num_kv_head) * head_dim_qk + head_dim_v * num_kv_head]).astype(dtype)
    cache_k = paddle.ones(shape=cache_k_shape).astype(dtype)
    cache_v = paddle.ones(shape=cache_v_shape).astype(dtype)
    softmax_scale = head_dim_qk ** (-0.5)
    # base_out = mqa_attention(qkv, token_num, softmax_scale, max_dec_len)
    
    s_time = 0
    for i in range(run_time + warm_up):
        if i == warm_up:
            s_time = time.time()
        # if i == run_time + warm_up - 1:
        transformer_nvtx = nvtx.start_range(message="append_attn_c16", color="red")
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
        # if i == run_time + warm_up - 1:
        nvtx.end_range(transformer_nvtx)
    # print(out)
    # out = paddle.reshape(out, [bsz, input_length, num_q_head, head_dim_qk])
    out = paddle.reshape(out, [1, -1])
    o_fa3_paddle = paddle.reshape(o_fa3_paddle, [1, -1])
    sim, l1, max_diff = precision_cmp_paddle(o_fa3_paddle, out)
    logger.debug(f"Cos sim: {sim}, L1: {l1}, Max Diff: {max_diff}")
    e_time = time.time()
    # print(out[:512])
    # print(f"Accu: {(out - o_fa3_paddle)[:100]}")
    print("prefill bsz:{}, num_kv_head:{}, cost_time:{}ms".format(bsz, num_kv_head, (e_time - s_time) / run_time * 1000))


def test_append_c8_attention():
    global max_dec_len, o_fa3_paddle
    q_flat = paddle.view(q_paddle, [bsz, input_length, head_dim_qk * num_q_head])
    k_flat = paddle.view(k_paddle, [bsz, input_length, head_dim_qk * num_kv_head])
    v_flat = paddle.view(v_paddle, [bsz, input_length, head_dim_v * num_kv_head])
    qkv_combined = paddle.concat([q_flat, k_flat, v_flat], axis=-1)
    qkv = paddle.view(qkv_combined, [bsz * input_length, (num_q_head + num_kv_head) * head_dim_qk + head_dim_v * num_kv_head]).astype("int32")
    # cache_k = paddle.ones(shape=cache_k_shape).astype(dtype)
    # cache_v = paddle.ones(shape=cache_v_shape).astype(dtype)
    softmax_scale = head_dim_qk ** (-0.5)
    # qkv = paddle.randn(shape=q_varlen_shape).astype("int32")
    max_input_length = input_length + 2
    block_num_per_seq = (max_length + block_size - 1) // block_size
    max_block_num = block_num_per_seq * bsz
    cache_shape_c8 = (
        max_block_num,
        num_kv_head,
        block_size,
        head_dim_qk,
    )
    head_dim = head_dim_qk
    cache_k_c8 = paddle.ones(shape=cache_shape_c8).astype("uint8")
    cache_v_c8 = paddle.ones(shape=cache_shape_c8).astype("uint8")
    rotary_embs_shape = [2, max_length, 1, head_dim if use_neox_rotary_style else head_dim // 2]
    rotary_embs = paddle.randn(shape=rotary_embs_shape).astype("float32")
    qkv_bias_shape = [num_q_head + 2 * num_kv_head, head_dim]
    qkv_scale = paddle.ones(shape=qkv_bias_shape).astype("float32")
    qkv_bias = paddle.randn(shape=qkv_bias_shape).astype(dtype)

    cache_k_scale = paddle.ones(shape=[num_kv_head]).astype(dtype)
    cache_v_scale = paddle.ones(shape=[num_kv_head]).astype(dtype)
    cache_k_out_scale = paddle.ones(shape=[num_kv_head]).astype(dtype)
    cache_v_out_scale = paddle.ones(shape=[num_kv_head]).astype(dtype)
    shift_bias = paddle.zeros(shape=[num_q_head, head_dim]).astype(dtype)
    smooth_weight = paddle.ones(shape=[num_q_head, head_dim]).astype(dtype)
    no_tensor = paddle.zeros(shape=[1]).astype("int32")
    s_time = 0

    # c8 settings
    seq_lens_enc = [input_length,] * bsz
    seq_lens_dec = [0, ] * bsz
    max_enc_len = max(seq_lens_enc)
    max_dec_len = max(seq_lens_dec)
    max_enc_dec_len = max(seq_lens_enc + seq_lens_dec)
    token_num = sum(seq_lens_enc)
    max_enc_dec_len = max_enc_dec_len
    seq_lens_encoder = paddle.to_tensor(seq_lens_enc, "int32")
    seq_lens_this_time = seq_lens_encoder
    seq_lens_decoder = paddle.to_tensor(seq_lens_dec, "int32")

    padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k = get_padding_offset(
        bsz, max_input_length, seq_lens_this_time
    )
    block_tables = paddle.zeros(shape=(bsz, block_num_per_seq), dtype="int32") * (-1)
    for i in range(bsz):
        free_list = list(range(max_block_num - 1, -1, -1))
        need_block_num = (seq_lens_enc[i] + max_dec_len + block_size - 1) // block_size
        for j in range(need_block_num):
            block_id = free_list.pop()
            block_tables[i, j] = block_id
    encoder_block_shape_q = 64
    decoder_block_shape_q = 16
    max_partition_size = 512
    encoder_max_partition_size = 32768
    max_enc_len_this_time = max(seq_lens_enc)
    max_dec_len_this_time = max(seq_lens_dec)
    max_enc_len_this_time = paddle.to_tensor([max_enc_len_this_time], "int32", place=paddle.CPUPlace())
    max_dec_len_this_time = paddle.to_tensor([max_dec_len_this_time], "int32", place=paddle.CPUPlace())
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
    for i in range(run_time + warm_up):
        if i == warm_up:
            s_time = time.time()
        transformer_nvtx = nvtx.start_range(message="append_attn_c8", color="blue")
        out = paddlenlp_ops.append_attention(
            qkv,
            cache_k_c8,
            cache_v_c8,
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
            None,   # rotary embeds
            None, # attn_mask
            qkv_bias,
            qkv_scale,
            cache_k_scale,
            cache_v_scale,
            cache_k_out_scale,
            cache_v_out_scale,
            None, # cache_k_zp
            None, # cache_v_zp
            shift_bias,
            smooth_weight,
            "bf16",
            "cache_int8", # cache_quant_type
            use_neox_rotary_style,
            max_length,
            softmax_scale,
            127.0,
            -127.0,
            0.0,  # out_linear_in_scale
            # encoder_block_shape_q, # encoder_block_shape_q
            # decoder_block_shape_q, # decoder_block_shape_q
            # max_partition_size, # max_partition_size
            # encoder_max_partition_size, # encoder_max_partition_size
            1,  # speculate_max_draft_token_num
            True,  # causal
            False,  # speculate_decoder
        )[0]
        paddle.device.synchronize()
        nvtx.end_range(transformer_nvtx)
    out = paddle.reshape(out, [1, -1])
    o_fa3_paddle = paddle.reshape(o_fa3_paddle, [1, -1])
    sim, l1, max_diff = precision_cmp_paddle(o_fa3_paddle, out)
    logger.debug(f"Cos sim: {sim}, L1: {l1}, Max Diff: {max_diff}")
    e_time = time.time()
    # print(out[:512])
    # print(f"Accu: {(out - o_fa3_paddle)[:512]}")
    print("prefill c8 attention cost_time: {} ms".format((e_time - s_time) / run_time * 1000))

if __name__ == "__main__":
    test_append_c16_attention()
    test_append_c8_attention()