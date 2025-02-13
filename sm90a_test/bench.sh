#!/bin/bash

rm -rf ./inputs && mkdir inputs
export CUDA_VISIBLE_DEVICES=6
python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024
python test_append_attn_std.py

ncu --nvtx --nvtx-include "append_attn" python test_append_attn_std.py
ncu --nvtx --nvtx-include "SA_paddle/FA3_torch/SA_torch" python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024
ncu --nvtx --nvtx-include "SA_paddle" python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024
ncu --nvtx --nvtx-include "FA3_torch" python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024
ncu --nvtx --nvtx-include "SA_torch" python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024