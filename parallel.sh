#!/bin/bash


# nsys profile --output=seq_len1k python test_paddle_fa_sa.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 1024
# nsys profile --output=seq_len4k python test_paddle_fa_sa.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 4096
# nsys profile --output=seq_len16k python test_paddle_fa_sa.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 16384
# nsys profile --output=seq_len32k python test_paddle_fa_sa.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 32768
# nsys profile --output=seq_len128k python test_paddle_fa_sa.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 131072


nsys profile --output=seq_len1k_triton python test_paddle_fa_sa_triton_ops.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 1024
nsys profile --output=seq_len4k_triton python test_paddle_fa_sa_triton_ops.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 4096
nsys profile --output=seq_len16k_triton python test_paddle_fa_sa_triton_ops.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 16384
nsys profile --output=seq_len32k_triton python test_paddle_fa_sa_triton_ops.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 32768
# nsys profile --output=seq_len128k_triton python test_paddle_fa_sa_triton_ops.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 131072
