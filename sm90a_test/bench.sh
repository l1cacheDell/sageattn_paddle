#!/bin/bash

rm -rf ./inputs && mkdir inputs
rm -rf ./*.nsys-rep
rm -rf ./*.sqlite
export CUDA_VISIBLE_DEVICES=6
export PATH=/opt/nvidia/nsight-systems/2023.1.1/bin/:$PATH
SEQ_LEN=8192
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o all3_${SEQ_LEN} --cuda-memory-usage true python test_all.py --batch_size 2 --num_heads 64 --head_dim 128 --seq_len ${SEQ_LEN}
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o append_attn_${SEQ_LEN} --cuda-memory-usage true python test_append_attn_std.py

# ncu --nvtx --nvtx-include "append_attn" python test_append_attn_std.py
# ncu --nvtx --nvtx-include "SA_paddle/FA3_torch/SA_torch" python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024
# ncu --nvtx --nvtx-include "SA_paddle" python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024
# ncu --nvtx --nvtx-include "FA3_torch" python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024
# ncu --nvtx --nvtx-include "SA_torch" python test_all.py --batch_size 1 --num_heads 64 --head_dim 128 --seq_len 1024