#!/bin/bash

rm -rf ./inputs && mkdir inputs
rm -rf ./*.nsys-rep
rm -rf ./*.sqlite
export CUDA_VISIBLE_DEVICES=5
export PATH=/opt/nvidia/nsight-systems/2023.1.1/bin/:$PATH
SEQ_LEN=1024
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o fp8_${SEQ_LEN} python test_flashattn_fp8.py --batch_size 2 --num_heads 64 --head_dim 128 --seq_len ${SEQ_LEN}
SEQ_LEN=4096
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o fp8_${SEQ_LEN} python test_flashattn_fp8.py --batch_size 2 --num_heads 64 --head_dim 128 --seq_len ${SEQ_LEN}
SEQ_LEN=8192
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o fp8_${SEQ_LEN} python test_flashattn_fp8.py --batch_size 2 --num_heads 64 --head_dim 128 --seq_len ${SEQ_LEN}
SEQ_LEN=16384
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o fp8_${SEQ_LEN} python test_flashattn_fp8.py --batch_size 2 --num_heads 64 --head_dim 128 --seq_len ${SEQ_LEN}
SEQ_LEN=32768
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o fp8_${SEQ_LEN} python test_flashattn_fp8.py --batch_size 2 --num_heads 64 --head_dim 128 --seq_len ${SEQ_LEN}
SEQ_LEN=65536
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o fp8_${SEQ_LEN} python test_flashattn_fp8.py --batch_size 2 --num_heads 64 --head_dim 128 --seq_len ${SEQ_LEN}
