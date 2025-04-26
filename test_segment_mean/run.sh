#!/bin/bash
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so
export PYTHONPATH=/root/paddlejob/workspace/env_run/output/dongyazhu/PaddleNLP
export CUDA_VISIBLE_DEVICES=5

# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o qwen_nsys \
# compute-sanitizer \
# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o test_n \
python test.py
