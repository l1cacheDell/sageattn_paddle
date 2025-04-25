#!/bin/bash
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so
export PYTHONPATH=/root/paddlejob/workspace/env_run/output/dongyazhu/PaddleNLP
export CUDA_VISIBLE_DEVICES=4

# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o qwen_nsys \
# compute-sanitizer \
python test.py
