#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o new_fused_op_th --cuda-memory-usage true python bench_sm90_new.py