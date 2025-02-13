#!/bin/bash

CUDA_VISIBLE_DEVICES=6 compute-sanitizer python test_sageattn_qk_int8_pv_fp8_cuda_sm90a.py > compile_output.txt 2>&1
grep "error:" compile_output.txt > errors_only.txt
cat errors_only.txt