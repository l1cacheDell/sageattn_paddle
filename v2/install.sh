#!/bin/bash

export PATH=/usr/local/cuda-12.4/bin:$PATH
nvcc -V

pip uninstall sageattn_dsk_v2 -y
rm -rf ./build
rm -rf ./dist
rm -rf ./sageattn_dsk_v2.egg-info
python setup_cuda.py install > compile_output.txt 2>&1
grep "error:" compile_output.txt > errors_only.txt
cat errors_only.txt

# python test.py