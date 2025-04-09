#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

bsz=32

seqlen=1024
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o pf_bsz${bsz}_seqlen${seqlen}_fp16 python profile_new.py --seqlen ${seqlen} --bsz ${bsz}

seqlen=4096
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o pf_bsz${bsz}_seqlen${seqlen}_fp16 python profile_new.py --seqlen ${seqlen} --bsz ${bsz}

seqlen=8192
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o pf_bsz${bsz}_seqlen${seqlen}_fp16 python profile_new.py --seqlen ${seqlen} --bsz ${bsz}

seqlen=16384
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o pf_bsz${bsz}_seqlen${seqlen}_fp16 python profile_new.py --seqlen ${seqlen} --bsz ${bsz}

seqlen=32768
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o pf_bsz${bsz}_seqlen${seqlen}_fp16 python profile_new.py --seqlen ${seqlen} --bsz ${bsz}

seqlen=65536
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o pf_bsz${bsz}_seqlen${seqlen}_fp16 python profile_new.py --seqlen ${seqlen} --bsz ${bsz}
