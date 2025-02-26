#!/bin/bash

set -e
HEAD_DIM=128

SEQ_LEN=1024
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o all3_${SEQ_LEN} --cuda-memory-usage true python test_dsk_v3.py --batch_size 2 --num_heads ${HEAD_DIM} --head_dim 128 --seq_len ${SEQ_LEN}
# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o append_attn_${SEQ_LEN} --cuda-memory-usage true python test_append_attn_std.py
SEQ_LEN=4096
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o all3_${SEQ_LEN} --cuda-memory-usage true python test_dsk_v3.py --batch_size 2 --num_heads ${HEAD_DIM} --head_dim 128 --seq_len ${SEQ_LEN}
# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o append_attn_${SEQ_LEN} --cuda-memory-usage true python test_append_attn_std.py
SEQ_LEN=8192
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o all3_${SEQ_LEN} --cuda-memory-usage true python test_dsk_v3.py --batch_size 2 --num_heads ${HEAD_DIM} --head_dim 128 --seq_len ${SEQ_LEN}
# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o append_attn_${SEQ_LEN} --cuda-memory-usage true python test_append_attn_std.py
SEQ_LEN=16384
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o all3_${SEQ_LEN} --cuda-memory-usage true python test_dsk_v3.py --batch_size 2 --num_heads ${HEAD_DIM} --head_dim 128 --seq_len ${SEQ_LEN}
# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o append_attn_${SEQ_LEN} --cuda-memory-usage true python test_append_attn_std.py
SEQ_LEN=32768
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o all3_${SEQ_LEN} --cuda-memory-usage true python test_dsk_v3.py --batch_size 2 --num_heads ${HEAD_DIM} --head_dim 128 --seq_len ${SEQ_LEN}
# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o append_attn_${SEQ_LEN} --cuda-memory-usage true python test_append_attn_std.py
SEQ_LEN=65536
/opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o all3_${SEQ_LEN} --cuda-memory-usage true python test_dsk_v3.py --batch_size 2 --num_heads ${HEAD_DIM} --head_dim 128 --seq_len ${SEQ_LEN}
# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile -o append_attn_${SEQ_LEN} --cuda-memory-usage true python test_append_attn_std.py