import matplotlib.pyplot as plt

FA3_time = [0.096,
0.361,
0.989,
3.777,
14.739,
61.641]

FA3_fp8_time = [0.114,
0.364,
0.878,
2.836,
10.386,
41.108]

SA_torch_time = [0.241,
0.522,
1.161,
3.276,
11.914,
44.136]

SA_paddle_time = [0.571,
0.686,
1.210,
3.373,
11.783,
45.558]

append_attn_time = [0.222,
0.658,
2.036,
7.552,
28.412,
115.521]

seq_len = ["1K", "4K", "8K", "16K", "32K", "64K"]

plt.plot(seq_len, FA3_time, label="FA3 (FP16)", marker='o', linestyle='--')
plt.plot(seq_len, FA3_fp8_time, label="FA3 (FP8)", marker='o', linestyle='--')
plt.plot(seq_len, SA_torch_time, label="SA (Torch) (FP16)", marker='o', linestyle='-.')
plt.plot(seq_len, SA_paddle_time, label="SA (Paddle) (FP16)", marker='o', linestyle='-.')
plt.plot(seq_len, append_attn_time, label="Append Attention (FP16)", marker='o')

# 添加标题和标签
plt.title("Kernel Latency (By Nsight) for Different Kernels")
plt.xlabel("Sequence Length")
plt.ylabel("Time (ms)")
plt.grid(True)

# 添加图例
plt.legend()

# 保存图像为文件
plt.savefig("execution_time_comparison.png")