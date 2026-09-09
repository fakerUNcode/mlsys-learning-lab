# 前置基础

1. **计算吞吐量（Compute Throughput）**：GPU 的处理核心（CUDA Core / Tensor Core）每秒能够完成的数学运算次数。
2. **内存带宽（Memory Bandwidth）**：GPU 显存通道每秒能够从全局内存（Global Memory）搬运到计算核心的数据体积。

# 公式推导

Nsight Compute（NCU）对内核进行性能诊断的核心基准是 **Roofline 模型（屋顶线模型）**。它通过计算程序的算术强度，来判断程序性能是卡在了计算上，还是卡在了访存上。



$$I \stackrel{\text{计算程序的算术强度}}{=} \frac{W_{ops}}{Q_{bytes}}$$

$$P_{roof} \stackrel{\text{取计算与显存两者的性能短板}}{=} \min(P_{peak}, I \times B_{peak})$$

## 符号说明

- $I$：算术强度（FLOPs/Byte，代表读取每字节数据后进行了多少次运算）。
- $W_{ops}$：内核在运行期间执行的总数学操作数。
- $Q_{bytes}$：内核在运行期间往返全局内存读写的总字节数。
- $P_{roof}$：当前内核受模型约束下的实际可达性能上限。
- $P_{peak}$：当前显卡（如 RTX 5080）的理论峰值计算性能。
- $B_{peak}$：当前显卡（如 RTX 5080）的理论峰值显存带宽。
- $\min$：取最小值算子。

# 程序实例

我们编写一个简单的向量乘加运算（FMA）内核 `roofline.cu`，它的算术强度极低，非常适合用来让 NCU 抓取“访存受限（Memory Bound）”的特征：

```c++
#include <stdio.h>

// 向量乘加运算：D = A * B + C
__global__ void mathKernel(float *a, float *b, float *c, float *d, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // 读取 3 个浮点数 (12 字节)，写入 1 个浮点数 (4 字节)
        // 仅进行了 1 次乘法和 1 次加法 (2 个操作)
        d[tid] = a[tid] * b[tid] + c[tid];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    float *d_a, *d_b, *d_c, *d_d;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_d, size);

    // 启动 1000 个 Block，每个 1024 线程，打满 GPU
    mathKernel<<<(N + 1023) / 1024, 1024>>>(d_a, d_b, d_c, d_d, N);
    cudaDeviceSynchronize();

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    return 0;
}
```

## 部署说明

1. **安全编译**：使用 `-lineinfo` 替代 `-G` 保留行号，并使用兼容架构 `sm_89` 防止编译器彻底清空无用代码：

   Bash

   ```
   nvcc roofline.cu -o roofline -arch=sm_89 -lineinfo
   ```

2. **快速性能采样**：我们不需要全量指标，只需让 NCU 输出基础调度与 Roofline 概览：

   Bash

   ```
   ncu --set basic ./roofline
   ```

   *执行后，终端会打印出类似 `Compute (SM) Throughput` 与 `Memory Bandwidth` 的百分比占比数据。*

## 直观理解

GPU 就像一家加工厂。算术强度就是“每吨原材料需要加工多少次”。如果加工步骤少、进货需求量极大，流水线就会被卡在“货车搬运”环节（访存受限 / Memory Bound）；如果加工极其繁琐（如复杂的加密哈希），流水线就会被卡在“工人加工”环节（计算受限 / Compute Bound）。NCU 就是厂长，负责把当前工厂处于哪种受限状态以百分比报表打印出来。



# 练习测评

请根据刚才讲解的 Roofline 理论公式回答以下两个问题：

**题 1**：在上面的 `mathKernel` 内核中，每次计算需要读取 $a, b, c$ 三个变量，并写回 $d$ 一个变量（都是 4 字节的 float），而数学计算只有 1 次乘法和 1 次加法（共 2 个 Ops）。请严格代入公式 $I = \frac{W_{ops}}{Q_{bytes}}$，计算出这个内核的算术强度 $I$ 是多少？

- ~~I = 2/4 = 0.5~~

- **计算算术强度**

  $$W_{ops} \stackrel{\text{1乘1加}}{=} 2$$

  读取 $a, b, c$ 消耗 $3 \times 4 = 12$ 字节，写入 $d$ 消耗 $1 \times 4 = 4$ 字节。

  $$Q_{bytes} \stackrel{\text{读写总和}}{=} 12 + 4 = 16$$

  $$I \stackrel{\text{代入公式}}{=} \frac{2}{16} = 0.125$$

  **结论**：算术强度极低，每读写 1 字节只做 0.125 次运算。

**题 2**：如果 Nsight Compute 终端打印出：`Memory Bandwidth 89%`，`Compute (SM) Throughput 5%`。请问你的这个内核目前是处于访存受限（Memory Bound）还是计算受限（Compute Bound）？如果你要升级硬件来加速这个程序，应该优先买更大显存带宽的显卡，还是更强算力的显卡？

- ~~I = 5/89 < 1 ，算数强度低，计算受限，应该优秀购买更强力的计算卡。~~

- **判断受限类型**

  NCU 面板给出的 `89%` 和 `5%` 指的是**硬件利用率**。

  - 显存通道利用率高达 `89%`，说明搬运数据的通路已经快被撑爆了，处于“大塞车”状态。
  - 计算核心利用率仅 `5%`，说明算力极度过剩，核心大部分时间都在干等数据送达。

  **结论**：程序处于**访存受限（Memory Bound）**。为了解决这个问题，你应该优先购买**更大显存带宽**的显卡（比如搭载 HBM 高带宽显存的加速卡），买再强的计算核心也只是继续闲置。

- 想象成一个剥瓜子机器（GPU）。 算术强度 $I=0.125$ 意味着“每运来 8 斤带壳瓜子，只能剥出 1 斤瓜子仁”，这是个力气活少但极占体积的活。 此时“运输卡车使用率 89%”（显存带宽满载），而“剥壳机开机率 5%”（计算核心闲置）。显然是卡在运输上了（访存受限），此时你需要买更多的运输卡车（升级显存带宽），而不是去升级更快的剥壳机（计算算力）。
