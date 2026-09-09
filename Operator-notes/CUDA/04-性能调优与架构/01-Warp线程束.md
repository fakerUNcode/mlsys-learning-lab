# 前置基础

1. **线程与线程块（Thread & Block）**：CUDA 的核心执行单元是线程（Thread），多个协同工作的线程会被打包成一个线程块（Block）。
2. **SIMT 架构**：单指令多线程（Single Instruction, Multiple Threads）。GPU 的处理核心在同一时刻会向一组线程广播同一条指令，让它们同步执行，但处理的数据不同。

# 核心推导

在 CUDA 中，Block 在硬件层面会被进一步划分为固定大小的调度单位，即 Warp（线程束）。

$$W_{size} \stackrel{\text{NVIDIA硬件标准定义}}{=} 32$$

设一维 Block 中的线程局部索引为 $T_{id}$，其对应的 Warp 编号 $W_{id}$ 计算如下：

$$T_{id} \stackrel{\text{读取CUDA内置变量}}{=} \text{threadIdx.x}$$

$$W_{id} \stackrel{\text{索引整除Warp容量}}{=} \lfloor \frac{T_{id}}{W_{size}} \rfloor$$

$$\text{Warp}_{total} \stackrel{\text{向上取整计算总数}}{=} \lceil \frac{B_{size}}{W_{size}} \rceil$$

## 符号说明

- $W_{size}$：单个 Warp 包含的物理线程数量。
- $T_{id}$：当前线程在 Block 内的局部一维索引。
- $W_{id}$：当前线程所在的 Warp 编号。
- $B_{size}$：当前 Block 包含的总线程数量（`blockDim.x`）。
- $\lfloor x \rfloor$ / $\lceil x \rceil$：向下取整与向上取整算子。

## 直观理解

你可以把 Block 想象成一个大型旅行团，Thread 是游客。导游（GPU 硬件）规定，每 32 名游客必须强制绑成一个小队（Warp）。小队长举旗喊“迈左腿”，这 32 人必须同时迈左腿（SIMT）。如果遇到岔路，一半人想往左，一半人想往右，他们不能拆伙。只能先等左边的人走完退回原地，右边的人再接着走。这种等待被称为**分支发散（Warp Divergence）**，会严重拖慢整个队伍的速度。



## 部署说明

```
nvcc warp_test.cu -o warp_test -arch=all
```

## 实例程序

以下代码展示了如何诱发以及如何避免 Warp Divergence（分支发散）：

```c++
#include <stdio.h>

__global__ void warpTest(float *data) {
    int tid = threadIdx.x;
    
    // 糟糕的写法：导致 Warp Divergence
    // 在同一个 Warp (0~31) 中，偶数线程满足 if，奇数线程走向 else。
    // 这导致 GPU 必须串行执行这两个分支，性能减半。
    if (tid % 2 == 0) {
        data[tid] = 1.0f;
    } else {
        data[tid] = 2.0f;
    }

    // 优秀的写法：避免 Warp Divergence
    // 前 32 个线程 (Warp 0) 全部走 if，后 32 个线程 (Warp 1) 全部走 else。
    // 每个 Warp 内部的 32 个线程行动高度一致，没有分支冲突。
    if (tid / 32 == 0) {
        data[tid] = 3.0f;
    } else {
        data[tid] = 4.0f;
    }
}

int main() {
    float *d_data;
    cudaMalloc((void**)&d_data, 64 * sizeof(float));
    
    // 启动 1 个 Block，包含 64 个线程
    warpTest<<<1, 64>>>(d_data);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    return 0;
}
```

​                                      

## 练习测评

**题 1**：如果我们在启动内核时，将参数设置为 `<<<1, 100>>>`（即 1 个 Block 包含 100 个线程），那么 GPU 硬件在底层实际上会分配几个 Warp 来执行这个 Block？

- 根据公式计算：

  $$W_{total} = \lceil \frac{100}{32} \rceil = 4$$

  **补充细节**：硬件虽然在底层分配了 4 个 Warp，但在第 4 个 Warp（即 Warp 3，负责 tid 96~99）中，只有前 4 个线程是活跃的。剩余的 28 个线程会被硬件掩码（Mask）标记为不活跃状态。它们不产生实际结果，但在指令发射时仍会占用执行周期。

**题 2**：在下面的代码片段中，会不会发生 Warp Divergence？为什么？

```c++
int tid = threadIdx.x;
if (tid > 10) {
    data[tid] = 5.0f;
}
```

- ~~会发生Warp Divergence，因为前10个线程都不运行该代码，而第10个线程后的所有线程都执行该代码。~~

- **分支发散（Warp Divergence）是针对单个 Warp 内部而言的**，并不是以整个 Block 为单位。我们来拆解一下：

  - **对于 Warp 0（tid 0~31）**：线程 0~10（共 11 个）跳过代码，线程 11~31（共 21 个）执行赋值。小队内部出现了分歧，必须串行两路分支，**发生发散**。
  - **对于 Warp 1（tid 32~63）及以后**：内部所有 32 个线程的判断结果全部为真。小队全员动作高度一致，直接并行执行赋值，**不发生发散**。

  因此，最严谨的答案是：**仅 Warp 0 会发生分支发散，其余 Warp 不会。**
