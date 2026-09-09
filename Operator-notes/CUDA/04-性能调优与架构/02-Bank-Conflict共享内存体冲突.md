# 前置基础

1. **共享内存（Shared Memory）**：位于 GPU 芯片内部的高速缓存，供同一个 Block 内的所有线程读写，速度远快于全局内存。

2. **线程束（Warp）**：CUDA 执行的基本单位，32 个线程为一个 Warp，同步执行指令（上一节已讲）。

3. **内存体（Bank）**：为了实现高带宽，共享内存被划分为 32 个相等大小的内存模块（Bank）。Warp 中的 32 个线程可以同时分别访问这 32 个 Bank，实现极速并发。

   ![image-20260906155357977](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/image-20260906155357977.png)

# 公式推导

在 CUDA 中，连续的 32 位（4 字节）内存字（Word）会被交替分配到 32 个 Bank 中。



$$I \stackrel{\text{按4字节切分得出字索引}}{=} \frac{A}{4}$$

$$B_{id} \stackrel{\text{对32取余映射至具体Bank}}{=} I \bmod 32$$

当 Warp 内的两个线程（设为线程 1 和线程 2）访问不同地址，且这些地址落在同一个 Bank 时，会发生冲突：

$$B_{id1} \stackrel{\text{假设两个地址落在同一Bank}}{=} B_{id2}$$

$$(I_1 \bmod 32) \stackrel{\text{代入Bank编号公式}}{=} (I_2 \bmod 32)$$

## 符号说明

- $A$：共享内存中变量字节的物理内存地址。
- $I$：基于 32 位字长（4 字节）划分的内存单元索引。
- $B_{id}$：共享内存体的物理编号（取值范围 0 ~ 31）。
- $\bmod$：取模（求余）算子。

## 直观理解

共享内存就像有 32 个办事窗口的大型银行，一个 Warp 的 32 个客户同时来办业务。如果每个人恰好去了不同的窗口，瞬间就能全部办完（无冲突）。但如果其中 5 个人非要挤在 3 号窗口（Bank Conflict），他们只能串行排队办理，整体耗时就会变成 5 倍。如果是 32 个人同时想看 3 号窗口张贴的同一张通知，柜员会直接大喇叭广播（广播机制），此时无需排队。



# 实例程序

下面的代码展示了如何引发 Bank Conflict，以及如何使用 **Padding（内存填充）** 技巧来消除它：

```c++
#include <stdio.h>

#define SIZE 32
#define ITERS 100000 // 循环 10 万次放大时间差异

// 存在严重冲突的内核
__global__ void testConflict(float *out) {
    // volatile 强迫每次必须真实访问共享内存，禁止寄存器缓存优化
    __shared__ volatile float s_data[SIZE][SIZE];
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERS; i++) {
        // 糟糕：32 个线程的 tid 不同，但读取的都是第一列，引发 32 路冲突
        sum += s_data[tid][0]; 
    }
    if (tid == 0) out[blockIdx.x] = sum;
}

// 做了 Padding 消除冲突的内核
__global__ void testPadding(float *out) {
    __shared__ volatile float s_data_pad[SIZE][SIZE + 1];
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERS; i++) {
        // 优秀：增加了一列 Padding 后，数据刚好被错开分配到 32 个不同的 Bank
        sum += s_data_pad[tid][0]; 
    }
    if (tid == 0) out[blockIdx.x] = sum;
}

int main() {
    float *d_out;
    int blocks = 10000; // 启动一万个 Block 让 RTX 5080 满载，拉开差距
    cudaMalloc((void**)&d_out, blocks * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_conflict = 0, ms_padding = 0;

    // 1. 测试冲突情况
    cudaEventRecord(start);
    testConflict<<<blocks, 32>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_conflict, start, stop);

    // 2. 测试消除冲突情况
    cudaEventRecord(start);
    testPadding<<<blocks, 32>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_padding, start, stop);

    printf("存在 Bank Conflict 耗时: %f ms\n", ms_conflict);
    printf("消除 Bank Conflict 耗时: %f ms\n", ms_padding);
    printf("性能差距: 约 %.1f 倍\n", ms_conflict / (ms_padding > 0 ? ms_padding : 1));

    cudaFree(d_out);
    return 0;
}
```

​                                              

## 部署说明

除了使用 `nvcc` 编译外，验证 Bank Conflict 最权威的工具是 Nsight Compute (`ncu`)。

```bash
nvcc time_conflict.cu -o time_conflict -arch=sm_89 -O3
./time_conflict
```

输出：

```bash
(base) futaba_sakura@FutabaSakura:~/workspace/operator$ nvcc time_conflict.cu -o time_conflict -arch=sm_89 -O3
(base) futaba_sakura@FutabaSakura:~/workspace/operator$ ./time_conflict
存在 Bank Conflict 耗时: 324.409515 ms
消除 Bank Conflict 耗时: 12.091168 ms
性能差距: 约 26.8 倍
```



# 练习测评

请根据刚才学习的公式（$I \bmod 32$）回答以下两个问题：



**题 1**：如果一个 Warp 里的 32 个线程按照 `tid` 访问共享内存 `__shared__ float arr[]`，代码为 `float val = arr[tid];`。请问会发生 Bank Conflict 吗？为什么？

- $$\text{Index} \stackrel{\text{读取数组下标}}{=} T_{id}$$

  $$B_{id} \stackrel{\text{代入映射公式}}{=} T_{id} \bmod 32$$

  当 $T_{id} = 0$ 时：

  $$B_{id}(0) \stackrel{\text{代入变量}}{=} 0 \bmod 32$$

  $$\stackrel{\text{计算取余}}{=} 0$$

  当 $T_{id} = 1$ 时：

  $$B_{id}(1) \stackrel{\text{代入变量}}{=} 1 \bmod 32$$

  $$\stackrel{\text{计算取余}}{=} 1$$

  **结论**：32 个线程的读取请求，刚好完美映射到 0~31 号不同的 Bank 上。由于无一重复，因此**完全没有冲突**（0 路冲突），这是最极致的高效访问。

**题 2**：如果访问模式改为步长为 2：代码为 `float val = arr[tid * 2];`。请问会发生 Bank Conflict 吗？如果会，是几路冲突（即几个人排队挤同一个窗口）？

- $$\text{Index} \stackrel{\text{代入2倍步长}}{=} T_{id} \times 2$$

  $$B_{id} \stackrel{\text{代入映射公式}}{=} (T_{id} \times 2) \bmod 32$$

  当 $T_{id} = 0$ 时：

  $$B_{id}(0) \stackrel{\text{代入变量}}{=} 0 \bmod 32$$

  $$\stackrel{\text{计算取余}}{=} 0$$

  当 $T_{id} = 16$ 时：

  $$B_{id}(16) \stackrel{\text{代入变量}}{=} 32 \bmod 32$$

  $$\stackrel{\text{计算取余}}{=} 0$$

  当 $T_{id} = 1$ 时：

  $$B_{id}(1) \stackrel{\text{代入变量}}{=} 2 \bmod 32$$

  $$\stackrel{\text{计算取余}}{=} 2$$

  当 $T_{id} = 17$ 时：

  $$B_{id}(17) \stackrel{\text{代入变量}}{=} 34 \bmod 32$$

  $$\stackrel{\text{计算取余}}{=} 2$$

  **结论**：线程 0 和线程 16 撞在了 Bank 0；线程 1 和线程 17 撞在了 Bank 2。每 2 个线程挤在一个物理窗口，因此它是 **2路冲突**，性能损失一倍。

如果访问模式改为非常极端的 `float val = arr[tid * 32];`，请问会发生几路冲突？请给出你的分析过程！

$$\text{Index}$$

$$\stackrel{\text{代入32倍步长}}{=}$$

$$T_{id} \times 32$$

$$B_{id}$$

$$\stackrel{\text{代入映射公式}}{=}$$

$$(T_{id} \times 32) \bmod 32$$

$$B_{id}$$

$$\stackrel{\text{任意整数乘32后对32取余必为0}}{=} 0$$

**结论**：不论 $T_{id}$ 是 0 还是 31，经过公式计算后，目标 Bank 编号 $B_{id}$ 永远等于 0。这意味着 Warp 中的全部 32 个线程都同时向 Bank 0 发起读写请求，引发了最极端的 **32路冲突（32-way Bank Conflict）**。这正是我们刚才运行性能测试代码时，耗时暴增 26.8 倍的根本原因。
