

无论是深度学习里的计算损失函数总和（Loss Sum）、求最大概率值（Softmax Max），还是大数据统计，归约都是无处不在的核心底层算子。

# 基础概念

在写代码前，我们必须先在脑海中建立两个核心概念：

1. **什么是“归约（Reduction）”？**

   归约的本质，就是**把一个包含很多元素的数组（比如 100 万个数字），通过某种结合律运算（如加法、求最大值、求最小值），“收缩”或“折叠”成一个单一的最终数值**。

   - *例如：* 数组 `[1, 2, 3, 4]` 进行求和归约，结果就是 `10`。

2. **串行思维的死胡同 vs 并行的冲突挑战**

   - **串行怎么做：** 在 CPU 上，我们通常用一个简单的 `for` 循环，从头累加到尾，时间复杂度是 $O(N)$。
   - **并行为什么难：** 如果让 100 万个线程同时去把自己的数字往同一个总结果变量里加，就会发生“数据竞争（Race Condition）”——大家同时去修改同一个内存地址，后面的数据会直接把前面的覆盖掉，导致结果完全错误。

# 公式与归约树（Tree Reduction）推导

为了解决并行冲突并实现极速归约，我们必须采用“树形归约（Tree-based Reduction）”思想。也就是像打淘汰赛一样，两两配对相加，直到决出总冠军。

假设我们有一个存放在共享内存中的数组 $s\_data$，长度为 $N$。我们通过步长（Stride）不断折半来实现并行归约。

## 核心步长与索引映射公式

$$s = 2^r \quad (r = 0, 1, 2, \dots)$$

$$Index_{target} = tid$$

$$Index_{source} = tid + s$$

## 符号全翻译

- $s$：当前轮次的步长（Stride），控制哪些线程需要跨越一段距离去抓取数据相加。
- $r$：归约迭代的轮次编号（从 0 开始递增）。
- $2^r$：以 2 为底数的幂运算，代表步长随着轮次呈几何级数倍增。
- $Index_{target}$：当前正在干活、负责接收累加结果的目标线程在共享内存中的槽位索引。
- $Index_{source}$：远端被读取并累加进来的数据来源索引。
- $tid$：当前线程在块内部的局部工号（对应代码中的 `threadIdx.x`）。
- $=$：数学上的赋值与定义符号。
- $+$：标量加法运算符号。

## 推导过程

为了让每个线程在每一轮安全地进行两两合并，我们需要分步推进：

$$s = 1$$

(初始化第一轮步长为 1)

(即相邻的两个元素进行配对相加)



$$Index_{target} = tid$$

(当前线程工号作为目标写入位置)



$$Index_{source} = tid + s$$

(目标位置向右偏移一个步长 s)

($\Rightarrow$ 确定了远端需要贡献数据的队友工号)



$$s_{data}[Index_{target}] = s_{data}[Index_{target}] + s_{data}[Index_{source}]$$

(目标线程将自己槽位的值)

(与远端队友槽位的值相加)

($\Rightarrow$ 将结果原地更新保存到目标槽位中)



随着外层循环让步长 $s$ 翻倍（$s = 1 \to 2 \to 4 \to 8 \dots$），存活参与计算的线程数减半，最终在根节点（`tid == 0`）汇聚出整个 Block 的总和。

![image-20260906104411508](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/20260906104411914.png)

# 高效归约算子实例程序 (`reduction_demo.cu`)

下面是一个基于 **Shared Memory 的并行求和归约（Reduction Sum）** 完整可运行程序。请在 WSL 中新建 `reduction_demo.cu` 并写入：

```c++
#include <iostream>
#include <cuda_runtime.h>

// ==========================================
// 模块一：并行归约求和算子 (Reduction Kernel)
// ==========================================
__global__ void reduceSumKernel(const float *d_in, float *d_out, int numElements) {
    // 1. 声明共享内存工作台，大小为 256
    __shared__ float s_data[256];

    int tid = threadIdx.x; // 线程局部工号 (0 到 255)
    int idx = blockIdx.x * blockDim.x + tid; // 全局绝对索引

    // 2. 协作搬运：把全局内存的数据初次加载到共享内存中
    if (idx < numElements) {
        s_data[tid] = d_in[idx];
    } else {
        s_data[tid] = 0.0f; // 越界部分补零，不影响求和结果
    }

    // 极其重要的同步屏障：确保所有数据都已到位
    __syncthreads();

    // 3. 树形归约核心逻辑 (Tree Reduction)
    // 步长 s 从 1 开始，每次循环翻倍，直到覆盖整个 Block
    for (int s = 1; s < blockDim.x; s *= 2) {
        // 只有当 tid 是当前步长的整数倍时，该线程才参与这一轮的相加
        // 这样可以避免重复计算和数据覆盖
        if (tid % (2 * s) == 0) {
            s_data[tid] += s_data[tid + s];
        }
        
        // 每一轮相加结束后，必须同步等待队友完成
        __syncthreads();
    }

    // 4. 汇总结果：每个 Block 的第 0 号线程，将最终算出的局部总和写回全局内存
    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

// ==========================================
// 模块二：CPU 主控程序 (Host Program)
// ==========================================
int main() {
    int numElements = 1000000; // 100 万个浮点数
    size_t size = numElements * sizeof(float);

    std::cout << "正在初始化 100 万个归约测试数据（全部设为 1.0）..." << std::endl;

    float *h_in = (float *)malloc(size);
    for (int i = 0; i < numElements; ++i) {
        h_in[i] = 1.0f; // 100万个 1.0 相加，最终期望总和应为 1000000.0
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void **)&d_in, size);
    // 注意：输出数组的大小等于 Block 的数量（每个 Block 产出一个局部和）
    cudaMalloc((void **)&d_out, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    std::cout << "正在启动并行归约算子..." << std::endl;
    
    // 启动归约核函数
    reduceSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, numElements);

    // 将各个 Block 的局部结果搬回 CPU
    float *h_out = (float * )malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_out, d_out, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // 在 CPU 端把各个 Block 的局部和做最后的简单累加
    float finalSum = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        finalSum += h_out[i];
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "归约计算最终结果: " << finalSum << std::endl;
    std::cout << "期望标准答案:     1.0000e+06" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

​                                   

# 例子 —— 篮球赛淘汰赛

为了绝对不忘，请把树形归约（Tree Reduction）想象成一场**1024 支队伍的篮球淘汰赛**：

- **初赛（协同搬运）：** 1024 名队员（线程）把各自的成绩登记在车间黑板（`s_data`）上。
- **第一轮淘汰赛（`s = 1`）：** 编号为偶数的队员和相邻的奇数队员（`tid + 1`）比拼并相加，留下 512 个人。
- **第二轮淘汰赛（`s = 2`）：** 留下的人再次两两合并，剩 256 个人。
- ……
- **决赛：** 经过 $\log_2(N)$ 轮激烈的淘汰，最终只剩下 1 个人（`tid == 0`）手里拿着全场的总冠军奖杯（`s_data[0]`），向全世界（全局内存）宣告总成绩！



# Reduction 性能优化

前面的基础版本已经利用 **Shared Memory + Tree Reduction**，将一个 Block 内的数据逐轮折半。

但“算法是并行的”并不代表“GPU 执行效率已经很高”。

基础版本的核心代码是：

```cpp
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
        s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
}
```

它能够得到正确结果，但存在几个明显的性能问题：

1. Warp 内线程执行路径不一致，产生 **Warp Divergence**
2. `%` 和索引计算存在额外指令开销
3. 后期只有少量线程工作，却仍然执行 Block 级 `__syncthreads()`
4. 数据先进入 Shared Memory，再不断读写 Shared Memory
5. 每个 Block 只处理少量元素，Kernel 启动和数据搬运开销占比仍然较高

因此，Reduction 是 CUDA 中最经典的优化案例之一。

Mark Harris 曾用一系列 Reduction Kernel 展示如何逐步逼近 GPU 的内存带宽极限。

其核心思想可以概括为：

> **减少线程分歧 → 改善访存模式 → 减少同步 → 减少 Shared Memory → 增加单线程工作量**

------

# 1. Warp Divergence

GPU 并不是让一个 Warp 中的 32 个线程完全独立地执行指令。

一个 Warp 内的线程共享指令执行过程。

假设：

```cpp
if (condition) {
    A();
} else {
    B();
}
```

如果 Warp 中：

```text
Thread  0 ~ 15 → condition = true
Thread 16 ~ 31 → condition = false
```

GPU 不能简单地让两组线程完全独立执行两条路径。

逻辑上可以理解为：

```text
            Warp
             │
       condition ?
        /        \
     true        false
      │            │
  执行 A()      执行 B()
  部分线程       另一部分线程
    active          active
        \          /
          重新汇合
```

因此，一个 Warp 内线程走不同控制流路径的现象称为：

**Warp Divergence（Warp 分歧）**

> 注意：现代 NVIDIA GPU 的具体线程调度机制已经比早期 SIMT 模型复杂，但“尽量避免同一 Warp 内不必要的控制流分歧”仍然是 CUDA 优化的重要原则。

------

# 版本0：交错寻址

基础 Reduction 使用：

```cpp
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
        s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
}
```

第一轮：

```text
tid:

 0  1  2  3  4  5  6  7 ...
 │     │     │     │
 ▼     ▼     ▼     ▼
工作   工作   工作   工作
```

对应：

```text
0 + 1
2 + 3
4 + 5
6 + 7
...
```

第二轮：

```text
0 + 2
4 + 6
8 + 10
...
```

第三轮：

```text
0 + 4
8 + 12
...
```

活跃线程呈现：

```text
第1轮：1010101010101010...
第2轮：1000100010001000...
第3轮：1000000010000000...
```

同一个 Warp 中，大量线程被交错关闭。

因此 Warp 内会出现明显的执行效率损失。

除此之外：

```cpp
tid % (2 * s)
```

还需要额外的整数运算。

所以第一个优化目标就是：

> **不要通过取模寻找工作线程，而让需要工作的线程连续排列。**



# 版本1：连续线程

将归约方向反过来。

不再：

```text
s = 1 → 2 → 4 → 8
```

而改成：

```text
s = 128 → 64 → 32 → 16 → ...
```

代码：

```cpp
// s >>= 1:把 s 的二进制整体右移 1 位。对于这里的正整数 s，效果等价于：s = s / 2;
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        s_data[tid] += s_data[tid + s];
    }

    __syncthreads();
}
```

例如一个 Block 有 256 个线程。

第一轮：

```text
Thread 0   += Thread 128
Thread 1   += Thread 129
Thread 2   += Thread 130
...
Thread 127 += Thread 255
```

于是活跃线程变成连续排列：

```text
第1轮：

[0 ................ 127] [128 ............... 255]
        active                 inactive


第2轮：

[0 ....... 63] [64 ........................ 255]
    active               inactive


第3轮：

[0 ... 31] [32 ............................ 255]
  active                 inactive
```

这样做有两个直接收益：

- 不再需要 `%`
- 活跃线程集中在连续 Warp 中

例如第一轮：

```text
Warp 0：32 个线程全部工作
Warp 1：32 个线程全部工作
Warp 2：32 个线程全部工作
Warp 3：32 个线程全部工作
Warp 4~7：全部不工作
```

相比交错寻址：

```text
Warp 0：

T0 工作
T1 空闲
T2 工作
T3 空闲
...
```

连续线程版本明显更符合 GPU 的 Warp 执行方式。

完整 Kernel：

```cpp
__global__ void reduceSequential(
    const float *d_in,
    float *d_out,
    int n
) {
    __shared__ float s_data[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx =
        blockIdx.x * blockDim.x + tid;

    s_data[tid] =
        (idx < n) ? d_in[idx] : 0.0f;

    __syncthreads();

    for (
        unsigned int s = blockDim.x / 2;
        s > 0;
        s >>= 1
    ) {
        if (tid < s) {
            s_data[tid] +=
                s_data[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}
```

------



### 工作流程

下面从 GPU 启动 Kernel 开始完整走一遍。

假设：

```
n = 1024

threadsPerBlock = 256
```

那么 Grid 有：

```
1024 / 256 = 4 Blocks
```

整个 GPU 接收到：

```
Block 0 → 处理 input[0   ~ 255]
Block 1 → 处理 input[256 ~ 511]
Block 2 → 处理 input[512 ~ 767]
Block 3 → 处理 input[768 ~ 1023]
            
                
            
            运行
        
```

可以画成：

```
Global Memory

d_in
│
├── [0 .............. 255]
│          ↓
│       Block 0
│
├── [256 ............ 511]
│          ↓
│       Block 1
│
├── [512 ............ 767]
│          ↓
│       Block 2
│
└── [768 ........... 1023]
           ↓
        Block 3
```

------

#### 第一步：分配输入

Kernel 一启动，每个线程先执行：

```
unsigned int tid = threadIdx.x;

unsigned int idx =
    blockIdx.x * blockDim.x + tid;
```

例如看 Block 1：

```
blockIdx.x = 1
blockDim.x = 256
```

其中：

```
Thread 0：

tid = 0

idx
───[全局索引公式]──→
1 × 256 + 0
───[计算]──→
256
```

于是这个线程负责：

```
d_in[256]
```

Thread 37：

```
tid = 37

idx
───[全局索引公式]──→
1 × 256 + 37
───[计算]──→
293
```

负责：

```
d_in[293]
```

Thread 255：

```
idx
───[全局索引公式]──→
1 × 256 + 255
───[计算]──→
511
```

负责：

```
d_in[511]
```

因此：

```
一个线程
   ↓
一个 idx
   ↓
负责一个输入元素
```

------

#### 第二步：搬到共享内存

接下来：

```
s_data[tid] =
    (idx < n) ? d_in[idx] : 0.0f;
```

这一步非常关键。

这里同时用了：

```
idx → 找 Global Memory

tid → 找 Shared Memory
```

例如 Block 1：

```
Thread 0

d_in[256]
    │
    ▼
s_data[0]


Thread 1

d_in[257]
    │
    ▼
s_data[1]


Thread 2

d_in[258]
    │
    ▼
s_data[2]
```

所以对于Block 1中的线程，整个过程是：

```
Global Memory
d_in[256 ... 511]

       ↓ 256 个线程并行搬运

Shared Memory
s_data[0 ... 255]
```



#  一线程两元素**First Add During Load** 

前面的 Kernel 中，一个线程首先只读取一个元素：

```cpp
s_data[tid] = d_in[idx];
```

假设：

```text
Block = 256 threads
```

那么一个 Block 一次只处理：

```text
256 elements
```

对于 256 个线程的 Block，传统实现每个线程只加载 1 个元素，因此一个 Block 每次只处理 256 个输入。实际上，可以让每个线程在加载阶段读取 2 个元素并立即相加，从而先完成一层 `512 → 256` 的局部归约。这样同样的 256 个线程可以一次处理 512 个输入元素，然后再进入原来的 `256 → 128 → ... → 1` Block 内归约过程。

既然如此，可以直接在读取 Global Memory 时完成第一次加法：

```cpp
s_data[tid] =
    d_in[idx] +
    d_in[idx + blockDim.x];
```

只记住 3 件事。

1. `d_in` 在 **Global Memory**，保存原始输入数据。
2. `s_data` 在 **Shared Memory**，保存当前 Block 后续归约要使用的部分和。
3. `d_in[idx] +  d_in[idx + blockDim.x];`在==还没进入shared memory前==就执行了一次，然后再赋值给`s_data[tid]`，所以这行代码完成了**First Add During Load** 

一个 Block 直接处理 512 个输入：

```
Global Memory

512
 │
 │ 加载 + 第一次加法
 ▼
Registers / Shared Memory

压缩成256个再输入到Shared memory
 │
 ▼
128
 │
 ▼
64
 │
 ▼
...
 ▼
1
```

原来 256 个人各拿 1 个包裹，进仓库后才开始两两合并。优化后还是 256 个人，但每人先从外面拿 2 个包裹，当场合成 1 个再进仓库。因此一次能处理 512 个包裹，而仓库内部仍然只需要处理 256 份。



代码：

```cpp
unsigned int idx =
    blockIdx.x * blockDim.x * 2
    + threadIdx.x;

float sum = 0.0f;

if (idx < n) {
    sum += d_in[idx];
}

if (idx + blockDim.x < n) {
    sum += d_in[idx + blockDim.x];
}

s_data[tid] = sum;
```

这里 `idx` 为什么多了一步将前面超过的Block数单独 *2

因为现在：

> 一个 Block 不再处理 `blockDim.x` 个输入，而是处理 `2 × blockDim.x` 个输入。

例如：

```
blockDim.x = 256
```

那么：

```
Block 0 → d_in[0    ~ 511]
Block 1 → d_in[512  ~ 1023]
Block 2 → d_in[1024 ~ 1535]
```

如果还使用原来的：

```cpp
blockIdx.x * blockDim.x + tid
```

相邻 Block 之间的数据范围就会发生重叠。

```
               0       256       512       768
               │        │         │         │
Block 0        ├────────┼─────────┤
               │ 读取   │  读取   │
               │0~255   │256~511  │
               └────────┴─────────┘

Block 1                 ├─────────┼─────────┤
                        │ 读取    │ 读取    │
                        │256~511  │512~767  │
                        └─────────┴─────────┘
                           ↑
                           │
                      重叠区域
```

因此 Grid 中包含的 Block 个数也变成：

```cpp
int blocks =
    (n + threads * 2 - 1)
    / (threads * 2);
```

好处是：

> **同样数量的数据，只需要大约一半数量的 Block。**

这也是 Reduction 优化中非常重要的思想：

**在加载数据的同时完成计算。**





# Warp Unrolling

在优化了**First Add During Load** 后继续观察归约后半段。

假设：

```text
Block = 256 threads
```

归约过程：

```text
256
 ↓
128
 ↓
64
 ↓
32
 ↓
16
 ↓
8
 ↓
4
 ↓
2
 ↓
1
```

当只剩最后一个 Warp 时：

```text
tid = 0 ~ 31
```

所有有效工作已经局限在一个 Warp 内。

因此传统 Reduction 会继续执行：

```cpp
if (tid < 32) ...
__syncthreads();

if (tid < 16) ...
__syncthreads();

if (tid < 8) ...
__syncthreads();
```

但此时已经没有必要继续进行整个 Block 范围的同步。

所以可以把： 

```text
32 → 16 → 8 → 4 → 2 → 1
```

直接展开。

现代 CUDA 中，更推荐使用 Warp primitives 来表达这种 Warp 内通信与同步；如果展示经典 Shared Memory 写法，则需要注意早期教程中依赖隐式 lockstep 的 `volatile` 技巧并不是现代代码的首选。

因此更推荐：

```cpp
if (tid < 32) {
    // 后续使用 warp-level primitive
}
```

然后使用 Warp Shuffle 完成最后一级归约。

------

# Warp Shuffle

从 Kepler 架构开始，CUDA 提供了 Warp Shuffle 指令。

其中 Reduction 最常使用：

```cpp
__shfl_down_sync()
```

它允许：

> **一个线程直接读取同一个 Warp 中另一个线程寄存器里的值。**

> 前面的 Shared Memory Reduction 是“把中间结果存进共享内存，再由其他线程读取”；
>  Warp Shuffle 则是在 **最后一个 Warp 内，绕过 Shared Memory，让线程直接取得其他 Lane 寄存器中的值**。

并不是数据从一开始就在“别的线程寄存器里”。而是优化到 Warp 阶段后，我们**改变了中间结果在线程之间传递的方式**。



**Warp Shuffle 为什么能绕过 Shared Memory** 串起来。先建立 GPU 的物理层级，再看 `lane` 和 `mask`。

## 前置基础

先记住 NVIDIA GPU 中大致的物理层级：

```
GPU
│
├── SM 0
│   ├── CUDA Cores / 执行单元
│   ├── Register File      ← 寄存器文件
│   ├── Shared Memory/L1   ← 片上存储
│   └── Warp Scheduler
│
├── SM 1
│   └── ...
│
├── SM 2
│   └── ...
│
└── L2 Cache
      │
      ▼
   显存 VRAM
   Global Memory
```

最关键的对应关系：

| CUDA 概念     | 典型物理位置                        | 范围             |
| ------------- | ----------------------------------- | ---------------- |
| Register      | SM 内的 Register File               | 每线程私有       |
| Shared Memory | SM 内片上 SRAM                      | Block 内共享     |
| Global Memory | GPU 板载显存/统一内存系统的设备内存 | 整个 Grid 可访问 |

不同 GPU 架构的缓存和 Shared Memory 组织方式会变化，所以不要把上图理解成所有型号完全相同的晶体管布局。

### Shared Memory

属于整个 Block：

```
             Block
               │
      ┌────────┴────────┐
      │  Shared Memory  │
      │                 │
      │ s_data[0]       │
      │ s_data[1]       │
      │ ...             │
      └─────────────────┘
        ↑      ↑      ↑
       T0     T1     T2
```

Block 内不同线程都可以访问。

所以之前我们写：

```
s_data[tid] += s_data[tid + s];
```

Thread 0 可以读取：

```
s_data[16]
```

即使 `s_data[16]` 原本是 Thread 16 负责写进去的。

### Register

寄存器通常是**线程私有**的。

==寄存器在哪里?==

- 这是理解 Shuffle 最关键的地方。

- GPU 的一个 SM 内部有一个很大的：**Register File（寄存器文件）**

概念上：

```
                    SM
┌──────────────────────────────────┐
│                                  │
│       Register File              │
│                                  │
│  Thread 0 的寄存器资源           │
│  Thread 1 的寄存器资源           │
│  Thread 2 的寄存器资源           │
│  ...                             │
│                                  │
│  Shared Memory / L1              │
│                                  │
│  Warp Scheduler                  │
│                                  │
│  Execution Units                 │
│                                  │
└──────────────────────────────────┘
```

因此，当我们说：

> “Thread 0 的寄存器”

不要想象 GPU 里真的有一个独立小芯片写着：

```
Thread 0 Register
```

更准确的理解是：

> **SM 有物理 Register File，硬件/编译器把其中的寄存器资源分配给当前驻留在该 SM 上的各个线程。**

![3138689e-55de-440f-adb1-6bc567ea223d](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/3138689e-55de-440f-adb1-6bc567ea223d.png)

例如：

```
float value = d_in[idx];
```

可以概念性理解成：

```
Thread 0          Thread 1          Thread 2

Register          Register          Register
value = 10        value = 20        value = 30
```

正常 CUDA C++ 代码中，Thread 0 不能写：

```
“给我 Thread 1 的 value”
```

因为 Thread 1 的寄存器不是普通共享地址空间。

而 Warp Shuffle 提供的就是一种特殊的 **Warp 内 Lane 间数据交换机制**。

## 举个例子

用 8 个线程简化。

注意：

> `tid` 是 Block 内编号，`lane` 是 Warp 内编号。

假设：

```
Lane

0   1   2   3   4   5   6   7

Register value

a   b   c   d   e   f   g   h
```

现在：

```c++
value += __shfl_down_sync(
    mask,
    value,
    4
);
```

`offset = 4`。

对于 Lane 0：

```
Lane 0
自己的 value = a

        +

读取 Lane 0+4
        ↓
     Lane 4
     value=e

        ↓

value = a + e
```

Lane 1：

```
自己的 b
   +
Lane 5 的 f
   ↓
 b + f
```

Lane 2：

```
c + g
```

Lane 3：

```
d + h
```

因此：

```
原来：

Lane     0   1   2   3   4   5   6   7
         │   │   │   │   │   │   │   │
value    a   b   c   d   e   f   g   h


__shfl_down_sync(..., value, 4)

         ┌───────────────┐
         │   ┌───────────────┐
         │   │   ┌───────────────┐
         │   │   │   ┌───────────────┐
         ▼   ▼   ▼   ▼
Lane     0   1   2   3   4   5   6   7
         ↑               │
         └──── offset=4 ──┘


结果：

Lane 0 → a + e
Lane 1 → b + f
Lane 2 → c + g
Lane 3 → d + h
```

最重要的是：

**没有共享内存 `s_data[]` 参与这次数据交换。**

基本形式：

```cpp
value = __shfl_down_sync(
    mask,
    value,
    offset
);
```

Reduction 通常写成：

```cpp
for (int offset = warpSize / 2;
     offset > 0;
     offset >>= 1) {

    value += __shfl_down_sync(
        0xffffffff,
        value,
        offset
    );
}
```

`mask` 是一个 **32-bit 位掩码**。

它用来指定：

> **这个 Warp 中哪些 Lane 是这次同步 Shuffle 操作的参与者。**

因为一个 Warp：32 lanes，刚好可以用：32 bits一一对应，0xffffffff代表32个Lane全部参与。

一个 Warp 有 32 个线程，因此：

```text
第一轮offset = 16
第二轮offset =  8
第三轮offset =  4
...offset =  2
...offset =  1
```



# Shuffle 图解

假设 Warp 中只有 8 个值，简化演示：

```text
Thread：

T0  T1  T2  T3  T4  T5  T6  T7

Value：

a   b   c   d   e   f   g   h
```

第一轮：

```text
offset = 4

T0 ← T4
T1 ← T5
T2 ← T6
T3 ← T7
```

得到：

```text
a+e
b+f
c+g
d+h
```

第二轮：

```text
offset = 2

T0 ← T2
T1 ← T3
```

得到：

```text
a+c+e+g
b+d+f+h
```

第三轮：

```text
offset = 1

T0 ← T1
```

最终：

```text
T0 = a+b+c+d+e+f+g+h
```

对于真实的 32-thread Warp：

```text
32
│ offset=16
▼
16
│ offset=8
▼
8
│ offset=4
▼
4
│ offset=2
▼
2
│ offset=1
▼
1
```

只需要 5 轮。



# Warp 归约函数

可以把 Warp 内 Reduction 单独封装：

```cpp
__device__ __forceinline__
float warpReduceSum(float value) {

    //unsigned整数掩码
    unsigned int mask = 0xffffffffu;

    for (int offset = warpSize / 2;
         offset > 0;
         offset >>= 1) {

        value += __shfl_down_sync(
            mask,
            value,
            offset
        );
    }

    return value;
}
```

如果能够保证调用该函数的是完整 Warp，那么：

```cpp
0xffffffffu
```

表示 32 个 Lane 都参与操作。

如果 Warp 可能只有部分 Lane 有效，则不能机械地使用完整 Mask，而应该根据实际参与线程生成 Mask，例如使用：

```cpp
__ballot_sync()
```

或在适合的控制流位置使用：

```cpp
__activemask()
```

因此：

> **Shuffle 的 Mask 表示哪些 Lane 参与这次 Warp-level collective operation。**

它不是一个可以随意忽略的参数。



# Block 级 Shuffle

一个 Block 往往不只有一个 Warp。

例如：

```text
256 threads
    ↓
8 Warps
```

因此可以采用两级 Reduction：

```text
Warp 0 ──→ partial 0 ┐
Warp 1 ──→ partial 1 │
Warp 2 ──→ partial 2 │
Warp 3 ──→ partial 3 ├─→ Warp 0 再归约
Warp 4 ──→ partial 4 │
Warp 5 ──→ partial 5 │
Warp 6 ──→ partial 6 │
Warp 7 ──→ partial 7 ┘
```

![d4aef259-01ca-4959-bce5-9acd56133630](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/d4aef259-01ca-4959-bce5-9acd56133630.png)

第一层：

**每个 Warp 使用 Shuffle 得到自己的局部和。**

第二层：

**每个 Warp 的 Lane 0 把结果写入 Shared Memory。**

第三层：

**Warp 0 再把这些 Warp 局部和归约成 Block 总和。**

代码：

```cpp
__device__ __forceinline__
float warpReduceSum(float value) {

    for (int offset = 16;
         offset > 0;
         offset >>= 1) {

        value += __shfl_down_sync(
            0xffffffffu,
            value,
            offset
        );
    }

    return value;
}


__global__ void reduceShuffle(
    const float *input,
    float *output,
    int n
) {
    __shared__ float warpSums[32];

    int tid = threadIdx.x;
// 使用First Add During Load
    int idx =
        blockIdx.x * blockDim.x * 2
        + tid;

    float value = 0.0f;

    if (idx < n) {
        value += input[idx];
    }

    if (idx + blockDim.x < n) {
        value += input[idx + blockDim.x];
    }

    // 第一级：Warp 内归约
    value = warpReduceSum(value);

    int lane = tid & 31;
    int warpId = tid >> 5;

    // 每个 Warp 只写一个结果
    if (lane == 0) {
        warpSums[warpId] = value;
    }

    __syncthreads();

    // Block 中 Warp 的数量
    int numWarps =
        (blockDim.x + 31) / 32;

    // Warp 0 读取所有 Warp 的结果
    if (warpId == 0) {

        value =
            (lane < numWarps)
            ? warpSums[lane]
            : 0.0f;

        value = warpReduceSum(value);

        if (lane == 0) {
            output[blockIdx.x] = value;
        }
    }
}
```

此时 Shared Memory 不再需要：

```text
256 floats
```

只需要保存：

```text
每个 Warp 一个 float
```

对于：

```text
256 threads = 8 Warps
```

只需要：

```text
8 floats
```



#  优化路线

经典 Reduction 优化通常可以理解成下面的演进过程：

```text
Interleaved Addressing
        │
        │ 消除取模和分散线程
        ▼
Sequential Addressing
        │
        │ 加载阶段直接做第一次加法
        ▼
First Add During Load
        │
        │ 最后一个 Warp 特殊处理
        ▼
Warp Unrolling
        │
        │ 编译期展开
        ▼
Complete Unrolling
        │
        │ 每线程处理更多元素
        ▼
Multiple Elements / Thread
```

Mark Harris 经典 Reduction 示例常按 Kernel 版本划分为：

```text
reduce0
  ↓
reduce1
  ↓
reduce2
  ↓
reduce3
  ↓
reduce4
  ↓
reduce5
  ↓
reduce6
```

需要特别区分：

> `__shfl_down_sync()` 并不是原始“经典 7 个 Reduction Kernel”中的一步。

经典教程形成时主要围绕 Shared Memory Reduction 展开。

Warp Shuffle 是后来 GPU 架构和 CUDA 提供的更现代的 Warp-level 优化手段。



#  atomicAdd 全局归约

前面的 Kernel 最终得到：

```text
Block 0 → partialSum[0]
Block 1 → partialSum[1]
Block 2 → partialSum[2]
...
Block N → partialSum[N]
```

常规程序做法是：

```text
GPU
 │
 │ cudaMemcpy
 ▼
CPU
 │
 │ for
 ▼
finalSum
```

另一种方案是：

**每个 Block 算出局部和后，直接使用 `atomicAdd()` 累加到同一个 Global Memory 变量。**

![ba1c4fdb-acc9-4120-86cf-5abe7ed5b313](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/ba1c4fdb-acc9-4120-86cf-5abe7ed5b313.png)

代码：

```cpp
__global__ void reduceAtomic(
    const float *input,
    float *result,
    int n
) {
    __shared__ float warpSums[32];

    int tid = threadIdx.x;

    int idx =
        blockIdx.x * blockDim.x * 2
        + tid;

    float value = 0.0f;

    if (idx < n) {
        value += input[idx];
    }

    if (idx + blockDim.x < n) {
        value += input[idx + blockDim.x];
    }

    value = warpReduceSum(value);

    int lane = tid & 31;
    int warpId = tid >> 5;

    if (lane == 0) {
        warpSums[warpId] = value;
    }

    __syncthreads();

    int numWarps =
        (blockDim.x + 31) / 32;

    if (warpId == 0) {

        value =
            (lane < numWarps)
            ? warpSums[lane]
            : 0.0f;

        value = warpReduceSum(value);

        if (lane == 0) {
            atomicAdd(result, value);
        }
    }
}
```



#  Atomic 优缺点

`atomicAdd()` 最大的优点是：

**实现非常简单。**

原本：

```text
GPU Block Reduction
        ↓
partial sums
        ↓
CPU Reduction
```

现在：

```text
GPU Block Reduction
        ↓
atomicAdd
        ↓
final result
```

但问题也非常明显。

所有 Block 最终都竞争：

```cpp
result
```

这一个地址。

因此：

```text
Block 0 ─┐
Block 1 ─┤
Block 2 ─┤
Block 3 ─┼──→ 同一个地址
Block 4 ─┤
Block 5 ─┤
...      │
```

Atomic 操作必须保证：

```text
Read
Modify
Write
```

整体具有原子性。

竞争越严重，吞吐越可能受到限制。

所以一般不要让：

```text
每个线程
    ↓
atomicAdd(globalResult)
```

更合理的是：

```text
一个 Block 内
先完成 Reduction
    ↓
只让 tid == 0
    ↓
atomicAdd(globalResult)
```

这样原子操作数量从：

```text
O(N)
```

降低到近似：

```text
O(number of Blocks)
```



#  多 Kernel 级联

对于非常大的数组，另一种更通用的方法是：

**GPU 自己不断对中间结果再次启动 Reduction Kernel。**

假设：

```text
N = 1,000,000,000
```

不需要把第一次产生的大量 Block 局部和搬回 CPU。

可以：

```text
1,000,000,000 elements
          │
          │ Kernel 1
          ▼
   1,953,125 partial sums
          │
          │ Kernel 2
          ▼
       3,815 partial sums
          │
          │ Kernel 3
          ▼
          8 partial sums
          │
          │ Kernel 4
          ▼
          1 result
```

假设：

```text
Block = 256 threads
```

并且每个线程一次处理两个元素，那么一个 Block 可以处理：

```text
256 × 2 = 512 elements
```

因此每轮输出数量近似：

```text
Nnext = ceil(Ncurrent / 512)
```

直到：

```text
Ncurrent = 1
```

#  级联拓扑

整个数据流：

```text
Global Memory

Input[N]
   │
   │ Reduction Kernel #1
   ▼
Buffer A[N/512]
   │
   │ Reduction Kernel #2
   ▼
Buffer B[N/512²]
   │
   │ Reduction Kernel #3
   ▼
Buffer A[N/512³]
   │
   │ ...
   ▼
Result[1]
```

可以准备两个临时 Buffer：

```text
Buffer A
Buffer B
```

然后 Ping-Pong：

```text
A → B
B → A
A → B
B → A
```

避免每一轮重新申请显存。

#  级联 Host 控制

Host 端可以写成：

```cpp
int currentN = n;

const float *currentInput = d_input;

float *bufferA;
float *bufferB;

cudaMalloc(
    &bufferA,
    blocksFirst * sizeof(float)
);

cudaMalloc(
    &bufferB,
    blocksFirst * sizeof(float)
);

float *currentOutput = bufferA;

while (currentN > 1) {

    int blocks =
        (currentN + threads * 2 - 1)
        / (threads * 2);

    reduceShuffle<<<blocks, threads>>>(
        currentInput,
        currentOutput,
        currentN
    );

    currentN = blocks;

    if (currentN == 1) {
        break;
    }

    if (currentOutput == bufferA) {
        currentInput = bufferA;
        currentOutput = bufferB;
    } else {
        currentInput = bufferB;
        currentOutput = bufferA;
    }
}
```

最终：

```cpp
float result;

cudaMemcpy(
    &result,
    currentOutput,
    sizeof(float),
    cudaMemcpyDeviceToHost
);
```

需要注意：

> Kernel launch 之间存在天然的全局执行边界：后一个 Kernel 在同一 CUDA stream 中会等待前一个 Kernel 完成。

因此：

```text
Kernel 1
   ↓
全局结果已经写完
   ↓
Kernel 2
   ↓
全局结果已经写完
   ↓
Kernel 3
```

这解决了单个 Kernel 内无法使用普通 `__syncthreads()` 对整个 Grid 进行同步的问题。

# 三种全局方案

当一个 Block 已经得到局部结果以后，最终全局归约主要有三种思路。

| 方案           | 数据流          | 优点                     | 缺点                      |
| -------------- | --------------- | ------------------------ | ------------------------- |
| CPU 二次归约   | GPU → CPU       | 最容易理解               | 需要传输大量 partial sums |
| `atomicAdd`    | GPU → Atomic    | 代码简单，只复制一个结果 | Block 很多时存在原子竞争  |
| 多 Kernel 级联 | GPU → GPU → GPU | 扩展性好，适合大规模数据 | 多次 Kernel Launch        |

可以粗略理解为：

```text
教学 / 小规模
     ↓
CPU 二次归约

代码简洁 / Block 数适中
     ↓
atomicAdd

大规模高性能通用实现
     ↓
多 Kernel 级联
```

实际工程中还需要根据 GPU 架构、数据规模、数据类型以及是否允许非确定性浮点累加顺序进行 Benchmark，而不能只根据数据量机械选择方案。

#  完整优化图

Reduction 的整个优化过程可以总结成：

```text
┌──────────────────────────────┐
│  Version 0：Interleaved      │
│  tid % (2*s) == 0            │
└──────────────┬───────────────┘
               │
               │ 减少 Warp Divergence
               ▼
┌──────────────────────────────┐
│  Version 1：Sequential       │
│  tid < s                     │
└──────────────┬───────────────┘
               │
               │ Load 时提前相加
               ▼
┌──────────────────────────────┐
│  Version 2：2 Elements       │
│  每线程加载两个元素          │
└──────────────┬───────────────┘
               │
               │ 最后一个 Warp 特殊处理
               ▼
┌──────────────────────────────┐
│  Version 3：Warp Unrolling   │
│  减少 Block 级同步           │
└──────────────┬───────────────┘
               │
               │ Warp 寄存器通信
               ▼
┌──────────────────────────────┐
│  Version 4：Warp Shuffle     │
│  __shfl_down_sync            │
└──────────────┬───────────────┘
               │
               ▼
        每个 Block 一个结果
               │
       ┌───────┼────────┐
       │       │        │
       ▼       ▼        ▼
     CPU    atomicAdd   Multi-Kernel
      │        │          │
      └────────┴──────────┘
               │
               ▼
          Final Result
```

------

# 性能本质

Reduction 做的数学运算非常少。

对于求和：

```text
读取一个 float
        ↓
执行一次 add
```

计算量很低，而需要读取大量数据。

因此 Reduction 通常属于：

**Memory-Bound（内存带宽受限）算子。**

优化 Reduction 的最终目标并不是单纯减少加法次数，因为：

```text
N 个元素求和
```

无论如何都需要大约：

```text
N - 1
```

次加法。

真正需要减少的是：

```text
无效线程执行
+
控制流分歧
+
Shared Memory 访问
+
同步次数
+
Kernel 启动次数
+
Global Memory 数据搬运
```

因此 Reduction 是理解 GPU 性能优化非常好的案例：

> **数学算法没有改变，改变的是数据如何移动、线程如何协作，以及同步发生在哪里。**

# 核心结论

Reduction 优化可以浓缩成四层：

```text
第一层：线程
连续线程工作
减少 Warp Divergence

        ↓

第二层：Warp
__shfl_down_sync
寄存器直接通信

        ↓

第三层：Block
Warp Reduction
+
少量 Shared Memory

        ↓

第四层：Grid
CPU / atomicAdd / Multi-Kernel
完成最终全局归约
```

最终形成 GPU Reduction 的典型层级：

```text
Thread
  ↓
Warp
  ↓
Block
  ↓
Grid
  ↓
Result
```

其中最值得记住的优化思想不是某一段具体代码，而是：

> **先在最小的同步范围内完成局部归约，再逐层向上合并。能在寄存器解决，就不要进入 Shared Memory；能在 Warp 内解决，就不要扩大到整个 Block；能先产生局部结果，就不要让所有线程竞争同一个全局地址。**





# 总结程序

```cpp
/***********************************************************************
 *
 * CUDA Reduction 完整示例
 *
 * 技术点：
 *
 * 1. First Add During Load
 * 2. Warp Shuffle Reduction
 * 3. Warp-level Register Communication
 * 4. Shared Memory Warp Aggregation
 * 5. atomicAdd Global Reduction
 *
 * 编译：
 *
 * nvcc reduction_atomic.cu -o reduction
 *
 * 运行：
 *
 * ./reduction
 *
 ***********************************************************************/


#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>


/***********************************************************************
 *
 * CUDA 错误检查宏
 *
 * CUDA API 不会自动抛异常。
 *
 * 例如：
 *
 * cudaMalloc(...)
 *
 * 如果失败，程序可能继续运行。
 *
 * 所以每一步 CUDA 调用后都检查。
 *
 ***********************************************************************/

#define CUDA_CHECK(call)                                  \
do                                                        \
{                                                         \
    cudaError_t err = call;                                \
                                                          \
    if(err != cudaSuccess)                                 \
    {                                                     \
        std::cerr                                         \
            << "CUDA Error: "                             \
            << cudaGetErrorString(err)                    \
            << std::endl;                                 \
                                                          \
        exit(EXIT_FAILURE);                               \
    }                                                     \
                                                          \
}while(0)



/***********************************************************************
 *
 * Warp Reduction
 *
 * 一个 Warp = 32 个线程
 *
 * value:
 *
 * 当前线程自己的局部数据
 *
 * 通常位于 Register File
 *
 *
 * __shfl_down_sync:
 *
 * 让当前线程读取同 Warp 中
 * 另一个 Lane 的寄存器值。
 *
 ***********************************************************************/

__device__ __forceinline__
float warpReduceSum(float value)
{

    /*
        offset:

        16
        8
        4
        2
        1


        32 个线程：

        32
         |
         v
        16
         |
         v
         8
         |
         v
         4
         |
         v
         2
         |
         v
         1


        最终 Lane 0 保存整个 Warp 的和。

    */


    for(
        int offset = 16;
        offset > 0;
        offset >>= 1
    )
    {

        /*
            0xffffffffu

            二进制：

            11111111111111111111111111111111


            表示：

            Warp 中 32 个 Lane
            全部参与此次 Shuffle。


            offset:

            当前线程读取：

            lane + offset

            的 value。


            例如 offset=16:

            Lane0 <- Lane16
            Lane1 <- Lane17

            ...

            Lane15 <- Lane31

        */


        value += __shfl_down_sync(
            0xffffffffu,
            value,
            offset
        );

    }


    return value;

}



/***********************************************************************
 *
 * Block Reduction + atomicAdd
 *
 *
 * 一个 Block:
 *
 * blockDim.x 个线程
 *
 *
 * 每个线程：
 *
 * 读取两个 Global Memory 元素
 *
 *      input[idx]
 *
 *      input[idx + blockDim.x]
 *
 *
 * 这一步叫：
 *
 * First Add During Load
 *
 ***********************************************************************/


__global__
void reduceAtomic
(
    const float* input,
    float* result,
    int n
)
{


    /**************************************************************
     *
     * Shared Memory
     *
     * 保存：
     *
     * 每个 Warp 的局部和
     *
     *
     * 假设：
     *
     * Block = 256 threads
     *
     * Warp 数量：
     *
     * 256 / 32 = 8
     *
     *
     * 所以只需要：
     *
     * warpSums[8]
     *
     *
     * 这里开 32 是为了支持更大 Block。
     *
     **************************************************************/

    __shared__
    float warpSums[32];



    /*
        tid:

        当前线程在 Block 内编号


        例如：

        Block:

        Thread0
        Thread1
        ...
        Thread255

    */

    int tid = threadIdx.x;



    /*
        为什么乘 2?


        因为：

        一个线程处理两个输入。


        一个 Block 处理：

        blockDim.x * 2

        个元素。


        所以下一个 Block 必须跳过：

        blockDim.x * 2

        个位置。


        否则 Block 之间会读取重叠数据。

    */


    int idx =
        blockIdx.x * blockDim.x * 2
        + tid;




    float value = 0.0f;



    /**************************************************************
     *
     * First Add During Load
     *
     *
     * 一个线程：

     *
     *      input[idx]
     *
     *          +
     *
     *      input[idx+blockDim.x]
     *
     *
     * 结果：

     *
     * 两个输入
     *
     *      ↓
     *
     * 一个局部和
     *
     **************************************************************/


    if(idx < n)
    {
        value += input[idx];
    }


    if(idx + blockDim.x < n)
    {
        value += input[idx + blockDim.x];
    }



    /**************************************************************
     *
     * 第一级：
     *
     * Warp 内 Reduction
     *
     *
     * 此时：
     *
     * value 在寄存器中。
     *
     *
     * 不需要 Shared Memory。
     *
     **************************************************************/


    value = warpReduceSum(value);



    /*
        lane:

        当前线程在 Warp 内的位置。


        tid:

        Block 内编号。


        例如：

        tid=37


        warpId:

        37 / 32 = 1


        lane:

        37 % 32 = 5

    */


    int lane =
        tid & 31;


    int warpId =
        tid >> 5;




    /*
        每个 Warp 只有 Lane0 写结果。


        例如：

        Warp0:

        Lane0 -> warpSums[0]


        Warp1:

        Lane0 -> warpSums[1]

    */


    if(lane == 0)
    {

        warpSums[warpId] = value;

    }



    /*
        等待所有 Warp 写完 Shared Memory

    */


    __syncthreads();




    /*
        当前 Block 有多少 Warp


        例如：

        blockDim=256


        numWarps:

        (256+31)/32

        =8

    */


    int numWarps =
        (blockDim.x + 31)
        /32;




    /*
        第二级：

        Warp0 负责归约所有 Warp 的结果


        其他 Warp 空闲。

    */


    if(warpId == 0)
    {


        /*
            Lane0:

            读取 warpSums[0]


            Lane1:

            读取 warpSums[1]


            ...


            Lane7:

            读取 warpSums[7]

        */


        value =
            (lane < numWarps)
            ?
            warpSums[lane]
            :
            0.0f;




        /*
            再次使用 Shuffle

            完成：

            Warp 数量级 Reduction

        */


        value =
            warpReduceSum(value);




        /*
            一个 Block 最终只有：

            Lane0

            得到 Block Sum


            使用 atomicAdd:

            多个 Block 同时累加到 result

        */


        if(lane == 0)
        {

            atomicAdd(
                result,
                value
            );

        }

    }

}



/***********************************************************************
 *
 * CPU Reference
 *
 * 用于验证 GPU 结果。
 *
 ***********************************************************************/


float cpuReduce(
    const std::vector<float>& data
)
{

    float sum = 0.0f;


    for(float x:data)
    {
        sum += x;
    }


    return sum;

}





int main()
{


    /*
        输入规模

        这里设置：

        1 << 24

        = 16777216

        个 float

    */


    int n = 1 << 24;



    std::cout
        << "Elements: "
        << n
        << std::endl;




    /**************************************************************
     *
     * CPU 创建数据
     *
     **************************************************************/


    std::vector<float> h_input(n);


    std::mt19937 gen(123);


    std::uniform_real_distribution<float>
        dist(0.0f,1.0f);



    for(auto& x:h_input)
    {
        x = dist(gen);
    }



    float cpuResult =
        cpuReduce(h_input);




    /**************************************************************
     *
     * GPU Memory
     *
     **************************************************************/


    float* d_input=nullptr;

    float* d_result=nullptr;



    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n*sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_result,
            sizeof(float)
        )
    );




    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n*sizeof(float),
            cudaMemcpyHostToDevice
        )
    );



    /*
        atomicAdd:

        多个 Block 最后累加到同一个地址。


        所以必须初始化为 0。

    */


    CUDA_CHECK(
        cudaMemset(
            d_result,
            0,
            sizeof(float)
        )
    );




    /**************************************************************
     *
     * Kernel Launch
     *
     **************************************************************/


    int threads = 256;


    /*
        每个 Block 处理：

        threads * 2

        个元素

    */


    int blocks =
        (n + threads*2 -1)
        /
        (threads*2);



    std::cout
        << "Blocks: "
        << blocks
        << std::endl;



    reduceAtomic
    <<<blocks,threads>>>
    (
        d_input,
        d_result,
        n
    );



    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );




    float gpuResult;



    CUDA_CHECK(
        cudaMemcpy(
            &gpuResult,
            d_result,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );




    std::cout
        << "CPU result: "
        << cpuResult
        << std::endl;


    std::cout
        << "GPU result: "
        << gpuResult
        << std::endl;



    std::cout
        << "Difference: "
        << abs(cpuResult-gpuResult)
        << std::endl;




    CUDA_CHECK(
        cudaFree(d_input)
    );


    CUDA_CHECK(
        cudaFree(d_result)
    );



    return 0;

}
```

