

在 CUDA 的世界里，流多处理器（SM）算力再强，如果显存（Memory）跟不上，整个 GPU 就会变成“空有一身力气却吃不饱饭的巨兽”。我们从最基础、容量最大、也是大家最常用的 **Global Memory（全局内存）** 开始学起。



# 基础概念

在写代码前，我们先在大脑中理清全局内存的物理定位和核心痛点：

1. **什么是 Global Memory（全局内存）？**
   - 它就是我们常说的 **“显存”（比如显卡上的 8GB/16GB GDDR6 或 HBM 显存）**。
   - **特点：** 容量巨大，CPU 和 GPU 都能访问它（前面我们用的 `cudaMalloc` 申请的空间，全部都在全局内存里）。
   - **缺点：** **极其遥远且缓慢**。从 SM（计算核心）去全局内存拿一次数据，通常需要付出 **200 到 400 个时钟周期**的漫长等待。
2. **什么是“内存合并访问（Coalesced Memory Access）”？**
   - 这是全局内存优化中**最核心、最生死攸关**的概念。
   - 想象一下：如果一个线程束（32个线程）同时要去全局内存拿数据：
     - **糟糕的情况（非合并）：** 32个线程去内存里乱七八糟、东一榔头西一棒子抓数据，显存控制器得发 32 次独立的快递，慢得像蜗牛。
     - **完美的情况（合并）：** 32个线程去拿一块**连续**的内存（比如线程 0 拿第 0 个，线程 1 拿第 1 个……），显存控制器只需要用“一个大卡车”在一瞬间把这连续的一大块数据全抓回来，效率直接拉满！



# 核心机制与公式推导

为了让你直观感受到全局内存“合并访问”与“非合并访问”的效率差距，我们通过一个简单的**数组倍增算子**来做对比，并推导其索引映射逻辑。

## 连续索引映射公式

$$I_{global} = B_{id} \times B_{dim} + T_{id}$$

## 符号全翻译（绝对不省略）

- $I_{global}$：当前线程在全局内存一维数组中的绝对访问地址（索引）。
- $B_{id}$：当前线程块的编号，对应代码中的 `blockIdx.x`。
- $B_{dim}$：每个线程块的大小（线程数），对应代码中的 `blockDim.x`。
- $T_{id}$：线程在块内的局部编号，对应代码中的 `threadIdx.x`。

## 全局内存优化实例程序 (`global_memory_demo.cu`)

进阶技巧：利用 `__restrict__` 与 `const` 榨干编译器性能

在 `scaleGlobalMemoryKernel` 算子设计里，可以通过加入特定的 C/C++ 修饰符来赋予 `nvcc` 编译器极大的激进优化空间，这是工程界最常用的低成本提速手段。  

**1. `const` 修饰符：激活只读缓存（Read-Only Cache）**

如果传入 Kernel 的数据在整个计算过程中**绝对只读不写**（例如输入数组 `d_in`），务必给指针加上 `const` 修饰符。现代 GPU 拥有专用的只读数据缓存通道（如 Texture / Constant Cache）。标注 `const` 后，编译器有可能会将这些加载指令路由到独立的只读通道，从而分担常规 L1/L2 缓存的压力。

**2. `__restrict__` 修饰符：打破指针别名（Pointer Aliasing）壁垒**

编译器生性多疑，它默认认为传入的两个指针（如 `d_in` 和 `d_out`）可能指向同一块物理内存。这种顾虑会导致编译器不敢大胆地将全局内存数据提前预取到极速的寄存器中。

加上 `__restrict__` 就等于向编译器做出绝对保证：“**在当前算子内，只有这个指针是访问该内存块的唯一途径！**” 这能让编译器毫无顾忌地进行指令重排和激进的寄存器缓存。

下面是一个包含完美合并访问（Coalesced）的完整可运行 CUDA 程序。请在 WSL 中新建 `global_memory_demo.cu` 并写入以下代码：

```c++
#include <iostream>
#include <cuda_runtime.h>

// ==========================================
// 模块一：全局内存算子 (Kernel Function)
// ==========================================
// 原始版本（性能一般）：
//__global__ void scaleGlobalMemoryKernel(float *d_out, float *d_in, float scale, int numElements) 

// 工业级满血版本（强烈推荐）：
__global__ void scaleGlobalMemoryKernel(float * __restrict__ d_out, const float * __restrict__ d_in, float scale, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        d_out[idx] = d_in[idx] * scale;
    }
}

// ==========================================
// 模块二：CPU 主控程序 (Host Program 带计时器)
// ==========================================
int main() {
    int numElements = 10000000; // 1000 万个浮点数
    size_t size = numElements * sizeof(float);

    std::cout << "正在初始化 1000 万个元素的全局内存测试数据..." << std::endl;

    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    for (int i = 0; i < numElements; ++i) {
        h_in[i] = 1.5f;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // ==========================================
    // 定义 CUDA 计时器事件
    // ==========================================
    cudaEvent_t startTotal, stopTotal, startKernel, stopKernel;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);

    // 【1. 开始记录“端到端总时间”（包含内存分配、H2D搬运、Kernel执行、D2H搬运）】
    cudaEventRecord(startTotal, 0);

    // 3. 将数据从 CPU 搬运到 GPU 全局内存
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "正在启动全局内存缩放算子..." << std::endl;

    // 【2. 开始记录“纯 Kernel 执行时间”】
    cudaEventRecord(startKernel, 0);
    
    scaleGlobalMemoryKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, 2.0f, numElements);
    
    // 【3. 停止“纯 Kernel 计时”】
    cudaEventRecord(stopKernel, 0);
    cudaEventSynchronize(stopKernel); // 等待 GPU 算子执行完毕

    // 6. 将计算结果从 GPU 搬回 CPU
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 【4. 停止“端到端总计时”】
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);

    // 计算并打印耗时
    float millisecondsTotal = 0;
    float millisecondsKernel = 0;
    cudaEventElapsedTime(&millisecondsTotal, startTotal, stopTotal);
    cudaEventElapsedTime(&millisecondsKernel, startKernel, stopKernel);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "[性能报告] 全局内存连续访问 (Global Memory):" << std::endl;
    std::cout << "  - 纯 Kernel 执行耗时: " << millisecondsKernel << " ms" << std::endl;
    std::cout << "  - 包含数据搬运总耗时: " << millisecondsTotal << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "验证结果 [0] = " << h_out[0] << " (期望值: 3.0)" << std::endl;

    // 销毁事件
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

​                                                            

### WSL 编译与运行流程

在终端中依次输入以下命令：

```bash
nvcc global_memory_demo.cu -o global_memory_demo
./global_memory_demo
```



本程序中的内存合并访问（Coalesced Access）最核心地体现在 `scaleGlobalMemoryKernel` 算子的这两行代码中：

```c++
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// ... (边界检查)
d_out[idx] = d_in[idx] * scale;
```

- **局部编号的连续性：** 在 GPU 底层硬件中，每 32 个相邻的线程会被打包成一个线程束（Warp）同时执行。对于同一个 Warp 内的线程，它们所处的 Block 是相同的（`blockIdx.x` 相同），每个 Block 的大小也是固定的（`blockDim.x` 相同），唯一不同的只有它们在 Block 内部的局部编号 `threadIdx.x`。而 `threadIdx.x` 刚好是 `0, 1, 2, ..., 31` 这样严格连续递增的。
- **全局索引（地址）的连续性：** 根据一维连续索引映射公式 $I_{global} = B_{id} \times B_{dim} + T_{id}$，既然只有尾部的 `T_{id}` 是连续递增的整数，那么这 32 个线程各自计算出的全局绝对索引 `idx` 也必然是一串严格相邻的连续整数（例如 0 到 31，或者 256 到 287）。  
- **硬件级的“合并装车”：** 当代码执行到 `d_in[idx]`（读数据）和 `d_out[idx]`（写数据）时，这 32 个线程会在同一个物理瞬间向全局内存发起请求。显存控制器发现这 32 个人要抓取的数据刚好在物理内存上是一整排紧密相连的，于是就会触发“合并”机制，用极少次数的内存事务（Transaction）一次性将这些数据成块地拉取或写入。  

这就是“极速流水线模式”在代码层面的完美映射：每个线程通过计算出连续的 `idx`，确保了对数组的访问是完全对齐且相邻的。  

### 类比： 快递分发与合并装车

- **非合并访问（慢速混乱模式）：**

  32个工人（一个线程束）要去仓库里拿包裹。工号 0 的人拿第 100 号包裹，工号 1 的人拿第 3 号包裹，工号 2 的人拿第 888 号包裹……仓库管理员（显存控制器）得满仓库跑 32 次，累得半死，效率极低。

- **合并访问（极速流水线模式）：**

  工号 0 的人拿第 0 号包裹，工号 1 的人拿第 1 号，工号 2 的人拿第 2 号……一直到工号 31 的人拿第 31 号包裹。大家拿的刚好是一整排**紧密相连**的货物。仓库管理员直接用一辆大卡车，“轰”的一下把这连续 32 个包裹一网打尽，一次性运给这 32 个人。这就是全局内存合并访问（Coalesced Access）的威力！



# **硬件级的量化规则（128 字节对齐事务）**

在 GPU 硬件底层，显存控制器（Memory Controller）并不以“单个变量”为单位来取数据，而是以 32 字节或 128 字节的连续物理内存块（Transaction，内存事务）为基本单位来发起请求。

正如笔记中所述，全局内存十分遥远且缓慢，SM 从显存拿一次数据需要付出 200 到 400 个时钟周期的等待时间。因此，尽可能减少硬件发起的“内存事务数量”是优化的核心。  

假设一个 Warp（32 个线程）都要读取单精度浮点数（`float`，占 4 字节），总共需要读取 $32 \times 4 = 128$ 字节的数据。此时不同的访问模式会导致截然不同的硬件开销：

| **访问模式**                 | **场景描述**                                                 | **显存地址映射公式示例**                        | **产生的硬件事务数量 (Transaction)**    | **显存总线带宽利用率**                      |
| ---------------------------- | ------------------------------------------------------------ | ----------------------------------------------- | --------------------------------------- | ------------------------------------------- |
| **最佳：连续且对齐**         | 32 个线程按顺序访问一段连续且首地址刚好是 128 字节整数倍（对齐）的内存。 | `base + threadIdx.x * sizeof(float)`            | **1 次** (128 字节满载事务)             | **100%** (一次拉满，效率最高)               |
| **次佳：连续但未对齐**       | 32 个线程按顺序访问，但首地址不是 128 的整数倍。             | `(未对齐的 base) + threadIdx.x * sizeof(float)` | **2 次** (128 字节事务)                 | **约 50%** (因为跨越了两个内存块的硬件边界) |
| **极差：跨步访问 (Strided)** | 线程跳跃式访问（例如步长为 2）。32 个数据散布在 256 字节的跨度内。 | `base + threadIdx.x * 2 * sizeof(float)`        | **2~4 次** (读取了大量不需要的中间数据) | **< 50%** (带宽被无用数据严重浪费)          |
| **最差：随机访问 (Random)**  | 32 个线程访问完全无关的离散地址（如查表或间接寻址）。        | `base + random_index[threadIdx.x]`              | **高达 32 次** (降级为 32 字节碎片事务) | **< 12.5%** (请求暴增，效率极度低下)        |

> **核心结论：** 完美合并访问发生的条件，不仅需要 32 个线程访问的数据连续，还需要它们刚好落在一个 128 字节对齐的物理内存块内。否则，显存控制器就需要发起多次快递。 

```c++
#include <iostream>
#include <cuda_runtime.h>

// ==========================================
// Kernel 1：完美对齐的连续访问 (Baseline)
// ==========================================
__global__ void alignedKernel(float * __restrict__ d_out, const float * __restrict__ d_in, float scale, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // 每个线程老老实实读自己对应位置的数据
        d_out[idx] = d_in[idx] * scale;
    }
}

// ==========================================
// Kernel 2：缓存终结者 - 跨步访问 (Strided)
// ==========================================
__global__ void stridedKernel(float * __restrict__ d_out, const float * __restrict__ d_in, float scale, int numElements, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // 核心暴击点：乘上 stride，让相邻线程去读相隔十万八千里的数据
        d_out[idx] = d_in[idx * stride] * scale;
    }
}

int main() {
    // 基础运算量：1000 万次乘法
    int numElements = 10000000; 
    // 跨步大小：32 (这意味着每个线程读取地址间隔 128 字节)
    int stride = 32; 
    
    // 输入数组需要容纳跨步后的最大索引，总计约 1.28 GB
    size_t size_in = numElements * stride * sizeof(float); 
    // 输出数组正常大小即可，约 40 MB (我们只测试读取带宽的瓶颈)
    size_t size_out = numElements * sizeof(float); 

    std::cout << "正在初始化跨步测试数据 (申请约 1.3 GB 显存)..." << std::endl;

    float *h_in = (float *)malloc(size_in);
    for (size_t i = 0; i < (size_t)numElements * stride; ++i) {
        h_in[i] = 1.0f;
    }

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void **)&d_in, size_in);
    cudaMalloc((void **)&d_out, size_out);
    cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float msAligned = 0, msStrided = 0;

    // --- 预热 ---
    alignedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, 2.0f, numElements);
    cudaDeviceSynchronize();

    // --- 测试 1：完美对齐 (连续读取) ---
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) { 
        alignedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, 2.0f, numElements);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msAligned, start, stop);

    // --- 测试 2：跨步读取 (缓存粉碎机) ---
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        stridedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, 2.0f, numElements, stride);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msStrided, start, stop);

    std::cout << "\n========================================" << std::endl;
    std::cout << "[跨步访问 (Strided Access) 破坏性测试]" << std::endl;
    std::cout << "  - 完美连续读取耗时      : " << msAligned / 10.0f << " ms (单次平均)" << std::endl;
    std::cout << "  - 步长 32 跨步读取耗时  : " << msStrided / 10.0f << " ms (单次平均)" << std::endl;
    std::cout << "  - 性能暴跌比例 (耗时增加): " << ((msStrided - msAligned) / msAligned) * 100 << "%" << std::endl;
    std::cout << "========================================\n" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in);

    return 0;
}
```

```
nvcc aligned_memory_test.cu -o aligned_memory_test
./aligned_memory_test
```

