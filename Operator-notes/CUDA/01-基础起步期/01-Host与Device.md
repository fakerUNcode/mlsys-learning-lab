# Host 与 Device

在写下任何 CUDA 代码之前，我们需要在你的大脑中建立几个核心概念：

1. **什么是“算子（Operator）”？**

   在深度学习中，算子就是**对数据进行特定数学运算的函数**。比如“矩阵乘法”、“向量加法”、“激活函数（ReLU）”，它们在底层就是一段代码。写 CUDA 算子，就是用 C++ 配合 GPU 的特性，把这些数学运算写得极快。

2. **CPU 与 GPU 的分工（Host 与 Device）**

   - **Host（主机 = CPU + 内存）：** 负责逻辑控制、读取文件、分配任务。它是“包工头”。
   - **Device（设备 = GPU + 显存）：** 负责极其庞大但简单的重复性计算。它是拥有成千上万名“工人”的“工厂”。

3. **C 语言指针基础复习**

   因为 GPU 有自己独立的内存（显存），我们需要用指针（Pointer）来记录数据存放在显存中的哪个地址。

   - `int *a;` 表示 `a` 是一个用来存放整数内存地址的变量。

4. **函数执行空间（修饰符速查）**
   CUDA 拓展了 C++ 语法，用修饰符明确区分代码的“调用方”与“执行方”：

   | 修饰符                | 执行方 (在哪跑) | 调用方 (谁来调) | 核心用途                                                     |
   | :-------------------- | :-------------- | :-------------- | :----------------------------------------------------------- |
   | `__global__`          | Device (GPU)    | Host (CPU)      | **Kernel 入口**，由 CPU 派发给成千上万个 GPU 线程并行执行    |
   | `__device__`          | Device (GPU)    | Device (GPU)    | **GPU 内部辅助函数**，仅能在 Kernel 或其它 device 函数内调用 |
   | `__host__`            | Host (CPU)      | Host (CPU)      | 普通 C++ 函数（默认行为，可省略）                            |
   | `__host__ __device__` | 双方皆可        | 双方皆可        | 编译器同时生成 CPU 和 GPU 两个版本，常见于通用数学工具函数   |

# 探查硬件参数：我们在怎样的工厂里施工？

在设计线程分配方案前，必须先看清物理硬件的边界限制。CUDA 提供了查询 API：

```cpp
#include <stdio.h>

int main() {
    int deviceId = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    printf("GPU 设备名称: %s\n", prop.name);
    printf("流式多处理器(SM)数量: %d\n", prop.multiProcessorCount);
    printf("每个 Block 允许的最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("每个 SM 允许的最大常驻线程数: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("每个 Block 可用的共享内存大小: %zu KB\n", prop.sharedMemPerBlock / 1024);
    return 0;
}
```

**关键启示**：`prop.maxThreadsPerBlock` 通常为 1024。这意味着在设计 Block 尺寸时，单个 Block 里的线程总数绝不能超过这一物理极限。



# 核心数学公式与索引推导

在 CUDA 中，成千上万个“工人”（线程）会同时执行同一段代码。为了让每个工人去处理不同的数据，工人必须知道自己的“工号（全局索引）”。

这是 CUDA 编程中最核心的基础公式。

## 全局索引计算公式

$$I_{global} = B_{id} \times B_{dim} + T_{id}$$

## 符号全翻译

- $I_{global}$：当前线程的全局唯一索引（Global Index），即“绝对工号”。

- $B_{id}$：当前线程所在的线程块的编号（Block ID），在代码中对应 `blockIdx.x`。

- $B_{dim}$：每个线程块里面包含的线程总数（Block Dimension），在代码中对应 `blockDim.x`。

- $T_{id}$：当前线程在自己所属线程块内部的局部编号（Thread ID），在代码中对应 `threadIdx.x`。

  

## 详细推导过程（等号说明书）

为了计算出当前工人的绝对工号，我们需要分两步：

$$I_{base} = B_{id} \times B_{dim}$$

(计算当前线程块之前的总人数)

(即：前面的块数乘以每块的人数)

$$I_{global} = I_{base} + T_{id}$$

(在前面总人数的基础上)

(加上当前工人在本块内的局部位次)

## Kernel 的异步启动与同步机制

当你写下 `myKernel<<<grid, block>>>(...);` 时，需要注意：

- **异步发射**：CPU（Host）向 GPU（Device）下达启动指令后，**不会停下等待 GPU 跑完**，而是瞬间返回并继续执行 CPU 的下一行代码。
- **隐式同步 vs 显式同步**：
  - `cudaMemcpy(..., cudaMemcpyDeviceToHost)`：属于**隐式同步点**。CPU 会阻塞在此处，直到 GPU 把计算做完并把数据写回内存。
  - `cudaDeviceSynchronize()`：属于**显式同步**。强行让 CPU 等待 GPU 队列中所有任务执行完毕。常用于测量 Kernel 实际执行耗时。



# 工程防护底线：CUDA 错误检查宏

CUDA API 失败（如显存不足、指针越界、非法的线程配置）通常不会直接崩溃，而是静默返回错误码。若不加检查，会导致后续逻辑输出全零或随机垃圾值。

工业界标准做法是封装 `do { ... } while(0)` 错误检查宏：

```cpp
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",        \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 1. 检查 Runtime API 调用（内存分配、数据拷贝）
// CUDA_CHECK(cudaMalloc(&d_A, size));
// CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

// 2. 检查异步 Kernel 的启动语法错误
// myKernel<<<grid, block>>>(d_A);
// CUDA_CHECK(cudaGetLastError()); // 抓取启动参数是否非法

// 3. 检查异步 Kernel 的实际运行错误
// CUDA_CHECK(cudaDeviceSynchronize()); // 等待执行并检查运行时错误
```

# 完整代码示例

本程序使用标准分离内存，分离维护两套（`h_A` 与 `d_A`），显式手动调用 `cudaMemcpy`，适用高性能算子落地、精细显存掌控（生产环境标准）

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ==========================================
// 1. 工程防护基石：标准错误检查宏
// ==========================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error:\n  File: %s\n  Line: %d\n  Code: %d (%s)\n  Call: %s\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ==========================================
// 2. 函数修饰符演示
// ==========================================

// __host__ __device__：双端编译的数学辅助工具函数
// 既可以在 CPU 上跑，也可以在 GPU 内部调用
__host__ __device__ float add_op(float a, float b) {
    return a + b;
}

// __global__：由 CPU(Host) 派发，在成千上万个 GPU(Device) 线程上并行执行的入口 Kernel
__global__ void vectorAddKernel(const float *d_A, const float *d_B, float *d_C, int n) {
    // 核心索引推导：计算当前工人的全局唯一绝对工号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界保护：防止线程总数超过数组实际大小时发生非法显存越界访问
    if (idx < n) {
        d_C[idx] = add_op(d_A[idx], d_B[idx]);
    }
}

// ==========================================
// 3. 主程序（Host 端流程）
// ==========================================
int main() {
    // --------------------------------------------------
    // Step 0: 认识你的施工工厂（硬件参数查询）
    // --------------------------------------------------
    int dev_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    printf("================ GPU 硬件属性 ================\n");
    printf("设备名称: %s\n", prop.name);
    printf("流式多处理器(SM)数量: %d\n", prop.multiProcessorCount);
    printf("每个 Block 允许的最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("全局显存总量: %.2f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("==============================================\n\n");

    // --------------------------------------------------
    // Step 1: 准备数据规模
    // --------------------------------------------------
    const int N = 1 << 20; // 1048576 个元素（约 100 万）
    const size_t bytes = N * sizeof(float);

    // --------------------------------------------------
    // Step 2: 分配 Host 内存并初始化
    // --------------------------------------------------
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Host 内存分配失败\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // --------------------------------------------------
    // Step 3: 分配 Device (显存) 空间
    // --------------------------------------------------
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // --------------------------------------------------
    // Step 4: 将数据从 Host 拷贝到 Device (H2D)
    // --------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // --------------------------------------------------
    // Step 5: 计算 Execution Configuration（执行配置）
    // --------------------------------------------------
    // 每个 Block 安排 256 个线程（通常选择 32 的整数倍，且不超过 prop.maxThreadsPerBlock）
    int threadsPerBlock = 256;
    // 向上取整计算需要的 Block 数量：ceil(N / threadsPerBlock)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("启动 Kernel: <<<Grid: %d, Block: %d>>>\n", blocksPerGrid, threadsPerBlock);

    // --------------------------------------------------
    // Step 6: 异步启动 Kernel & 错误捕获两步走
    // --------------------------------------------------
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 检查点 A: 抓取 Kernel 启动参数是否合法（如配置参数非法、资源超限）
    CUDA_CHECK(cudaGetLastError());

    // 检查点 B: 显式同步，强制 CPU 等待 GPU 运行完毕，同时捕获 Kernel 执行过程中的显存非法访问等硬件错误
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------
    // Step 7: 将结果从 Device 拷回 Host (D2H，自带隐式同步)
    // --------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------
    // Step 8: 在 CPU 端验证结果的准确性
    // --------------------------------------------------
    bool is_correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != 3.0f) {
            is_correct = false;
            printf("校验失败: h_C[%d] = %f (预期值 3.0f)\n", i, h_C[i]);
            break;
        }
    }

    if (is_correct) {
        printf("计算校验成功！所有数据均正确无误。\n");
    }

    // --------------------------------------------------
    // Step 9: 释放资源（先 Device 后 Host）
    // --------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

`vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);`

$$\underbrace{\text{vectorAddKernel}}_{\text{执行的任务}} \overbrace{\text{<<<blocksPerGrid, threadsPerBlock>>>}}^{\text{组织架构与兵力部署}} (\underbrace{\text{d\_A, d\_B, d\_C, N}}_{\text{任务所需原料与参数}});$$

1. `vectorAddKernel`（任务名称）

- 必须是一个带有 `__global__` 修饰符声明的函数。
- 它代表由 CPU 派发、但在 GPU 硬件核心上执行的代码实体。



2. `<<<blocksPerGrid, threadsPerBlock>>>`（执行配置，Execution Configuration）

- 这是 CUDA 扩展 C++ 的特殊语法，用来告诉 GPU 的硬件调度器**该分配多少人、怎么组织队伍**：

- **`threadsPerBlock`（每块线程数 / 连队规模）**：

  - 每个 Block（线程块）内部包含多少个线程。
  - 例如若设为 `256`，代表将 256 个线程编为一组，同一组内的线程共享同一块 Shared Memory（共享内存）并能通过 `__syncthreads()` 同步。

- **`blocksPerGrid`（网格块数 / 连队数量）**：

  - 整个 Grid（网格）总共要启动多少个 Block。
  - 例如若设为 `4096`，代表在 GPU 上同时部署 4096 个连队。

- **总兵力（启动的总线程数）**：

  $$\text{总线程数} = \text{blocksPerGrid} \times \text{threadsPerBlock}$$

  若为 $4096 \times 256$，则 GPU 会瞬间唤醒 **1,048,576 个线程**。



3. `(d_A, d_B, d_C, N)`（参数传递）

- **`d_A, d_B, d_C`**：存放于 GPU 显存（Device Memory）中的数据首地址指针（以 `d_` 开头是工业界通用命名规范，表示 Device）。
- **`N`**：普通数值（值传递），告知 Kernel 实际需要计算的向量长度。



Kernel 启动例如：

```
reduceSequential<<<4, 256>>>(
    d_in,
    d_out,
    n
);
```

表示：

```
Grid
│
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread 255
│
├── Block 1
│   ├── Thread 0
│   ├── ...
│
├── Block 2
│
└── Block 3
```

注意一个关键点：

> `threadIdx.x` 在每个 Block 内都会重新从 0 开始。

所以：

```
Block 0 有 tid = 0
Block 1 也有 tid = 0
Block 2 也有 tid = 0
```

因此单独使用 `tid` 无法定位整个数组。



# 补充说明

这两组对比是 CUDA 内存体系中极易混淆但也最核心的概念。

## 内存管理模式：统一内存 vs 显式分离内存

这是**编程范式与数据搬运策略**层面的对比，决定了程序员如何管理 Host 与 Device 之间的数据流。

### 核心对比

| **对比维度**      | **显式分离内存（Explicit Memory）**                       | **统一内存（Unified Memory）**                               |
| ----------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| **API 接口**      | `malloc` / `free` (CPU) + `cudaMalloc` / `cudaFree` (GPU) | `cudaMallocManaged` / `cudaFree`                             |
| **指针管理**      | 双套独立指针（`h_ptr` 位于 Host，`d_ptr` 位于 Device）    | 单套共享指针（CPU 与 GPU 使用同一个虚拟地址指针）            |
| **数据迁移方式**  | **显式手动搬运**：由程序员显式调用 `cudaMemcpy` 指定方向  | **隐式按需迁移**：底层依托硬件缺页异常（Page Fault）自动搬运 |
| **硬件/系统开销** | 传输耗时完全可预测，无 Page Fault 中断开销                | 首次访问存在页错误中断延迟，性能有不可控波动                 |
| **显存超额申请**  | 不支持。显存不足时 `cudaMalloc` 直接报错返回错误码        | 支持（Over-subscription）。显存不足时自动将冷数据换出到 CPU 内存 |
| **典型应用场景**  | **高性能生产环境、核心算子开发、追求极限吞吐**            | **算法原型验证、复杂树/图结构（避免深拷贝指针修复）、教学开发** |

### 底层数据迁移机制差异

- **显式分离内存**：就像传统的“大卡车送货”。必须由 CPU 发起装车命令（`cudaMemcpy`），一次性把指定大小的连续物理内存通过 PCIe 总线拉到显存。数据在什么时候过去、什么时候回来完全由代码决定。  
- **统一内存**：就像“外卖按需配送”。调用 `cudaMallocManaged` 时，系统只分配了一段虚拟地址空间，并没有立刻把数据塞进 GPU 显存。当 GPU 线程第一次尝试读取该地址时，触发硬件 **Page Fault（缺页异常）**，CUDA 驱动中断当前指令，把对应的内存页（通常是 4KB/64KB 粒度）通过 PCIe 搬到显存中，然后再恢复线程执行。





## 存储层次结构：共享内存 vs 全局内存

这是物理硬件位置与存储层级（Memory Hierarchy）层面的对比，决定了 GPU 算子能跑多快。



###  核心对比

| **对比维度**          | **全局内存（Global Memory / 显存）**                         | **共享内存（Shared Memory / 块内缓存）**                     |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **物理位置**          | 位于 GPU 芯片外的板载 DRAM（即显卡上的 16GB/24GB 显存）      | 位于 GPU 芯片内部的每个 SM（流多处理器）内部，与 L1 缓存共享物理 SRAM |
| **容量大小**          | 巨大：几 GB 到几十 GB                                        | 极小：几十 KB 到百余 KB（如每个 Block 通常 48KB ~ 99KB）     |
| **访问延迟**          | **极慢**：通常需要 **200 ~ 400 个时钟周期**                  | **极快**：通常仅需 **20 ~ 30 个时钟周期**（接近寄存器速度）  |
| **可见范围（Scope）** | **全局可见**：网格中的所有线程（所有 Block）以及 Host 端都可以访问 | **块内可见**：仅对同一个 Block 内部的线程可见，Block 间数据隔离 |
| **生命周期**          | 由 Host 端显式申请到显式释放（`cudaMalloc` 到 `cudaFree`）   | 随着所属 Block 的开始而创建，随着 Block 执行结束而销毁       |
| **代码声明方式**      | 普通指针传参（如 `float *d_A`）                              | 静态声明：`__shared__ float s_data[256];`  动态声明：`extern __shared__ float s_data[];` |
| **数据同步机制**      | 跨 Block 无法直接轻量同步，需结束 Kernel 或借由原子操作      | 使用 `__syncthreads()` 实现 Block 内线程间的硬件级栅障同步   |

## 直观类比与优化法则

- **全局内存**：就像**城市郊区的大型总仓库**。容量管够，能放几十吨货物，但每次派车去取货都要花几十分钟（几百个时钟周期的高延迟）。如果每个工人都动不动跑一趟总仓库，GPU 核心大部分时间都在干等数据发呆。  
- **共享内存**：就像**车间工位旁边的小工具箱**。容量只有一丁点（几十 KB），但一伸手就能拿到（超高带宽、极低延迟）。
- **经典优化模式（Tiling / 分块瓷砖算法）**： 由于全局内存太慢，高性能算子（如矩阵乘法 GEMM、卷积）的标准写法是：  
  1. 让同一个 Block 内的 256 个线程分工合作，**协同把一块数据从慢速的“全局内存”搬进快速的“共享内存”**。
  2. 调用 `__syncthreads()` 确保车间内的工人都把零件搬到位了。
  3. 所有线程直接在超高速的“共享内存”中重复读取这些数据进行乘加运算。
  4. 算完后，再把最终结果一次性写回“全局内存”。通过这种“复用（Data Reuse）”，原本成百上千次的慢速全局显存访问被压缩到了极少数几次，算子性能得以成倍提升。
