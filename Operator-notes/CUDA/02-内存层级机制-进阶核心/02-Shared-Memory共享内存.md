

如果说 Global Memory（全局内存）是远在郊区、去一次要走几百步的“中央大仓库”，那么 Shared Memory（共享内存）就是每个车间内部、触手可及的“专属工作台”。学会使用共享内存，是区分 CUDA 程序员是“初学者”还是“高手”的分水岭。



# 基础概念复习

在写代码前，我们必须在脑海中重建关于 Shared Memory 的三个核心概念：

1. **什么是 Shared Memory（共享内存）？**
   - 它是一块集成在 GPU 芯片内部（On-Chip）、极小但极快的内存空间。
   - **作用范围：** 它是**属于同一个 Block（线程块）内部的所有线程共享的**。Block A 的线程绝对访问不到 Block B 的共享内存。
   
     - **Shared Memory 专属于对应的 Block**
   
       ```
       __shared__ float s_data[256];
       ```
   
       意味着：
   
       ```
       Block 0 → 自己的 s_data[256]
       
       Block 1 → 自己的 s_data[256]
       
       Block 2 → 自己的 s_data[256]
       ```
   
       不同 Block **不能直接共享这个数组**。
   - **速度优势：** 它的访问延迟只有全局内存的几十分之一甚至上百分之一（大约 1~2 个时钟周期）。
2. **什么是 `__shared__` 关键字？**
   - 这是在 CUDA 中声明共享内存的专属修饰符。当你在 Kernel 函数里写下 `__shared__ float s_data[256];` 时，编译器就知道这块数据要存放在高速的共享内存里。
3. **什么是 `__syncthreads()`（同步屏障）？**
   - 因为同一个 Block 里的线程是并发执行的，如果线程 A 正在往共享内存里写数据，而线程 B 却急着去读这个数据，就会发生“数据竞态（Data Race）”（读到乱码）。
   - `__syncthreads()` 的作用就是设置一个“集合点”：当前 Block 里的所有线程必须全部到达这一行代码后，才能继续往下执行。谁先到谁就得在原地等其他人。

## **静态分配 vs 动态分配（Dynamic Shared Memory）** 

在前面的示例中，我们使用的是**静态分配**，即在代码编译期就写死大小（例如 `__shared__ float s_cache[256];`）。但在真实的工业工程中，每个 Block 需要处理的数据维度往往是在程序运行时刻（Runtime）由外部输入的变量决定的。这时候就需要使用**动态共享内存**。  

- **代码声明区别：**

  不再写具体数字，而是加上 `extern` 关键字，声明为一个未定大小的数组：

  ```c++
  // 必须写在 Kernel 函数内部或外部，并且不能指定具体大小
  extern __shared__ float s_dynamic[]; 
  ```

- **启动算子的变化：**

  当使用动态共享内存时，必须在主机端（CPU）启动 Kernel 的 `<<< >>>` 尖括号中，通过**第三个参数**明确告知 GPU 当前这个 Block 需要申请多少字节（Bytes）的共享内存：

  

  ```c++
  // 计算动态共享内存的字节数
  size_t sharedMemBytes = threadsPerBlock * sizeof(float);
  
  // 尖括号的第三个参数传入 sharedMemBytes
  myDynamicKernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(...);
  ```

  *(注：如果一个 Kernel 中需要分配多个不同类型的动态共享内存数组，通常需要通过申请一整大块 `extern __shared__ char smem[]`，然后在代码中手动计算指针偏移量来进行拆分分配。)*

# 核心公式

在接下来的共享内存实例中，我们会让每个线程把全局内存的数据搬运到共享内存的工作台上，然后利用共享内存进行高效的协同计算。其基础索引映射公式依然是：

$$I_{global} = B_{id} \times B_{dim} + T_{id}$$

$$I_{shared} = T_{id}$$

## 符号全翻译

- $I_{global}$：线程在全局内存中的绝对索引地址。
- $I_{shared}$：线程在当前 Block 共享内存数组中的局部存放位置。
- $B_{id}$：当前线程块的编号，对应代码中的 `blockIdx.x`。
- $B_{dim}$：每个线程块包含的线程总数，对应代码中的 `blockDim.x`。
- $T_{id}$：线程在块内部的局部编号，对应代码中的 `threadIdx.x`。

## 推导

为了让每个线程安全地把数据从“大仓库（全局内存）”搬到“工作台（共享内存）”，我们需要执行以下两步推导：

$$I_{global} = B_{id} \times B_{dim} + T_{id}$$

(计算当前线程在全班组中负责的绝对数据位置)

$$\Rightarrow I_{shared} = T_{id}$$

(将该数据搬运到共享内存中时)

(直接存放在以自己局部工号 T_id 为下标的槽位里)



# 真正的共享内存实例程序 (`shared_memory_demo.cu`)

下面是一个使用 **Shared Memory 进行数组缓存加速（协作式搬运与缩放）** 的完整可运行 CUDA 程序。请在 WSL 中新建 `shared_memory_demo.cu` 并写入以下代码：

```c++
#include <iostream>
#include <cuda_runtime.h>

// ==========================================
// 模块一：使用共享内存的算子 (Kernel Function)
// ==========================================
__global__ void sharedMemoryDemoKernel(float *d_out, const float *d_in, int numElements) {
    __shared__ float s_cache[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < numElements) {
        s_cache[tid] = d_in[idx];
    } else {
        s_cache[tid] = 0.0f;
    }

    __syncthreads();

    if (idx < numElements) {
        float val = s_cache[tid] * 3.0f;
        d_out[idx] = val;
    }
}

// ==========================================
// 模块二：CPU 主控程序 (Host Program 带计时器)
// ==========================================
int main() {
    int numElements = 10000000; // 为了公平对比，统一拉高到 1000 万个元素
    size_t size = numElements * sizeof(float);

    std::cout << "正在初始化 1000 万个元素的共享内存测试数据..." << std::endl;

    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    for (int i = 0; i < numElements; ++i) {
        h_in[i] = 2.0f;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // ==========================================
    // 核心新增：定义 CUDA 计时器事件
    // ==========================================
    cudaEvent_t startTotal, stopTotal, startKernel, stopKernel;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);

    // 1. 开始端到端计时
    cudaEventRecord(startTotal, 0);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "正在启动共享内存加速算子..." << std::endl;

    // 2. 开始纯 Kernel 计时
    cudaEventRecord(startKernel, 0);
    
    sharedMemoryDemoKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, numElements);
    
    // 3. 停止纯 Kernel 计时
    cudaEventRecord(stopKernel, 0);
    cudaEventSynchronize(stopKernel);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 4. 停止端到端计时
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);

    float millisecondsTotal = 0;
    float millisecondsKernel = 0;
    cudaEventElapsedTime(&millisecondsTotal, startTotal, stopTotal);
    cudaEventElapsedTime(&millisecondsKernel, startKernel, stopKernel);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "[性能报告] 共享内存协作缓存 (Shared Memory):" << std::endl;
    std::cout << "  - 纯 Kernel 执行耗时: " << millisecondsKernel << " ms" << std::endl;
    std::cout << "  - 包含数据搬运总耗时: " << millisecondsTotal << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "验证结果 [0] = " << h_out[0] << " (期望值: 6.0)" << std::endl;

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

## WSL 编译与运行流程

在你的 WSL 终端中，依次输入以下命令进行编译和运行：

```BASH
nvcc shared_memory_demo.cu -o shared_memory_demo
./shared_memory_demo
```

**预期终端输出：**

```
正在初始化共享内存测试数据...
正在启动共享内存加速算子...
验证结果 [0] 位置: h_out[0] = 6 (期望值: 6.0)
验证结果 [1023] 位置: h_out[1023] = 6 (期望值: 6.0)
共享内存演示程序执行完毕！
```



# 共享内存 vs 全局内存

```bash
(base) futaba_sakura@FutabaSakura:~/workspace/operator$ ./global_memory_demo
正在初始化 1000 万个元素的全局内存测试数据...
正在启动全局内存缩放算子...
----------------------------------------
[性能报告] 全局内存连续访问 (Global Memory):
  - 纯 Kernel 执行耗时: 0.597504 ms
  - 包含数据搬运总耗时: 20.2106 ms
----------------------------------------
验证结果 [0] = 3 (期望值: 3.0)
(base) futaba_sakura@FutabaSakura:~/workspace/operator$ ./shared_memory_demo
正在初始化 1000 万个元素的共享内存测试数据...
正在启动共享内存加速算子...
----------------------------------------
[性能报告] 共享内存协作缓存 (Shared Memory):
  - 纯 Kernel 执行耗时: 0.593024 ms
  - 包含数据搬运总耗时: 22.1789 ms
----------------------------------------
验证结果 [0] = 6 (期望值: 6.0)
```

- **纯 Kernel 耗时**：两者几乎完全一致（约 0.59 ms，在误差范围内）。
- **总耗时（含数据搬运）**：全局内存版本反而比共享内存版本还要快了近 2 毫秒（20.2ms vs 22.1ms）。



之所以会出现“全局内存比共享内存更快（或持平）”的反直觉现象，核心原因在于：**这个简单的“缩放算子（Scale）”根本用错了场景，强行使用共享内存反而成了“多此一举”的累赘。**



我们可以从以下三个底层原因来彻底剖析：

## 核心痛点 —— “数据复用”

共享内存（Shared Memory）之所以能加速，绝不是因为它本身的读写速度快（虽然它确实快），而是因为它解决了“同一个全局内存数据被多个线程重复读取”的痛点。

- **什么时候该用共享内存？（如矩阵乘法 Matrix Multiplication、图像卷积 Stencil）**

  在这些复杂算法中，全局内存里的同一个数据，会被周围好几个不同的线程反复读取几十次。如果我们先把这个数据缓存到共享内存里，大家直接在工作台上拿，就能避免几十次去挤慢速的全局内存大仓库。

- **为什么本程序不需要？（单次使用 Single-Use）**

  在我们刚才写的 `scaleGlobalMemoryKernel` 和 `sharedMemoryDemoKernel` 中，每个浮点数从全局内存读进来，**乘以缩放因子后，立刻就写回去了，全程只被使用了一次**。

  这就好比你去超市买一瓶矿泉水（全局内存），本来拿起来直接喝就行，你非要先倒进一个杯子里（共享内存）再喝。杯子虽然干净，但你多做了一道“倒水”的动作，反而浪费了时间。

## 多出来的“额外开销（Overhead）”

在共享内存版本的代码中，为了完成“倒水”这个动作，我们强行增加了以下几行额外的指令：

1. `s_cache[tid] = d_in[idx];` （多了一次全局内存写入共享内存的指令）
2. `__syncthreads();` （所有线程必须停下来互相等待的同步指令）
3. `float val = s_cache[tid] * 3.0f;` （多了一次从共享内存读取的指令）

这些额外的指令需要消耗硬件时钟周期。正因如此，它不仅没有带来加速，反而因为指令变多、逻辑变复杂，拉长了总耗时。



## 第三部分：硬件瓶颈 —— 内存带宽（Bandwidth Bound）

在当前的简单算子中，程序的性能天花板根本不在于“计算单元算得慢”，而在于“显存搬运数据速度的极限（Memory Bandwidth Limit）”。



- 无论是全局内存版本还是共享内存版本，它们从显存（Global Memory）里读取的数据总量是**一模一样的**（都是 1000 万个浮点数），写回显存的数据量也**一模一样**。
- 既然总搬运量没有减少，GPU 的显存控制器就已经被榨干了。共享内存无法凭空变出更多的显存带宽。

## 总结与优化

- **结论：**

  对于“每个元素只用一次（Element-wise）”的简单运算（如向量加法、简单缩放、激活函数 ReLU），直接用全局内存（配合合并访问）是性能最高、代码最简的方案。不要为了用共享内存而用共享内存。 只有当算法涉及到大量的交叉访存和数据复用时，共享内存才是真正的救世主。



## 硬件容量限制与占用率权衡

进阶高压线：共享内存的容量限制与占用率 (Occupancy) 权衡

虽然我们将共享内存比作“专属工作台”，可以极大提升数据复用效率，但这个“工作台”的物理面积是极度稀缺的。  

**1. 物理容量的硬上限**

现代 GPU 的流多处理器（SM）内部，留给共享内存的物理空间非常小。例如：

- **Turing/Ampere 架构：** 每个 SM 的共享内存总量最高通常只有 **100 KB 到 164 KB** 左右。

**2. 核心性能指标：占用率 (Occupancy)**

GPU 之所以能靠“零开销调度”掩盖全局内存的延迟，核心在于一个 SM 里面同时驻留了大量的 Block 和 Warp，这就是 **Occupancy（占用率：活跃的 Warp 数量与 SM 最大支持 Warp 数量的比例）**。

**3. 惨痛的权衡陷阱 (The Trade-off)**

共享内存的申请量会直接决定 SM 能容纳多少个 Block。

假设你的 GPU 每个 SM 只有 100 KB 的共享内存：

- **场景 A（省吃俭用）：** 每个 Block 只申请 10 KB 共享内存。那么一个 SM 在容量允许下，最多可以同时驻留 10 个 Block。这 10 个 Block 里的海量线程可以来回切换，完美隐藏延迟。
- **场景 B（贪得无厌）：** 程序员为了缓存更多数据，让每个 Block 申请了 80 KB 的共享内存。结果是，一个 SM **最多只能塞下 1 个 Block**。如果这个 Block 在等待某些指令，整个 SM 就只能处于闲置发呆状态，导致 Occupancy 暴跌，算力严重浪费。

> **工业级优化准则：**
>
> 共享内存绝对不是越大越好！在使用共享内存进行分块（Tiling）优化时，**单 Block 的共享内存使用量应尽量控制在 16KB 到 32KB 之间**。必须在“减少全局内存访问次数”和“保持足够高的 SM 占用率（Occupancy）”之间找到一个最佳平衡点。可以使用 NVIDIA 官方提供的 `CUDA Occupancy Calculator` 表格工具来推算最佳的参数配置。

# 共享内存的实用场景

```c++
#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // 每个 Block 管理 16x16 的分块

// ==========================================
// 方案一：纯全局内存实现的矩阵乘法 (无复用，每次都去显存拿)
// ==========================================
__global__ void matrixMulGlobalKernel(const float *A, const float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            // 每次循环都要从慢速的全局内存中读取 A 和 B 的元素
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// ==========================================
// 方案二：使用共享内存分块 (Tiling) 优化的矩阵乘法
// ==========================================
__global__ void matrixMulSharedKernel(const float *A, const float *B, float *C, int width) {
    // 声明两个共享内存工作台，用于缓存矩阵 A 和 B 的小方块
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // 当前线程负责计算的目标矩阵 C 中的绝对行号和列号
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // 沿着横纵方向把大矩阵分成若干个 Tile 逐步推进
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int m = 0; m < numTiles; ++m) {
        // 1. 协同搬运：把当前 Tile 的数据从全局内存搬到共享内存
        if (row < width && (m * TILE_WIDTH + tx) < width) {
            s_A[ty][tx] = A[row * width + (m * TILE_WIDTH + tx)];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        if (col < width && (m * TILE_WIDTH + ty) < width) {
            s_B[ty][tx] = B[(m * TILE_WIDTH + ty) * width + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // 2. 同步：等待该 Block 内所有线程搬运完毕
        __syncthreads();

        // 3. 高速复用：直接在共享内存工作台上进行当前 Tile 的乘加运算
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // 4. 同步：等待大家都计算完了，才能进入下一轮 Tile 搬运
        __syncthreads();
    }

    // 将最终累加结果写回全局内存
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// ==========================================
// 主控程序：分别测试两者的耗时
// ==========================================
int main() {
    int width = 1024; // 1024 x 1024 的矩阵 (共约 100 万个元素，计算量极大)
    size_t size = width * width * sizeof(float);

    std::cout << "正在初始化 " << width << "x" << width << " 矩阵乘法测试数据..." << std::endl;

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_global = (float *)malloc(size);
    float *h_C_shared = (float *)malloc(size);

    for (int i = 0; i < width * width; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH); // 16x16 = 256 线程
    dim3 blocksPerGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==========================================
    // 测试 1：纯全局内存版本
    // ==========================================
    std::cout << "正在运行【纯全局内存】矩阵乘法..." << std::endl;
    cudaEventRecord(start, 0);
    
    matrixMulGlobalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeGlobal = 0;
    cudaEventElapsedTime(&timeGlobal, start, stop);
    cudaMemcpy(h_C_global, d_C, size, cudaMemcpyDeviceToHost);

    // ==========================================
    // 测试 2：共享内存分块优化版本
    // ==========================================
    std::cout << "正在运行【共享内存分块(Tiling)】矩阵乘法..." << std::endl;
    cudaEventRecord(start, 0);
    
    matrixMulSharedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeShared = 0;
    cudaEventElapsedTime(&timeShared, start, stop);
    cudaMemcpy(h_C_shared, d_C, size, cudaMemcpyDeviceToHost);

    // ==========================================
    // 性能报告打印
    // ==========================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "        1024x1024 矩阵乘法性能对比          " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  - 纯全局内存耗时: " << timeGlobal << " ms" << std::endl;
    std::cout << "  - 共享内存优化耗时: " << timeShared << " ms" << std::endl;
    std::cout << "  - 性能提升倍数: " << (timeGlobal / timeShared) << " 倍！" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "验证结果 [0,0]: Global=" << h_C_global[0] << ", Shared=" << h_C_shared[0] << " (期望值: 2048.0)" << std::endl;

    // 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_global); free(h_C_shared);

    return 0;
}
```



```
(base) futaba_sakura@FutabaSakura:~/workspace/operator$ ./matrix_mul_comparison
正在初始化 1024x1024 矩阵乘法测试数据...
正在运行【纯全局内存】矩阵乘法...
正在运行【共享内存分块(Tiling)】矩阵乘法...

========================================
        1024x1024 矩阵乘法性能对比
========================================
  - 纯全局内存耗时: 144.296 ms
  - 共享内存优化耗时: 0.92736 ms
  - 性能提升倍数: 155.598 倍！
========================================
验证结果 [0,0]: Global=2048, Shared=2048 (期望值: 2048.0)
(base) futaba_sakura@FutabaSakura:~/workspace/operator$ ./matrix_mul_comparison
正在初始化 1024x1024 矩阵乘法测试数据...
正在运行【纯全局内存】矩阵乘法...
正在运行【共享内存分块(Tiling)】矩阵乘法...

========================================
        1024x1024 矩阵乘法性能对比
========================================
  - 纯全局内存耗时: 1.91082 ms
  - 共享内存优化耗时: 1.0201 ms
  - 性能提升倍数: 1.87317 倍！
========================================
验证结果 [0,0]: Global=2048, Shared=2048 (期望值: 2048.0)
```



1. **第一次运行：** 全局内存耗时 **144.2 ms**，共享内存耗时 **0.92 ms**（差距 155 倍）。
2. **后续连续运行：** 全局内存瞬间暴降到 **1.4ms ~ 1.9ms** 之间，而共享内存依然稳定在 **0.9ms 左右**。

为什么后来全局内存又追赶上来了？

## 基础概念

在现代 GPU 架构中，除了显存（Global Memory）和共享内存（Shared Memory）之外，还隐藏着一层极其聪明、自动工作的硬件机制——**L1 / L2 硬件缓存（Cache）**。

1. **冷启动（Cold Start，第一次运行）：**
   - 当你刚打开程序、第一次调用 `matrixMulGlobalKernel` 时，GPU 芯片上的 L2 缓存（甚至显存控制器）是完全“干净”的。矩阵 $A$ 和 $B$ 的数据全部躺在遥远的物理显存里。
   - 全局内存版本因为没有任何缓存命中，每次都要去显存硬取，导致严重的带宽阻塞，耗时飙升到 **144 ms**。
2. **热启动（Warm-up，后续运行）：**
   - 当全局内存版跑过一次后，GPU 的 L2 缓存已经自动把这 1024×1024 矩阵的大部分热点数据“顺手缓存”在芯片内部的高速缓存里了。
   - 等你第二次、第三次运行纯全局内存版本时，虽然你的代码里没有写 `__shared__` 共享内存，但**硬件自动帮你做了缓存命中**！因此速度从 144ms 暴跌到了 1.4ms 左右。

## 为什么全局内存有缓存帮忙，共享内存依然更快？



1. **自动缓存（L2 Cache） vs 程序员手动精准控制（Shared Memory）：**

   - **硬件 L2 缓存：** 它是被动、盲目的。它用的是通用淘汰算法（如 LRU），当数据量太大或者访问模式复杂时，缓存很容易发生“抖动（Cache Thrashing）”，导致数据被频繁踢出、重新从显存加载。
   - **共享内存（Shared Memory）：** 它是**程序员通过代码主动、精确锁定**的高速工作台。数据一旦搬进来，绝对不会被硬件意外踢走，直到当前 Block 计算结束。这种确定性的高速控制，效率永远高于硬件盲目猜想。

2. **访存指令层面的优化：**

   共享内存版本在分块协同搬运时，最大化地保证了线程束访存的合并性与对齐性，消除了硬件缓存命中时的额外管理开销。

## 科学测试的黄金准则（Benchmark 建议）

在真实的高性能计算（HPC）和 AI 框架（如 PyTorch / TensorRT）性能测试中，为了消除这种“冷启动缓存命中”带来的数据噪声，我们通常会遵循以下规范：

1. **预热（Warm-up Runs）：** 在正式计时前，先空跑 3 到 5 次 Kernel，让所有数据自动加载进缓存。
2. **多次平均：** 正式计时时，连续运行 50 次或 100 次，取平均值作为最终性能指标。
