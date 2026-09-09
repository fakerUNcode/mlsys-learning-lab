# cudaMalloc 与 cudaMemcpy 内存管理



## 直观记忆例子：工厂大楼

- **Grid（整个 GPU 任务）：** 想象一栋巨大的办公大楼。
- **Block（线程块 $B_{id}$）：** 大楼里的某一个楼层。每个楼层的工位数量是固定的（$B_{dim}$）。
- **Thread（线程 $T_{id}$）：** 坐在楼层某个工位上的具体工人。
- **计算 $I_{global}$：** 如果你要找第 3 层的第 5 号工人（假设每层 100 人）。那么他的绝对工号就是：前面 3 层的人数（$3 \times 100$）加上他在本层的座位号（$5$），所以绝对工号是 $305$。

## 第一个 CUDA 算子实例（向量加法）

在 WSL 中，请确保你已经安装了 NVIDIA 驱动支持和 CUDA Toolkit。

新建一个文件命名为 `vector_add.cu`（`.cu` 是 CUDA 源码的后缀）。



```C++
#include <iostream>
#include <cuda_runtime.h>

// 这就是我们写的一个最基础的 CUDA 算子 (Kernel函数)
// __global__: 这是一个标记词。贴上这个标签的函数，意味着它由 CPU 下达命令启动，但是完全在 GPU 上执行。这被称为“核函数（Kernel）”。
__global__ void vectorAddOperator(const float *A, const float *B, float *C, int numElements) {
    // 1. 根据刚刚推导的公式，计算当前线程负责的全局索引 I_global
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 为了防止索引越界（比如一共1000个数据，但启动了1024个线程），必须加判断
    if (i < numElements) {
        C[i] = A[i] + B[i]; // 当前线程只负责计算这 1 个元素的加法
    }
}

int main() {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    // 1. 在 CPU (Host) 上分配内存并初始化
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 2. 在 GPU (Device) 上分配显存 (cudaMalloc)
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    //CUDA Memory Allocate（CUDA 内存分配）。它的作用是在 GPU 的显存中，强行圈出一块空白地盘，用来存放数据。
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
	// 【新增】2.1 显存数据初始化 (cudaMemset)
	// 类似于 C 语言的 memset，按字节填充显存。常用于将输出缓冲区 d_C 显式清零，防止残留脏数据影响累加等操作。
	cudaMemset(d_C, 0, size);
    // 3. 把数据从 CPU 搬运到 GPU (cudaMemcpy),
    //CUDA Memory Copy（CUDA 内存拷贝）。它就是连接 CPU 和 GPU 的那辆“快递货车”，负责把数据从一边搬到另一边。
    //cudaMemcpyHostToDevice：这是 cudaMemcpy 货车的行驶方向，表示数据从 Host（CPU）发往 Device（GPU）。
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. 设置工人组织结构 (Block 和 Grid)
    int threadsPerBlock = 256; // B_dim: 每个块 256 个线程
    // 计算需要多少个块才能覆盖所有元素 (向上取整)
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock; 

    // 5. 启动 CUDA 算子 (特殊的 <<< >>> 语法)
    vectorAddOperator<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // 6. 把计算结果从 GPU 搬运回 CPU
    //cudaMemcpyDeviceToHost：货车返程，数据从 Device（GPU）发往 Host（CPU）。
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 7. 打印验证一下结果
    std::cout << "C[0] = " << h_C[0] << std::endl; // 应该是 3.0

    // 8. 释放显存和内存，好习惯
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
```

*在 WSL 终端中编译并运行的命令：*

```sh
nvcc vector_add.cu -o vector_add
./vector_add
```





# 现代 CUDA 范式：Unified Memory（统一内存）

在上面的标准流程中，我们需要分别维护 Host 指针（`h_A`）与 Device 指针（`d_A`），并手动进行 `H2D` 和 `D2H` 拷贝。
CUDA 6.x 引入了 **Unified Memory（统一内存）**，允许 CPU 与 GPU **共享同一个指针地址空间**，底层由系统页错误机制（Page Fault）自动按需迁移数据。

## 简化版向量加法代码

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    int N = 50000;
    size_t size = N * sizeof(float);

    // 1. 单一指针分配：CPU 和 GPU 都能直接访问 A, B, C
    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // 2. CPU 直接初始化数据（无需 malloc 和 cudaMemcpyHostToDevice）
    for (int i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // 3. 启动 Kernel（此时 GPU 访问触发缺页中断，硬件自动把数据拉到显存）
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(A, B, C, N);

    // 4. 关键：必须显式等待 GPU 执行完毕！
    // 否则 CPU 会立即向下执行，读取到未计算完成的数据
    cudaDeviceSynchronize();

    // 5. CPU 直接读取结果（无需 cudaMemcpyDeviceToHost）
    std::cout << "C[0] = " << C[0] << std::endl;

    // 6. 统一释放
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
```

### 优缺点与面试考察点

- **优势**：指针数量减半，大幅简化代码架构与原型开发复杂度。
- **劣势（性能陷阱）**：初次访问时触发硬件 Page Fault 带来额外开销，延迟高于经过精细规划的手动预先拷贝（Pin Memory + 显式流水线）。
- **工程准则**：算法原型验证、教学或显存超额申请（Over-subscription）时用 Unified Memory；极限性能榨取算子仍然依赖手动显存管理。



## 性能极致：异步拷贝与 CUDA 流（Streams）

在标准流水线中，`cudaMemcpy` 默认是**同步阻塞**的：
> CPU 派发拷贝任务 $\to$ **CPU 卡住等待数据搬完** $\to$ 启动 Kernel $\to$ **GPU 计算** $\to$ CPU 卡住等待数据拷回。

现代 GPU 拥有独立的**计算引擎（Compute Engine）**和**双向数据拷贝引擎（Copy Engine）**，两者在硬件层面是可以完全并行工作的。

### 核心利器：CUDA Streams（任务队列）
- **CUDA Stream（流）**：GPU 任务的执行队列。同一个 Stream 内的任务严格按顺序执行；**不同的 Stream 之间的任务可以并发执行**。
- **cudaMemcpyAsync**：非阻塞的数据搬运指令，发出后 CPU 立即返回，配合 Stream 可以实现**“边搬运下批数据，边计算当前批数据”**。

### 经典吞吐翻倍模型：分块流水线（Pipeling / Overlap）
将 1GB 数据切成 4 份，分配到两个流中：
- **Stream 1**: 拷贝第 1 块 $\to$ 计算第 1 块 $\to$ 拷回第 1 块
- **Stream 2**: 
  - 当 Stream 1 在计算第 1 块时，Stream 2 已经在使用拷贝引擎**提前搬运第 2 块数据**。

> ⚠️ **前置硬件条件**：异步拷贝（`cudaMemcpyAsync`）要求 Host 端的内存必须是通过 `cudaMallocHost` 或 `cudaHostAlloc` 申请的**固定内存（Pinned/Page-locked Memory）**，普通的 `malloc` 内存无法直接实现真正的异步硬件 DMA 拷贝。
