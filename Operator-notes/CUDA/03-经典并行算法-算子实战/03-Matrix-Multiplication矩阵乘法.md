欢迎来到 CUDA 算子开发中真正的核心战场——**矩阵乘法（Matrix Multiplication，简称 GEMM）**。

在深度学习的大语言模型（Transformer 的 Linear、Attention 计算）以及传统科学计算中，**90% 以上的 GPU 算力消耗都集中在矩阵乘法上**。可以说，谁能把矩阵乘法写到极致，谁就能主导大模型的推理与训练速度。



# 基础概念

在写算子前，我们先把线性代数中两个矩阵相乘的规矩在大脑中彻底还原：

1. **矩阵乘法的维度约束：**
   - 矩阵相乘不是对应位置相乘，而是“前行乘以后列”。
   - 假设矩阵 $A$ 的大小为 $M \times K$（$M$ 行 $K$ 列），矩阵 $B$ 的大小为 $K \times N$（$K$ 行 $N$ 列）。
   - 只有当矩阵 $A$ 的**列数**等于矩阵 $B$ 的**行数**（均为 $K$）时，乘法才能成立。
   - 最终产生的目标矩阵 $C$ 的大小为 $M \times N$（$M$ 行 $N$ 列）。
2. **二维平展成一维的内存布局（行主序 Row-Major）：**
   - 在 C++/CUDA 显存中，大小为 $M \times K$ 的二维矩阵 $A$，在物理上是一根排成长条的一维地址线。
   - 矩阵 $A$ 中第 $r$ 行、第 $c$ 列的元素，在一维显存中的访问地址是：$r \times K + c$（跳过前面的行，加上当前列）。

# 公式与推导

## 单元素计算公式（点积公式）

对于结果乘积结果矩阵 $C$ 中的任意一个元素 $C_{row, col}$：

$$C_{row, col} = \sum_{k=0}^{K-1} A_{row, k} \times B_{k, col}$$

## 符号全翻译

- $C_{row, col}$：目标矩阵 $C$ 中第 $row$ 行、第 $col$ 列的元素。
- $A_{row, k}$：矩阵 $A$ 中第 $row$ 行、第 $k$ 列的输入元素。
- $B_{k, col}$：矩阵 $B$ 中第 $k$ 行、第 $col$ 列的输入元素。
- $K$：矩阵相乘的公共维度长度（即 $A$ 的总列数，或 $B$ 的总行数）。
- $\sum$：数学求和累加符号。
- $k$：累加循环变量，从 $0$ 遍历到 $K-1$。
- $\times$：标量乘法。
- $=$：数学定义与赋值符号。

## 推导

以计算 $C_{row, col}$ 为例，展开整个计算的动作：

$$sum = 0$$

(初始化局部累加器为 0)

$$term_k = A[row \times K + k] \times B[k \times N + col]$$

(从矩阵 A 的第 row 行取出第 k 个元素)

(从矩阵 B 的第 col 列取出第 k 个元素)

(将两者相乘，得到一个乘积项)

$$sum = sum + term_k$$

(将刚刚计算出的乘积项累加到临时总和中)

$$C[row \times N + col] = sum$$

(当 k 从 0 到 K-1 完整循环结束后)

(将最终结果写入显存矩阵 C 的绝对位置)



# 通用尺寸的矩阵乘法 (`gemm_naive.cu`)

为了完全支持真实的任意长宽矩阵计算（比如大语言模型中常见的 $M \neq N \neq K$ 形状），我们编写一个完全不限制宽高的完整程序。



请在 WSL 中创建 `gemm_naive.cu` 并写入以下代码：

```c++
#include <iostream>
#include <cuda_runtime.h>

// ==========================================
// 模块一：通用矩阵乘法算子 (Naive GEMM: C = A * B)
// A: M x K,  B: K x N,  C: M x N
// ==========================================
__global__ void gemmNaiveKernel(const float *A, const float *B, float *C, int M, int N, int K) {
    // 1. 计算当前线程负责的目标矩阵 C 的列坐标 col (X方向) 与 行坐标 row (Y方向)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. 边界检查：防止线程超出矩阵 C 的物理范围 (M 行 N 列)
    if (row < M && col < N) {
        float sum = 0.0f;

        // 3. 核心计算：点积内积循环 (前行乘以后列)
        for (int k = 0; k < K; ++k) {
            // A 的行主序索引: 当前行 * A的总宽度(K) + 当前列(k)
            float valA = A[row * K + k];
            // B 的行主序索引: 当前行(k) * B的总宽度(N) + 当前列(col)
            float valB = B[k * N + col];

            sum += valA * valB;
        }

        // 4. 将点积结果写入矩阵 C 的对应一维显存地址
        C[row * N + col] = sum;
    }
}

// ==========================================
// 模块二：CPU 主控与性能测试程序
// ==========================================
int main() {
    // 定义非正方形的真实形状尺寸:
    // M = 1024 (Batch或Token序列长), K = 512 (隐藏层输入维度), N = 2048 (隐藏层输出维度)
    int M = 1024;
    int K = 512;
    int N = 2048;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    std::cout << "正在初始化 GEMM 矩阵: A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")..." << std::endl;

    // 1. Host 内存分配与初始化
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;

    // 2. Device 显存分配与拷贝
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // 3. 配置二维 Block 与 Grid 维度
    // 采用标准的 16x16 线程块 (共 256 线程)
    dim3 threadsPerBlock(16, 16);
    // 根据目标矩阵 C 的尺寸 (宽度 N, 高度 M) 进行向上取整网格分配
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 4. 使用 CUDA 事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // 启动通用矩阵乘法算子
    gemmNaiveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);

    // 5. 传回结果
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // 6. 验证结果：每个元素是 K 个 (1.0 * 2.0) 的累加，期望值为 K * 2.0 = 512 * 2 = 1024.0
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "GEMM Kernel 耗时: " << elapsed << " ms" << std::endl;
    std::cout << "验证结果 C[0, 0] = " << h_C[0] << " (期望值: " << (float)(K * 2.0f) << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 7. 清理现场
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
```

​                              

## WSL 编译与运行流程

在 WSL 终端中依次输入：

```bash
nvcc gemm_naive.cu -o gemm_naive
./gemm_naive
```

**预期输出：**



```
正在初始化 GEMM 矩阵: A(1024x512) * B(512x2048) = C(1024x2048)...
----------------------------------------
GEMM Kernel 耗时: X.XX ms
验证结果 C[0, 0] = 1024 (期望值: 1024)
----------------------------------------
```





# 对矩阵乘法的优化

在实战中优化矩阵乘法，我们必须直面之前在 `gemm_naive.cu` 中遇到的最大性能杀手——**显存带宽瓶颈（Memory Bandwidth Bound）**。

1. **朴素版（Naive）的致命缺陷：**

   在 $C = A \times B$ 的计算中，为了算出 $C$ 中的 1 个元素，线程必须从全局内存中读取 $A$ 的一整行（$K$ 个数）和 $B$ 的一整列（$K$ 个数）。

   这就意味着，矩阵 $A$ 里的同一个数字，会被不同的线程**重复从慢速显存中读取 $N$ 次**！这种上亿次的重复读取会瞬间堵死硬件的数据通道。

2. **实战优化的核心手段 —— 分块与共享内存（Tiling & Shared Memory）：**

   我们将巨大的矩阵切割成一个个 $16 \times 16$ 的小方块（Tile）。

   每次计算时，线程块内的 256 个线程先**协同合作**，把 $A$ 和 $B$ 的一个小方块从“慢速显存”搬运到“高速共享内存工作台”上。

   然后，大家直接在工作台上完成乘加运算。由于工作台速度极快，这把**对全局内存的访问次数直接降低了 16 倍**。

## 边界填充与滑动索引推导

在真实的工业实战中，矩阵的尺寸（$M, N, K$）往往是不规则的（例如 $K=500$），未必能被块大小（如 $T_{dim}=16$）整除。

当单个 Block 沿着公共维度 $K$ 滑动处理点积时，最后一块很有可能会悬空并“超出边界”。为了防止非法显存访问且不影响乘加结果，必须使用边界补零（Zero Padding）技巧，即在超出边界的槽位处填入 $0.0f$。

### 核心数学公式

$$P_{total} = \frac{K + T_{dim} - 1}{T_{dim}}$$

$$Offset_{X} = p \times T_{dim} + t_x$$

$$Offset_{Y} = p \times T_{dim} + t_y$$

$$I_{A} = row \times K + Offset_{X}$$

$$I_{B} = Offset_{Y} \times N + col$$

### 符号全翻译

- $P_{total}$：单个 Block 沿着公共维度 $K$ 滑完当前行/列所需的**循环阶段总数（Phase 总数）**。

- $K$：矩阵乘法的公共维度长度（即矩阵 $A$ 的单行元素总数，亦即矩阵 $B$ 的单列元素总数）。

- $T_{dim}$：每个分块（Tile）的边长尺寸，在代码中对应 `TILE_WIDTH`（例如 16）。

- $p$：当前 Block 内部沿 $K$ 维度推进的阶段循环变量，取值范围为 $[0, P_{total} - 1]$。

- $t_x$：当前线程在块内的局部横向列坐标，对应代码 `threadIdx.x`。

- $t_y$：当前线程在块内的局部纵向行坐标，对应代码 `threadIdx.y`。

- $row$：当前线程负责计算的目标元素在矩阵 $C$ 中的全局行号，对应代码 `blockIdx.y * TILE_WIDTH + threadIdx.y`。

- $col$：当前线程负责计算的目标元素在矩阵 $C$ 中的全局列号，对应代码 `blockIdx.x * TILE_WIDTH + threadIdx.x`。

- $Offset_{X}$：矩阵 $A$ 当前元素相对于其所属行起始地址的**行内横向偏移量（即该元素的全局列坐标）**。

- $Offset_{Y}$：矩阵 $B$ 当前元素相对于其所属列起始地址的**列内纵向偏移量（即该元素的全局行坐标）**。

- $I_{A}$：当前线程搬运矩阵 $A$ 时，对应在显存线性一维空间中的**最终绝对物理地址**。

- $I_{B}$：当前线程搬运矩阵 $B$ 时，对应在显存线性一维空间中的**最终绝对物理地址**。

- $N$：目标矩阵 $C$ 及矩阵 $B$ 的总列宽（即每行的完整元素跨度）。

  

### 推导

#### 步骤一：推导阶段总数 $P_{total}$（向上取整）

$$P_{raw} = \frac{K}{T_{dim}}$$

(整数除法会丢弃余数，若直接采用会导致末尾不整除的部分被遗漏)

$$P_{total} = \frac{K + T_{dim} - 1}{T_{dim}}$$

(在被除数上预先加入除数减一的偏移量)

($\Rightarrow$ 借助整数除法的截断特性，实现对长度 $K$ 覆盖步数的严密向上取整)

#### 步骤二：推导矩阵 $A$ 的一维读取地址 $I_A$

$$Offset_{X} = p \times T_{dim} + t_x$$

(计算当前 Tile 在横向上跨过的完整块跨度)

(加上线程在当前小块内的局部列号)

($\Rightarrow$ 得到当前元素在矩阵 $A$ 行内的局部横向偏移量，即列坐标)

$$Base_A = row \times K$$

(计算当前行在显存中的起始基准地址)

(即：前面已经跨过的完整行数乘以每行跨度 $K$)

$$I_A = Base_A + Offset_{X}$$

(将行基准起始地址与行内横向偏移量相加)

($\Rightarrow I_A = row \times K + (p \times T_{dim} + t_x)$，获得直接可寻址的绝对线性下标)

#### 步骤三：推导矩阵 $B$ 的一维读取地址 $I_B$

$$Offset_{Y} = p \times T_{dim} + t_y$$

(计算当前 Tile 在纵向上跨过的完整块跨度)

(加上线程在当前小块内的局部行号)

($\Rightarrow$ 得到当前元素在矩阵 $B$ 列内的纵向偏移量，即行坐标)

$$Base_B = Offset_{Y} \times N$$

(计算该行在显存中的起始基准位置)

(即：纵向行号乘以每行跨度 $N$)

$$I_B = Base_B + col$$

(将行基准起始位置与全局目标列号相加)

($\Rightarrow I_B = (p \times T_{dim} + t_y) \times N + col$，获得直接可寻址的绝对线性下标)



### 类比： 铺设走廊地砖

想象我们要用一块 $16 \times 16$ 厘米的方形模板（Tile），去测量并加工一条极其细长的走廊（长度为 $K$）。

- **滑动阶段（循环 p）：** 我们拿着模板，从走廊的最左边开始，每画完一格，就向右移动 16 厘米，直到走廊尽头。这就叫“分块滑动”。
- **越界补零（Zero Padding）：** 假设走廊总长度是 50 厘米。我们移动到第 4 次时（覆盖 48~64 厘米），模板的右半边已经**悬空**超出了走廊。如果工人在悬空的地方强行钻孔（越界读取），机器就会报错。
- **解决方案：** 当发现当前位置超出了走廊长度时，工人就假装那里有一块**隐形的空心砖（值为 0.0）**。把 0.0 放进共享内存参与乘法运算，因为任何数乘以 0 还是 0，加到总和里完全不会改变最终结果！这就是**最优雅的安全防越界策略**。

# 实战优化级代码 (`gemm_tiled.cu`)

这是能够真正应对任意不规则 $M, N, K$ 尺寸的**工业级共享内存优化算子**。

请在 WSL 中新建 `gemm_tiled.cu` 并写入：

```c++
#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16 // 每一个小方块的尺寸为 16x16

// ==========================================
// 模块一：支持任意尺寸与边界补零的 Shared Memory GEMM
// A: M x K,  B: K x N,  C: M x N
// ==========================================
__global__ void gemmTiledKernel(const float *A, const float *B, float *C, int M, int N, int K) {
    // 1. 声明高速共享内存工作台
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    // 获取当前线程的绝对与局部坐标
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // 当前线程负责计算的目标矩阵 C 中的绝对坐标
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // 计算总共需要滑动多少个阶段 (向上取整)
    int numPhases = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // 2. 沿着 K 维度逐步滑动小方块
    for (int p = 0; p < numPhases; ++p) {
        
        // ---------------------------------------------------
        // 步骤 2.1: 协作搬运矩阵 A，并进行安全越界补零
        // ---------------------------------------------------
        // 计算当前要搬运的 A 元素的全局列坐标
        int global_A_col = p * TILE_WIDTH + tx; 
        // 检查：当前行不能超过 M，当前列不能超过 K
        if (row < M && global_A_col < K) {
            s_A[ty][tx] = A[row * K + global_A_col]; // 安全读取
        } else {
            s_A[ty][tx] = 0.0f; // 越界部分填 0，防止段错误，且不影响加法
        }

        // ---------------------------------------------------
        // 步骤 2.2: 协作搬运矩阵 B，并进行安全越界补零
        // ---------------------------------------------------
        // 计算当前要搬运的 B 元素的全局行坐标
        int global_B_row = p * TILE_WIDTH + ty;
        // 检查：当前行不能超过 K，当前列不能超过 N
        if (global_B_row < K && col < N) {
            s_B[ty][tx] = B[global_B_row * N + col]; // 安全读取
        } else {
            s_B[ty][tx] = 0.0f; // 越界补零
        }

        // 步骤 2.3: 必须等待当前 Block 所有线程都把数据安全搬运到共享内存
        __syncthreads();

        // ---------------------------------------------------
        // 步骤 2.4: 在高速共享内存中执行乘加计算
        // ---------------------------------------------------
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // 步骤 2.5: 必须等待所有线程计算完毕，才能进入下一次循环去覆盖共享内存
        __syncthreads();
    }

    // 3. 将最终累加的结果写回目标矩阵 C (检查边界，防止 C 越界写入)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==========================================
// 模块二：CPU 主控程序
// ==========================================
int main() {
    // 我们故意设置无法被 16 整除的奇葩尺寸，测试补零逻辑是否生效
    int M = 1000;
    int K = 500;
    int N = 2000;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    std::cout << "正在初始化不规则矩阵: A(" << M << "x" << K << ") * B(" << K << "x" << N << ")" << std::endl;

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                       (M + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    // 启动实战优化算子
    gemmTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Tiled GEMM 耗时: " << elapsed << " ms" << std::endl;
    // 因为每个元素都是 K 个 (1.0 * 2.0) 的累加，K=500，期望值为 1000.0
    std::cout << "验证结果 C[0, 0] = " << h_C[0] << " (期望值: " << (float)(K * 2.0f) << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
```

​            

*（在实战工业界，极致的优化还会结合寄存器分块 Register Tiling、向量化访存 float4 以及调用 cuBLAS / Tensor Core，但本代码是全部底层逻辑的基石所在。）*



**题目 1：边界补零的条件判定**

在代码中，搬运矩阵 $B$ 时我们写了 `if (global_B_row < K && col < N)`。

请问：

1. `global_B_row < K` 是为了防止小方块在沿着哪个方向滑动时超出矩阵 B 的物理内存边界？
   1. B的行数不允许超过边界
2. `col < N` 是为了防止处于 Grid 右侧边缘的线程越过矩阵 B 的哪一条物理边界？
   1. 不能超过B的总列宽

**题目 2：同步的灾难**

在代码中，我们在计算完 `sum += s_A[ty][k] * s_B[k][tx]` 后，紧接着写了第二个 `__syncthreads()`。

请设想一下，如果把这第二个 `__syncthreads()` 删掉，当外层循环进入下一个阶段（`p + 1` 轮）时，读取速度特别快的线程会怎样破坏共享内存 `s_A` 里的数据？这会导致速度慢的线程计算出什么错误结果？

若删除同步代码：

- 读取速度过快的线程会把新数据过早拉入共享内存，导致速度慢的线程取错数据。
