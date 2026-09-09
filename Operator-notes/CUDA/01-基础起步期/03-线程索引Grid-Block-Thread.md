



# 旧知识与基础概念复习



**1. 内存的一维本质（数组展平）**

在计算机的物理内存中，不存在真正的“二维表格”或“三维空间”。所有的内存地址都是排成一条直线的（一维数组）。

如果我们有一个 $3 \times 4$（3 行 4 列）的二维矩阵，计算机在存储时，会采用“行主序（Row-Major）”的方式，也就是先把第 0 行的所有元素放完，接着放第 1 行，再放第 2 行。



**2. 二维坐标映射到一维地址的经典公式**

假设我们有一个宽度为 $W$ 的二维矩阵。我们要找到其中坐标为 $(x, y)$ 的元素（即第 $y$ 行，第 $x$ 列），它在内存这条直线上的绝对位置 $I$ 是多少？

核心法则是：**你前面有几行，就要跳过几个完整的宽度，再加上你在当前行的列偏移。**



# 数学公式与多维索引推导

在 CUDA 中，NVIDIA 提供了一个三层架构来组织线程：

- **Grid（网格）**：最外层容器。
- **Block（线程块）**：包含在 Grid 中。
- **Thread（线程）**：包含在 Block 中。

我们最常用的是**二维结构**（用来处理图像或矩阵）。CUDA 自动为我们提供了内置变量，我们需要利用这些变量推导出一个线程的**全局二维坐标 $(I_x, I_y)$**，进而推导出它对应处理的数据在**一维内存中的绝对索引 $I_{global}$**。



## 线程绝对索引推导公式

$$I_x = B_x \times D_x + T_x$$

$$I_y = B_y \times D_y + T_y$$

$$I_{global} = I_y \times W + I_x$$

## 符号全翻译（绝对不省略）

- $I_x$：当前线程在整个 Grid 中的全局 X 坐标（列坐标）。
- $I_y$：当前线程在整个 Grid 中的全局 Y 坐标（行坐标）。
- $I_{global}$：当前线程负责处理的数据在内存中的一维绝对位置（索引）。
- $B_x$：当前线程所在的 Block 在 X 方向（横向）的编号，对应代码 `blockIdx.x`。
- $B_y$：当前线程所在的 Block 在 Y 方向（纵向）的编号，对应代码 `blockIdx.y`。
- $D_x$：每个 Block 在 X 方向包含的线程总数（宽度），对应代码 `blockDim.x`。
- $D_y$：每个 Block 在 Y 方向包含的线程总数（高度），对应代码 `blockDim.y`。
- $T_x$：当前线程在自己所属 Block 内部的 X 方向局部坐标，对应代码 `threadIdx.x`。
- $T_y$：当前线程在自己所属 Block 内部的 Y 方向局部坐标，对应代码 `threadIdx.y`。
- $W$：我们要处理的数据矩阵的总宽度（一共有多少列）。

## 详细推导过程

为了确定一个线程的全局身份和它要处理的数据位置，我们分三步推导：

**第一步：计算全局 X 坐标（列）**

$$Base_x = B_x \times D_x$$

(计算当前块前面有多少个完整的线程宽度)

(即：块的横向编号乘以每个块的横向人数)

$$I_x = Base_x + T_x$$

(在前面积累的总线程数偏移基础上，加上当前线程在块内的横向局部编号)



**第二步：计算全局 Y 坐标（行）**

$$Base_y = B_y \times D_y$$

(计算当前块前面有多少个完整的块高度)

(即：块的纵向编号乘以每个块的纵向人数)

$$I_y = Base_y + T_y$$

(在前面积累的总行数偏移基础上，加上当前线程在块内的纵向局部编号)



**第三步：将二维坐标展平为一维内存地址**

$$Offset_y = I_y \times W$$

(计算当前线程前面完整跨过了多少行数据)

(即：全局 Y 坐标乘以矩阵的总宽度 $W$)

$$I_{global} = Offset_y + I_x$$

(在跨过完整行数的基准上)

(加上当前线程在当前行内的 X 偏移)



## 三维索引推导（3D 体积数据与 3D 卷积场景）

在处理 3D 卷积、医学 CT 体积数据或流体力学网格时，数据由三维张量 $(D, H, W)$ 即深度、高度、宽度组成。CUDA 原生支持 3D 维度的 `dim3`（包含 $x, y, z$ 分量）。

- **D (Depth - 深度)**：通常指数据的“层数”。在医学 CT 中，它代表扫描切片（Slice）的总数；在时空序列中，可能代表时间帧数；对应三维坐标系中的 **Z 轴**。

- **H (Height - 高度)**：指单层切片或截面的高度（垂直方向的像素或网格数）；对应三维坐标系中的 **Y 轴**。

- **W (Width - 宽度)**：指单层切片或截面的宽度（水平方向的像素或网格数）；对应三维坐标系中的 **X 轴**。

### 三维空间坐标计算公式
$$I_x = B_x \times D_x + T_x$$
$$I_y = B_y \times D_y + T_y$$
$$I_z = B_z \times D_z + T_z$$

- $I_x, I_y, I_z$：当前线程在 3D 空间中的全局长、宽、高坐标。
- 对应代码：
  - `int x = blockIdx.x * blockDim.x + threadIdx.x;`
  - `int y = blockIdx.y * blockDim.y + threadIdx.y;`
  - `int z = blockIdx.z * blockDim.z + threadIdx.z;`

### 三维坐标展平为一维显存地址公式（行优先/切片优先）
我们要定位坐标为 $(x, y, z)$ 的体素（Voxel）在一维线性显存中的绝对地址：
$$I_{3D\_global} = I_z \times (W \times H) + I_y \times W + I_x$$

- **推导等号说明书**：
  - $I_z \times (W \times H)$：当前体素之前已经完整跳过了 $I_z$ 个完整二维切片（Slice），每个切片包含 $W \times H$ 个数据。
  - $+ I_y \times W$：在当前切片内，完整跳过了前面的 $I_y$ 行数据。
  - $+ I_x$：加上当前行内的列偏移。



# 直观例子 —— 城市、街区与门牌号

请把你调用的这个算子（Kernel）想象成一个**庞大的城市（Grid）**：

- **Grid（城市）：** 这个城市被划分为规则的网格状街区。
- **Block（街区 $B_x, B_y$）：** 比如 $(B_x=2, B_y=1)$ 代表“东三区，北二区”（注意编号从 0 开始）。每个街区的长宽面积是统一规定的（$D_x, D_y$）。
- **Thread（房屋 $T_x, T_y$）：** 街区内部的某栋房子。
- **计算 $I_x, I_y$ 的意义：** 如果你想给整个城市的所有房子发一个全局唯一的二维坐标（比如“全市第 15 大道，第 8 街”），你就必须先算出现在处于哪一个街区（跳过前面的街区），再加上房子在街区内部的相对位置。

![jLLlRz9067_FftZ2te0oqkjB4in4ZJ7ndIIwE6CwUWEoqYxbbj4Oaqt7X0qTSrbOtDIHw4OsGboY3i5bNuPhsRVm-iTIivinHlkY77r-p-_FFLUOA1999_C59LJCI0e651T3eU9iNfsJTAa0Sr9cLihYUyHWDHrdYAHc5Hm0o7X6cbT86fV8TIMVLGjO9P1XDwo7GbutX6_hUHaYvjRdGVUGldz5](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/jLLlRz9067_FftZ2te0oqkjB4in4ZJ7ndIIwE6CwUWEoqYxbbj4Oaqt7X0qTSrbOtDIHw4OsGboY3i5bNuPhsRVm-iTIivinHlkY77r-p-_FFLUOA1999_C59LJCI0e651T3eU9iNfsJTAa0Sr9cLihYUyHWDHrdYAHc5Hm0o7X6cbT86fV8TIMVLGjO9P1XDwo7GbutX6_hUHaYvjRdGVUGldz5.svg)



# 二维线程索引实战

下面是一个完整的、注释详尽的二维矩阵加法程序。请把重点放在 `matrixAddOperator` 内部的索引计算，以及 `main` 函数中 `dim3` 类型的资源分配上。

```c++
#include <iostream>
#include <cuda_runtime.h>

// ==========================================
// 模块一：GPU 算子函数 (Kernel Function)
// ==========================================
__global__ void matrixAddOperator(const float *A, const float *B, float *C, int width, int height) {
    // 1. 计算当前线程的全局 X 坐标 (列索引 I_x)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 计算当前线程的全局 Y 坐标 (行索引 I_y)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 3. 边界检查：防止线程坐标超出了真实矩阵的边界
    if (col < width && row < height) {
        // 4. 将二维坐标 (col, row) 展平为一维内存数组的绝对索引 I_global
        // 公式：当前行号 * 矩阵宽度 + 当前列号
        int index = row * width + col;
        
        // 5. 让当前线程负责计算属于它的那个元素
        C[index] = A[index] + B[index];
    }
}

// ==========================================
// 模块二：CPU 主控程序 (Host Program)
// ==========================================
int main() {
    // 1. 定义矩阵的几何尺寸
    int width = 1000;  // 矩阵的宽度 (总列数)
    int height = 1000; // 矩阵的高度 (总行数)
    int numElements = width * height; // 矩阵总元素个数 (100万个元素)
    size_t size = numElements * sizeof(float); // 总共需要的字节数

    std::cout << "正在初始化大小为 " << width << "x" << height << " 的矩阵..." << std::endl;

    // 2. 在 CPU (Host) 上分配内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 3. 在 CPU 端初始化输入数据
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0f; // 矩阵 A 所有元素设为 1.0
        h_B[i] = 2.0f; // 矩阵 B 所有元素设为 2.0
        h_C[i] = 0.0f; // 结果矩阵 C 初始化为 0.0
    }

    // 4. 在 GPU (Device) 上分配显存
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 5. 把数据从 CPU 内存搬运到 GPU 显存 (Host to Device)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 6. 配置二维的 Block 和 Grid 维度
    // 定义每个 Block 内部的工人布局：16 行 x 16 列 = 256 个线程
    dim3 threadsPerBlock(16, 16); 

    // 计算整个 Grid 需要多少个 Block 才能覆盖整个矩阵 (向上取整)
    int blocksX = (width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksY = (height + threadsPerBlock.y - 1) / threadsPerBlock.y;
    
    // 组装成二维的 Grid 布局
    dim3 blocksPerGrid(blocksX, blocksY);

    std::cout << "正在启动 GPU 算子进行并行计算..." << std::endl;

    // 7. 启动算子，传入配置好的多维调度参数和数据指针
    matrixAddOperator<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);

    // 8. 把计算结果从 GPU 显存搬运回 CPU 内存 (Device to Host)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 9. 验证计算结果并打印
    std::cout << "验证结果 [0, 0] 位置: C[0] = " << h_C[0] << " (期望值: 3.0)" << std::endl;
    std::cout << "验证结果 [末尾] 位置: C[" << numElements - 1 << "] = " << h_C[numElements - 1] << " (期望值: 3.0)" << std::endl;

    // 10. 释放显存和内存，养成好习惯
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "程序执行完毕，资源已全部释放！" << std::endl;

    return 0;
}
```

请打开你的 WSL 终端，切换到你保存该文件的目录，严格按顺序执行以下指令：

## 步骤 1：编译源码

输入以下编译命令并回车：

Bash

```
nvcc matrix_add.cu -o matrix_add
```

- **符号解释：**
  - `nvcc`：调用 NVIDIA 官方 CUDA 编译器。
  - `matrix_add.cu`：你刚刚写好的源代码文件。
  - `-o matrix_add`：指定编译后的可执行程序名字叫 `matrix_add`。
- **如何验证成功：** 终端没有任何报错，并且在当前目录下多出一个白绿相间的可执行文件 `matrix_add`。

## 步骤 2：运行程序

输入以下运行命令并回车：

```bash
./matrix_add
```

- **符号解释：**

  - `./`：代表当前目录。
  - `matrix_add`：刚才生成的目标程序。

- **预期终端输出：**

  Plaintext

  ```
  正在初始化大小为 1000x1000 的矩阵...
  正在启动 GPU 算子进行并行计算...
  验证结果 [0, 0] 位置: C[0] = 3 (期望值: 3.0)
  验证结果 [末尾] 位置: C[999999] = 3 (期望值: 3.0)
  程序执行完毕，资源已全部释放！
  ```

## 工业红线：Block 与 Grid 的硬件物理限制

在配置 `<<<blocksPerGrid, threadsPerBlock>>>` 时，并非参数越大越好，硬件有严格的物理边界。若超出边界，Kernel 会直接发射失败（返回 `cudaErrorInvalidConfiguration`）：

| 硬件参数维度             | 物理上限 (Compute Capability $\ge$ 3.0，现代 GPU 通用) | 常见选择与工程建议                                           |
| :----------------------- | :----------------------------------------------------- | :----------------------------------------------------------- |
| **单 Block 线程总数**    | **最大 1024** ($D_x \times D_y \times D_z \le 1024$)   | **切忌超标**。工业界通常设为 128, 256 或 512（必须是 32 的整数倍） |
| **Block 内部维度限制**   | $x \le 1024,\; y \le 1024,\; z \le 64$                 | 三维 Block 布局时，注意 $z$ 轴最大只能到 64                  |
| **Grid 的 X 维度上限**   | **$2^{31} - 1$** (约 21.4 亿，几乎无限制)              | 一维数组或超大展平计算主要拉大 Grid 的 X 轴                  |
| **Grid 的 Y/Z 维度上限** | **65535** ($2^{16} - 1$)                               | **高危陷阱**：若处理高分辨二维图像，`blocksY` 一旦超过 65535 会直接静默报错 |

> **核心结论**：`dim3 threadsPerBlock(16, 16)` 共有 $16 \times 16 = 256$ 个线程，小于 1024，合法；但若设为 `dim3(32, 32)`（1024 线程）虽在边缘，但若设为 `dim3(33, 32)`（1056 线程）将直接崩溃。





# 并行的因素

一个初学者的困惑：**“为什么从代码上看，我写的好像只是一个普通的加法，也没有写 `for` 循环去遍历 100 万个数据，但它偏偏就能瞬间并行跑完？”**



## 基础概念

复习计算机科学中的两种核心执行模式：

1. **串行执行（Serial Execution）：**
   - 特点：一条指令接着一条指令执行。就像流水线上的一个工人，做完第 1 个零件，再做第 2 个零件。
   - 谁在做：CPU。CPU 虽然核心少（比如只有 8 核或 16 核），但每个核心极其聪明，擅长处理复杂的逻辑控制。
2. **并行执行（Parallel Execution）：**
   - 特点：成千上万个任务在**同一个绝对时间点**同时发生。就像 100 万个工人每人手里拿一个零件，同时开工。
   - 谁在做：GPU。GPU 的核心极多（几千个甚至上万个），虽然单个核心不如 CPU 聪明，但架不住人多势众。

## 哪些是串行，哪些是并行？

在我们上一个完整的 `matrix_add.cu` 程序中，执行过程被严格切分成了两部分：



### 纯粹的【串行操作】（由 CPU 独占完成）

以下代码行在运行时刻，绝对是一条接一条线性执行的，不存在任何硬件层面的同时发生：

- `malloc` / `free`（在 CPU 内存里申请和释放空间）。
- `cudaMalloc` / `cudaFree`（CPU 通过总线向 GPU 发送指令，让 GPU 划分显存）。
- `cudaMemcpy`（数据搬运：无论是由 CPU 发往 GPU，还是从 GPU 拉回 CPU，底层总线传输都是按指令步骤依次进行的）。
- `main()` 函数里的打印语句 (`std::cout`)。

### 纯粹的【并行操作】（由 GPU 的百万线程同时执行）

- 当代码执行到 `matrixAddOperator<<<blocksPerGrid, threadsPerBlock>>>(...)` 这一行时：
  - CPU 发出启动指令后，GPU 的硬件调度器瞬间唤醒了 **1,000,000 个线程**（因为 $1000 \times 1000 = 1,000,000$）。
  - 这 100 万个线程在硬件上**同时**执行算子内部的代码。
  - 每一个线程只干一件极其微小的事情：算出自己的 `col` 和 `row`，找到对应的 `index`，然后把 $A[index] + B[index]$ 算出来赋给 $C[index]$。这 100 万次加法运算是**完全在同一物理瞬间并行发生的**。

### 为什么从代码上看，完全看不出并行？

这是一个极佳的观察。传统 CPU 编程（比如 OpenMP 或多线程）如果要并行处理 100 万个数据，你通常需要在代码里写一个 `for` 循环，然后显式地把循环切成几块分给不同的线程。

但在 CUDA 中，你为什么看不到 `for` 循环？

1. **SPMD 编程模型（单程序多数据）：**

    `matrixAddOperator` 代码，并不是“指挥全局”的宏观代码，而是“单个线程的视角（POV）”。

   - 换句话说：**你只需要编写“全中国 14 亿人里，某一个普通中国人该怎么过日子”的代码。**
   - 你不需要在代码里写“张三去上班，李四去上学”，你只需要写一句：“如果一个人成年了，他就去上班。”
   - CUDA 运行时会自动把这套逻辑复制 100 万份，让 100 万个线程同时去套用这套逻辑。

2. **用“工号”代替“循环”：**

   在传统的串行代码中，我们用 `for (int i = 0; i < 1000000; ++i)` 来让一个工人干 100 万次活。

   而在 CUDA 中，我们取消了 `for` 循环，而是派出了 100 万个工人。每个工人通过公式：

   $$index = row \times width + col$$

   各自算出自己的**绝对工号（唯一身份）**。工号为 0 的工人只算第 0 个数据，工号为 999999 的工人只算最后 1 个数据。大家各司其职，瞬间并发完成。

### CUDA 是怎样在底层实现并行的？

既然代码里只写了一个线程的逻辑，GPU 硬件是怎么把它变成百万级并行的呢？

我们可以通过一个公式来理解 GPU 硬件的调度机制：

$$C_{total} = SM_{count} \times Core_{per\_SM}$$

#### 符号全翻译

- $C_{total}$：GPU 内部物理拥有的并行计算核心总数。
- $SM_{count}$：GPU 内部流多处理器（Streaming Multiprocessor）的数量（可以理解为工厂里的大车间数量）。
- $Core_{per\_SM}$：每个车间内部包含的 CUDA 核心（计算工人）数量。
- $\times$：乘法运算。
- $=$：赋值符号。

####  硬件执行原理

当你在代码中写下 `<<<blocksPerGrid, threadsPerBlock>>>` 时，GPU 的硬件调度器（Hardware Scheduler）会发生以下物理动作：

当这 100 万个线程的“登记表”压到 GPU 上时，GPU 内部的硬件调度器会进行以下物理操作：

![image-20260906101619055](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/image-20260906101619055.png)

$$\text{Step 1: 分配任务到大车间 (SM)}$$

(GPU把 100 万个线程按 Block 为单位，塞进几十个名为 SM 的流多处理器车间里排队)

$$\text{Step 2: 组装成线程束 (Warp)}$$

(在车间内部，硬件把 32 个线程打包成一个“Warp（线程束）”，这是 GPU 硬件调度的最小物理单位)

$$\text{Step 3: 极速硬件轮转 (Zero-Overhead Scheduling)}$$

(假设一个物理核心同时分配了 64 个 Warp 任务。当第 1 个 Warp 正在等待从显存读取数据（几百个时钟周期的延迟）时，物理核心**瞬间**切换到第 2 个 Warp 开始计算，中间没有任何软件开销！)

本质上我们是使用有限的空间去创造更多的时间富裕，实际上我们还是在做==线程切换==来假装的并行。但GPU 的“线程切换”和 CPU 的“线程切换”，虽然名字都叫切换，但它们的物理代价有着天壤之别。GPU 并不是靠变出无限的物理核心来硬抗百万线程，而是靠海量的轻量级虚拟线程，加上零开销的硬件切换，把显存等待的漫长延迟给“藏”了起来。



# 工业级进阶模式：网格跨步循环 (Grid-stride Loop)

在前文的 SPMD 范式中，我们遵循“1 个线程只计算 1 个数据点”，靠 `if (idx < N)` 来防止越界。
但在真实生产环境中，如果数据量达数千万甚至上亿，盲目无限制地启动成千上万个 Block 会带来巨大的硬件调度负担，甚至撞上 Grid 维度上限。

工业界标准模式是引入 **Grid-stride Loop（网格跨步循环）**：让固定数量的常驻线程池，以“网格总线程数”为步长，循环把整批数据吃完。

## 经典模板代码

```cpp
__global__ void vectorAddGridStride(const float *A, const float *B, float *C, int N) {
    // 1. 当前线程的全局绝对工号（起点）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 整个 Grid 一次能调动的工人总规模（步长）
    int stride = blockDim.x * gridDim.x;

    // 3. 跨步循环：每个工人处理第 idx 个数据，然后跳过 stride 距离处理下一个
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
```

## 直观类比：发牌模型

- **前文的单元素模式**：一副有 100 张牌（$N=100$），你找来 100 个人，每人手里拿一张牌就算完。
- **Grid-stride 模式**：你只组织了一桌 4 个人的固定牌局（`stride = 4`）。
  - 0号工人拿第 0、4、8、12... 张牌；
  - 1号工人拿第 1、5、9、13... 张牌；
  - 依此类推，直到发完全部 100 张牌。

##  为什么这是 NVIDIA 推荐的 Best Practice？

1. **解耦硬件调度与数据规模**：无论数据量 $N$ 是 1000 还是 10 亿，你都可以把 Grid 大小固定在能打满 GPU SM 的最优参数（如固定 128 个 Block，每个 Block 256 线程），无需每次动态重算 Grid 尺寸。
2. **天然支持内存合并访问（Coalesced Access）**：在网格跨步循环中，每个线程循环的起点是自己计算出的全局绝对工号 `idx`。相邻的两个线程（比如线程 0 和线程 1），它们的初始 `idx` 本身就是连续的整数 0 和 1。每次循环结束时，所有线程都会同时向前跳过一个完全相同的固定距离，这个距离就是 `stride`（整个 Grid 一次能调动的工人总规模）。在任意第 `k` 次循环中，相邻的这两个线程访问的数组索引分别是 `k * stride + 0` 和 `k * stride + 1`。无论循环进行到哪一步，这两个目标地址相减的差值永远是绝对的 1。即相邻线程在同一步循环里访问的显存地址依然是连续的（`idx, idx+1, idx+2...`），保证显存总线带宽利用率拉满。
3. **便于本地调试**：只要在 Host 端把线程配置写成 `<<<1, 1>>>`，该 Kernel 就会瞬间无缝退化为标准单线程 CPU 风格循环，极易单步断点 debug。
