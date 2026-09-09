# 阶段 0：从“能运行”到“能编译”的 CUDA 环境入门

## 1. 先看结论

阶段 0 先记住两句话：

1. **PyTorch 能使用 GPU**：说明已经编译好的 PyTorch CUDA 程序可以通过 NVIDIA 驱动在 GPU 上运行。
2. **系统能编译 CUDA 代码**：说明本机有完整的 CUDA Toolkit、nvcc、头文件、开发库和 C++ 编译器。

它们有关联，但不是同一件事。

一个简单比喻：

- CUDA Runtime 像“播放器”：播放已经做好的视频。
- CUDA Toolkit 和 nvcc 像“剪辑工作室”：把源素材编译成新视频。
- NVIDIA 驱动像“显卡设备驱动”：让软件真正访问显卡。
- PyTorch 像“应用”：带来很多已编译的 CUDA 算子，但不一定带来完整开发工具。

所以：

~~~text
能播放已有 CUDA 程序 ≠ 能编译新的 CUDA 程序
~~~

## 2. 本阶段要学会什么

完成后，你应该能回答：

- Python 虚拟环境、PyTorch、CUDA Runtime、CUDA Toolkit、驱动分别是什么？
- nvcc、CMake、Ninja 分别处在哪一步？
- 为什么 torch.cuda.is_available() 为 True，仍可能没有 nvcc？
- 一段 Python/PyTorch 程序如何从源代码走到 GPU 输出？
- 为什么驱动显示 CUDA 12.0 时，CUDA 12.8 Runtime 不一定必然失败？

## 3. 八个名词，用白话解释

| 名词 | 白话解释 | 主要作用 |
| --- | --- | --- |
| Python 虚拟环境 | 项目自己的 Python 和依赖目录 | 防止不同项目的包互相污染；本项目是 .venv/ |
| PyTorch | Python 深度学习框架 | 提供 Tensor、自动求导、模型和 CPU/GPU 算子 |
| CUDA Runtime | 程序运行时使用的 CUDA 库 | 让已编译的程序申请显存、启动 kernel、传输数据 |
| CUDA Toolkit | CUDA 开发工具包 | 提供编译器、头文件、开发库和分析工具 |
| NVIDIA 驱动 | 操作系统与 GPU 的翻译员 | 让应用真正访问 NVIDIA 硬件 |
| nvcc | CUDA Toolkit 里的 CUDA 编译器 | 把 .cu 源代码编译成 GPU 代码 |
| CMake | 构建配置工具 | 根据 CMakeLists.txt 生成构建文件 |
| Ninja | 实际执行构建的工具 | 调用 g++、nvcc 并安排编译顺序 |

最关键的区别：

~~~text
Runtime：让已经编译好的 CUDA 程序运行
Toolkit：让你把新的 CUDA 源代码编译出来
~~~

## 4. CUDA 算子：从数学规则到 GPU 上真正发生的事

这一章依据你的 Host/Device、显存管理和 Grid/Block/Thread 笔记整理。目标是分清四个容易混淆的词：

学习素材： [Host 与 Device](../Operator-notes/CUDA/01-基础起步期/01-Host与Device.md)、[cudaMalloc 与 cudaMemcpy](../Operator-notes/CUDA/01-基础起步期/02-cudaMalloc与cudaMemcpy内存管理.md)、[Grid、Block、Thread](../Operator-notes/CUDA/01-基础起步期/03-线程索引Grid-Block-Thread.md)。这些笔记提供概念和示例；本章将其放入完整的编译与运行链路中解释。

| 名词 | 一句话定义 | 它关心的层次 |
| --- | --- | --- |
| 数学运算 | 例如 C[i] = A[i] + B[i] | 要算出什么结果 |
| 算子（operator） | 对输入 Tensor 施加确定规则的计算单元，带有输入/输出/shape/dtype/device 语义 | 框架和用户看到的功能 |
| Kernel（核函数） | 在 GPU 上由很多线程并行执行的一段具体程序 | 如何在 GPU 上计算 |
| PyTorch extension | 让 Python/PyTorch 能调用 C++/CUDA 实现的一层封装 | 如何把实现接入框架 |

最重要的关系是：

~~~text
一个算子可以有多个实现；
一个 CUDA kernel 是算子的某个 GPU 实现；
一个复杂算子也可以由多个 kernel 组成；
extension 是把这些实现暴露给 PyTorch 的包装，不等于算子本身。
~~~

例如“矩阵加法”这个算子可以有 CPU 版、PyTorch 内置 CUDA 版、你写的朴素 CUDA 版、使用向量化加载的优化 CUDA 版。它们的数学语义都应相同，但速度、显存使用和支持的输入条件可能不同。

### 4.1 算子究竟接收什么、产出什么

以向量加法为例：

~~~text
输入：
A：N 个 float32，位于 GPU 显存
B：N 个 float32，位于 GPU 显存

规则：
对每个合法下标 i，C[i] = A[i] + B[i]

输出：
C：N 个 float32，位于 GPU 显存
~~~

一个完整算子不只是那一行加法，还必须明确：

| 要素 | 为什么必须明确 |
| --- | --- |
| 数学语义 | 说明输入如何得到输出 |
| 输入数量与形状 | 决定要处理多少元素、怎样广播或对齐 |
| dtype | float32、float16、int32 的字节宽度、精度和指令不同 |
| device | 数据在 CPU RAM 还是 GPU VRAM，决定调用哪个后端 |
| 内存布局/stride | Tensor 的相邻逻辑元素是否在物理内存中连续 |
| 错误与边界 | N 不是 block 大小时也不能越界 |
| 性能约束 | 要减少多余拷贝、访存和同步 |
| 可微性（训练时） | 是否还要提供反向传播规则 |

**人类世界类比**

算子是一张“产品规格书”：规定原料 A、B 必须按什么规则变成成品 C。Kernel 是具体的生产工艺；同一张规格书可以有手工生产线、普通机器或高速自动化机器。

### 4.2 Host 与 Device：谁控制，谁计算

CUDA 把系统分成两侧：

~~~text
Host：CPU + 主机 RAM + 操作系统
Device：GPU + GPU 显存（VRAM）+ GPU 执行单元
~~~

Host 主要负责：

- 读取文件、创建进程、准备输入；
- 在 RAM 中构造或保存主机侧数据；
- 分配/释放 GPU 资源；
- 设置 Grid、Block 和 kernel 参数；
- 发射 kernel、检查错误、等待或继续做 CPU 工作；
- 把结果保存、打印或送到下一步。

Device 主要负责：

- 在 VRAM 中保存 GPU 侧 Tensor；
- 按线程并行读取输入数据；
- 执行算术、逻辑和访存指令；
- 将结果写回 VRAM。

**工厂类比**

Host 是项目经理和普通仓库，Device 是有大量相同机器的专用车间和车间仓库。项目经理决定生产什么、准备多少原料、何时发单；真正同时加工百万个零件的是车间。

Host 不会自动“看见” Device 指针指向的内容，GPU 也不能把普通 Host 指针当成普通 VRAM 地址随便解引用。两侧有各自的地址空间，数据移动必须由 CUDA Runtime/驱动安排。

### 4.3 指针在 Host 与 Device 两侧代表什么

在 C/C++ 中，指针是“内存位置的地址值”。但地址必须连同它所属的内存空间一起理解。

~~~text
float* h_A：Host 指针，指向 CPU RAM
float* d_A：Device 指针，指向 GPU VRAM
~~~

名称里的 h_ 和 d_ 只是人给的命名习惯；真正决定位置的是分配 API：

| 分配方式 | 返回的指针主要指向哪里 | 谁可以直接读写 |
| --- | --- | --- |
| malloc/new | Host RAM | CPU |
| cudaMalloc | Device VRAM | GPU kernel |
| cudaMallocManaged | 统一虚拟地址空间中的托管内存 | CPU/GPU，数据由系统迁移 |
| torch.empty(..., device="cuda") | PyTorch 管理的 Device VRAM | GPU kernel |

**计算机科学视角**

现代系统使用虚拟地址。d_A 不应被当作 CPU 可以直接解引用的普通数组；它是由 CUDA 上下文和驱动管理的 Device 地址。GPU kernel 收到 d_A 后，才会在自己的地址空间中把它当成数据位置。

**工厂类比**

h_A 是普通仓库货架编号，d_A 是 GPU 车间仓库货架编号。两个仓库都可能有“第 12 排第 3 格”，但位置不在同一栋楼，不能拿错地图。

### 4.4 cudaMalloc、cudaMemcpy 与 cudaFree：谁搬运数据

以标准分离内存模式为例：

~~~cpp
cudaMalloc(&d_A, bytes);
cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_A, ...);
cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
cudaFree(d_A);
~~~

每一步都在做不同的事：

| 调用 | 输入 | 软件动作 | 硬件/数据动作 | 输出 |
| --- | --- | --- | --- | --- |
| cudaMalloc | 需要的字节数 | Runtime 向驱动申请 Device 内存 | GPU 显存管理器预留一段 VRAM | d_A：Device 地址 |
| cudaMemcpy H2D | h_A、d_A、字节数、方向 | Runtime 建立复制请求 | copy engine/传输通路把字节从 RAM 运到 VRAM | d_A 中有 A 的副本 |
| kernel launch | kernel、参数、Grid、Block | Runtime/驱动提交执行命令 | GPU SM 读取 VRAM、执行指令、写结果 | d_C 中产生结果 |
| cudaMemcpy D2H | d_C、h_C、字节数、方向 | Runtime 建立返程复制请求 | 数据从 VRAM 运到 RAM | CPU 可读取 h_C |
| cudaFree | d_A | Runtime 释放资源 | VRAM 可被后续任务复用 | d_A 不再可用 |

这里有两个常见误解：

1. cudaMalloc 只“圈出”显存，不会自动把 h_A 的内容复制进去。
2. cudaMemcpy 只搬运字节，不理解“这是矩阵”还是“这是图片”。shape、dtype 和语义由你的代码或 PyTorch Tensor 元数据解释。

**统一内存的边界**

cudaMallocManaged 让 CPU/GPU 可以使用同一个指针变量，看起来省去了显式 H2D/D2H。底层仍要让数据在 RAM 与 VRAM 之间迁移；初次访问的 page fault 可能带来额外延迟。它适合教学、原型和某些特定场景，不等于“没有数据搬运”。

### 4.5 Kernel 是什么：一段程序，不是一颗 GPU 芯片

Kernel（核函数）是由 Host 发射、在 Device 上执行的函数。CUDA C++ 中通常写成：

~~~cpp
__global__ void vector_add(
    const float* A, const float* B, float* C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}
~~~

它不是“GPU 的核心硬件”，而是一段会被很多 GPU 线程同时执行的程序。GPU 的硬件核心是 SM、CUDA Core、Tensor Core 等；kernel 是这些硬件执行的指令集合。

函数修饰符的含义：

| 修饰符 | 函数在哪里执行 | 谁可以调用它 | 典型用途 |
| --- | --- | --- | --- |
| __global__ | GPU | CPU Host（通过 <<< >>>） | kernel 入口 |
| __device__ | GPU | GPU 内部其他 device/global 函数 | Device 辅助函数 |
| __host__ | CPU | CPU | 普通 C++ 函数，默认可省略 |
| __host__ __device__ | CPU 与 GPU | 两边 | 同时编出两份通用小函数 |

**工厂类比**

kernel 是一张发给每位车间工人的统一作业卡。每个人都读同一张卡，但凭自己的工号处理不同原料位置。

### 4.6 <<<Grid, Block>>>：不是函数参数，而是派工配置

下面这句包含两类信息：

~~~cpp
vector_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);
~~~

| 部分 | 是什么 | 作用 |
| --- | --- | --- |
| vector_add | kernel 名称 | 要执行哪段 GPU 程序 |
| <<<blocks_per_grid, threads_per_block>>> | execution configuration | 启动多少线程、怎样分组 |
| (d_A, d_B, d_C, N) | 普通函数参数 | 给每个线程的数据地址和元数据 |

一维下标公式：

~~~text
global_index = blockIdx.x * blockDim.x + threadIdx.x
~~~

例如 N = 1000、threads_per_block = 256：

~~~text
blocks_per_grid = ceil(1000 / 256) = 4
总启动线程 = 4 × 256 = 1024
最后 24 个线程没有真实元素可做，所以需要 if (i < N)。
~~~

二维矩阵则先计算全局行列坐标：

~~~text
col = blockIdx.x * blockDim.x + threadIdx.x
row = blockIdx.y * blockDim.y + threadIdx.y
linear_index = row * width + col
~~~

这是因为物理内存是一维地址序列；行主序矩阵要先跳过 row 个完整行，每行 width 个元素，再加 col。

**硬件如何处理**

Host 不会真的一次把 1024 个“人”塞进 GPU。Host 只提交 Grid 和 Block 描述。驱动把 block 放入可用 SM；每个 block 中的线程按 warp（通常 32 线程）分批调度。GPU 根据寄存器、共享内存、SM 容量等资源限制，决定同一时刻有多少 block 常驻。一个 block 的线程总数不能超过硬件上限；现代 GPU 常见上限是 1024，但程序应通过设备属性和目标架构确认。

### 4.7 一次向量加法的逐字节旅程

下面把“数据从哪里来、经过什么、最后输出什么”完整串起来：

~~~text
1. CPU 在 RAM 分配 h_A、h_B、h_C
2. CPU 写入 h_A/h_B 的 float 字节
3. cudaMalloc 在 VRAM 分配 d_A、d_B、d_C
4. H2D 复制：A/B 的字节从 RAM 到 VRAM
5. Host 发射 kernel；参数中保存 VRAM 地址、N、Grid/Block
6. GPU 的每个线程计算自己的 i
7. 每个合法线程从 d_A[i]、d_B[i] 读 float
8. ALU 做浮点加法，把结果写入 d_C[i]
9. D2H 复制：d_C 的字节从 VRAM 回到 h_C
10. CPU 读取 h_C，验证、打印或把结果交给下一程序
~~~

这里真正算 C[i] = A[i] + B[i] 的是第 8 步 GPU 运算单元；第 4、9 步只是搬运；第 5 步只是下达控制命令。

### 4.8 异步、同步与错误：为什么“发射成功”不代表“计算成功”

kernel launch 通常是异步的：

~~~text
CPU 提交 kernel → CUDA Runtime/驱动放入队列 → Python/C++ 控制流可继续
                                      ↓
                                  GPU 稍后执行
~~~

所以需要区分三件事：

| 检查 | 能发现什么 |
| --- | --- |
| cudaGetLastError() | launch 配置等立即可发现的问题，例如非法 Grid/Block |
| cudaDeviceSynchronize() | 等待队列完成，并暴露 kernel 执行中发生的非法访存等错误 |
| cudaMemcpy DeviceToHost | 通常也会形成需要等待结果的同步点，随后 CPU 才能读取结果 |

CUDA API 往往通过返回错误码报告问题。生产代码应检查 cudaMalloc、cudaMemcpy、kernel launch 和同步结果。否则“终端没有崩溃”并不保证得到正确 Tensor。

## 5. 从源码到 .so：每个构建产物到底是什么

现在解释小白最容易看到、但很少被逐个解释的链条：

~~~text
CUDA/C++ 源码
  ↓
预处理与编译
  ↓
目标文件 .o
  ↓
链接
  ↓
可执行文件 或 .so 动态库
  ↓
Python import / 操作系统动态加载器
  ↓
运行
~~~

### 5.1 源文件、头文件与编译单元

| 文件 | 定义 | 作用 |
| --- | --- | --- |
| .cu | CUDA C++ 源文件 | 可同时包含 Host C++ 和 Device kernel |
| .cpp/.cc | C++ 源文件 | 常放 Python binding、CPU 实现或通用逻辑 |
| .h/.hpp | 头文件 | 放声明、类型、模板、宏，让多个源文件共享接口 |
| 编译单元 | 一个源文件加上其 include 展开后的结果 | 编译器一次独立处理的输入 |

预处理器会先处理 #include、#define 和条件编译，把相关文本拼成编译单元。此时仍是“给编译器看的程序文本”，不是最终机器码。

### 5.2 g++：宿主 C++ 编译器

g++ 是 GNU C++ 编译器的命令行驱动。它把 CPU 侧 C++ 源码变成 CPU 能执行的机器码。

它负责的典型内容：

- main 函数和 Host 控制流程；
- 调用 CUDA Runtime API 的 C++ 代码；
- PyTorch C++ binding；
- 普通 CPU 算法；
- 模板实例化、类型检查、CPU 优化。

它主要用 CPU、RAM 和 SSD 工作：CPU 执行编译器本身，RAM 保存语法树/中间表示，SSD 读源码并写 .o。g++ 不是 GPU 编译器，也不把 Tensor 数值拿去计算。

### 5.3 nvcc 与 Device 代码：PTX、SASS、CUBIN、FATBIN

nvcc 是 NVIDIA CUDA Compiler Driver，属于 CUDA Toolkit；这里的 driver 指“编译器调度器”，不是显卡驱动。它把同一个 .cu 文件拆为 Host C++ 和 Device CUDA 两条路径：

~~~text
Host C++  → 交给 g++ 等宿主编译器
Device CUDA → 生成 PTX 和/或目标 GPU 的原生代码
~~~

#### 四个产物，只记各自负责的一层

| 名词 | 全称/性质 | 它装的是什么 | 用途 |
| --- | --- | --- | --- |
| PTX | Parallel Thread Execution；虚拟 GPU ISA | 与具体 GPU 型号无关的中间指令 | 由驱动在运行时 JIT（just in time 运行时编译） 为原生代码，提供后备兼容性 |
| SASS | NVIDIA 常称 SASS assembly；原生 GPU 指令 | 面向一个 SM 架构的机器指令 | GPU 直接执行 |
| CUBIN | CUDA Binary；单一目标架构的设备二进制 | 一个或多个 kernel 的 SASS、参数和元数据 | 供匹配架构的 GPU 直接加载 |
| FATBIN | Fat Binary；多架构容器 | 多个 CUBIN 和/或 PTX | 让同一程序适配多代 GPU |

关键区别：

~~~text
SASS 是“指令本身”
CUBIN 是“装有某一架构 SASS 的二进制文件”
FATBIN 是“装有多个架构 CUBIN/PTX 的容器”
~~~

#### 完整架构图：分支编译，再统一打包

~~~text
                         编译阶段

vector_add.cu（Host C++ + 一个 Device kernel）
                         │
                         ▼
                       nvcc
          ┌──────────────┴───────────────────┐
          │                                  │
          ▼                                  ▼
  Host C++ → g++ → CPU 目标代码      同一个 Device kernel
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
              sm_75 CUBIN               sm_89 CUBIN           sm_120 CUBIN
               （Turing）                 （Ada）              （Blackwell）
                    │                        │                        │
                    └────────────────────────┴────────────────────────┘
                                             │
                         构建时可选包含：compute_120 PTX（JIT 后备）
                                             │
                                             ▼
                    FATBIN = 多个 CUBIN 和/或 PTX 的容器
                                             │
                                             ▼
                     嵌入 .o，再链接进 .so 或可执行文件

                         运行阶段

           驱动读取当前 GPU 的 Compute Capability
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   找到匹配 CUBIN                没有匹配 CUBIN
   例如 sm_120                  但有兼容 PTX
            │                         │
            ▼                         ▼
      直接执行 SASS          JIT：PTX → 当前 GPU 的 SASS
                                      │
                                      ▼
                                  缓存后执行
~~~

FATBIN 中的多份代码实现的是同一个 kernel 语义，只是分别针对不同架构优化；它不是“整代 Blackwell/Rubin 的全部指令集合”。这里的“构建时可选包含 PTX”表示开发者决定是否把 PTX 放进发布产物：它不是与 CUBIN 同时执行的第二份程序，而是在没有匹配 CUBIN 时提供给驱动 JIT 的后备表示。

#### 你的 GPU：Compute Capability、compute_120 与 sm_120

当前 RTX 5080 Laptop：

~~~text
硬件架构：Blackwell
Compute Capability：12.0
虚拟编译目标：compute_120 → PTX
真实编译目标：sm_120 → SASS/CUBIN
~~~

Compute Capability 是硬件能力版本；sm_120 是 nvcc 为该硬件目标生成原生代码的标签。两块不同型号 GPU 即使都支持 12.0，SM 数量、显存和频率也可以不同。

可用下面的显式配置同时保留原生代码和 PTX 后备：

~~~bash
nvcc vector_add.cu -o vector_add \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120
~~~

第一条 gencode 生成 sm_120 的 CUBIN/SASS；第二条保留 compute_120 PTX。运行时驱动优先选择匹配的 CUBIN，只有没有匹配 CUBIN 且 PTX 兼容时才 JIT。

若要单独查看一个 CUBIN，可生成并反汇编：

~~~bash
nvcc -arch=sm_120 --cubin vector_add.cu -o vector_add_sm120.cubin
nvdisasm vector_add_sm120.cubin
~~~

### 5.4 .o：还不能独立运行的目标文件

.o 是 object file，中文常称目标文件。它已经包含某一个编译单元的机器码、符号表和重定位信息，但通常仍缺少其他文件提供的函数/变量定义，因此不能独立运行。

例子：

~~~text
binding.cpp → binding.o
kernel.cu  → kernel.o（包含 Host 代码和嵌入的 Device 代码）
~~~

**工厂类比**

.o 是已经加工好的零件，不是完整设备。它知道“我需要调用某个函数”，但还没确定这个函数最终在哪个库里。

### 5.5 链接器：把符号与零件接成完整产物

链接器的核心任务是：

1. 收集多个 .o 与库；
2. 解决符号引用：例如 binding.o 调用的 kernel wrapper 或 PyTorch 函数在哪里；
3. 调整地址和重定位；
4. 生成最终可执行文件或共享库。

它不是简单“把文件拼接起来”。它要保证每个被调用的函数、变量都有正确来源，并建立运行时可用的地址关系。

**工厂类比**

链接器是总装与接口检验部门：不仅把零件放进同一箱子，还要确认插头、螺丝、接口型号全部匹配。找不到符号就是“缺零件”；重复符号就是“两个零件都声称自己占同一个接口”。

### 5.6 .so 动态库：可以被进程在运行时装载的二进制组件

.so 是 Linux 的 shared object，共享对象/动态库。它是 ==ELF 二进制格式==的一种产物，不是文本文件，也通常不是你在终端直接执行的主程序。

> ELF 的全称是 Executable and Linkable Format，中文常译为“可执行与可链接格式”。
>
> 它是 Linux/Unix 系统中保存二进制程序的一种标准文件格式。以下文件通常都是 ELF：
>
> ```
> ./vector_add              # 可执行程序
> kernel.o                  # 目标文件
> my_extension.so           # 动态库
> 某些 .cubin               # CUDA 设备二进制
> ```
>
> 可以把 ELF 想成“二进制程序的标准包装箱”。箱子里不只装机器指令，还会贴上详细标签，告诉系统：
>
> ```
> ELF 文件
> ├── 文件头：这是哪种架构、什么类型的文件
> ├── .text：机器指令
> ├── .data：已初始化的全局变量
> ├── .bss：未初始化的全局变量
> ├── 符号表：函数和变量叫什么、在哪里
> ├── 重定位信息：链接时哪些地址需要修正
> └── 动态依赖：运行时还需要加载哪些 .so
> ```
>
> 在 CUDA 场景中，一个 CUBIN 常使用 ELF 结构来保存：
>
> ```
> CUBIN
> ├── kernel 的 SASS 指令
> ├── kernel 参数与常量
> ├── 寄存器、共享内存等元数据
> └── 符号与链接信息
> ```
>
> 你可以查看 Linux 文件是否为 ELF：
>
> ```
> file 某个文件
> ```
>
> 查看 ELF 头和段：
>
> ```
> readelf -h 某个文件
> readelf -S 某个文件
> ```
>
> 例如：
>
> ```
> file /bin/ls
> ```
>
> 通常会看到类似：
>
> ```
> ELF 64-bit LSB pie executable, x86-64
> ```
>
> 意思是：这是一个 64 位、小端序、x86-64 架构的 Linux ELF 可执行文件。

对 PyTorch extension 来说：

~~~text
Python 代码 import my_extension
             ↓
操作系统动态加载器把 my_extension.so 映射进 Python 进程
             ↓
解析它依赖的 Python、PyTorch、CUDA 等共享库
             ↓
Python binding 把导出的 C++/CUDA 函数注册为可调用 API
             ↓
Python 调用 my_extension.vector_add(...)
~~~

为什么叫“动态”：

- 不是编译阶段把全部依赖复制到一个大文件里；
- 进程启动或 import 时才加载；
- 多个进程可共享同一份只读代码页，节省 RAM；
- 依赖库版本、ABI、路径不匹配时，也可能在 import 阶段失败。

.so 不等于 wheel。wheel 是 Python 包分发格式，里面可以装 Python 文件、元数据和一个或多个 .so；pip 安装 wheel 后，Python 才能 import 其中的 .so。

### 5.7 CMake、Ninja、nvcc、g++、链接器之间的精确关系

~~~text
你写：CMakeLists.txt 或 setup.py/pyproject.toml
          ↓
CMake：生成构建规则，例如 build.ninja
          ↓
Ninja：根据依赖图决定要执行哪些命令
          ↓
nvcc/g++：实际把源码编译为 .o
          ↓
链接器：把 .o 和库变成 .so 或可执行文件
          ↓
pip/安装流程：把产物放到 Python 环境
          ↓
Python import：动态加载 .so
~~~

可以用一句话记忆：

| 工具 | 它的“性质” | 它亲手产出的主要东西 |
| --- | --- | --- |
| CMake | 构建系统生成器/配置工具 | Ninja 或 Make 的构建规则 |
| Ninja | 构建执行器/调度器 | 启动编译和链接命令 |
| nvcc | CUDA 编译器驱动 | Device 代码与相应目标文件 |
| g++ | C++ 编译器驱动 | Host CPU 目标文件 |
| 链接器 | 二进制装配器 | .so 或可执行文件 |
| pip | Python 包安装前端 | site-packages 中可 import 的包 |
| 动态加载器 | 操作系统运行时装载器 | 进程内可调用的共享库映射 |

### 5.8 可执行文件与 .so 的区别

| 对比项 | 可执行文件 | .so 动态库 |
| --- | --- | --- |
| 典型启动方式 | ./vector_add | Python import 或被其他程序加载 |
| 是否通常有 main 函数 | 是 | 不需要 |
| 主要用途 | 独立程序 | 被别的进程复用的功能模块 |
| 本项目的学习场景 | 直接 nvcc vector_add.cu -o vector_add | PyTorch C++/CUDA extension |
| 输出如何出现 | 程序自己 printf/std::cout | Python 调用接口后返回 Tensor |

初学阶段先用独立可执行文件学习 cudaMalloc、cudaMemcpy 和 kernel launch；随后再学习 .so extension，把相同 kernel 接入 PyTorch。两者共享底层 CUDA 思想，差别是入口和封装方式。


## 6. 追踪一次 vector_add.cu：从源文件到输出数值

这一节只跟踪一个具体程序。每一步都说明：它处理什么数据、由谁执行、数据去了哪里。

数学规则：

~~~text
对每个 0 ≤ i < N：
C[i] = A[i] + B[i]
~~~

假设本次运行的输入是：

~~~text
N = 8，dtype = float32

A = [1, 2, 3, 4, 5, 6, 7, 8]
B = [10, 20, 30, 40, 50, 60, 70, 80]
预期 C = [11, 22, 33, 44, 55, 66, 77, 88]

float32 每个元素 4 字节；A、B、C 各占 8 × 4 = 32 字节。
~~~

先区分三种数据：

| 数据类别 | 例子 | 谁处理它 |
| --- | --- | --- |
| 源码/构建数据 | vector_add.cu、.o、CUBIN、FATBIN | nvcc、g++、链接器、Ninja |
| 控制数据 | N、Device 地址、Grid/Block、stream、kernel 参数 | Host、CUDA Runtime、驱动 |
| 数值数据 | A/B/C 的 float32 字节 | RAM、传输通路、VRAM、GPU SM |

### 6.1 源文件：一份 .cu 写了两种程序

~~~cpp
__global__ void vector_add(
    const float* A, const float* B, float* C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = A[i] + B[i];
}

int main() {
  // Host：分配内存、拷贝数据、启动 kernel、验证结果
}
~~~

| 代码部分 | 谁执行 | 作用 | 工厂类比 |
| --- | --- | --- | --- |
| Host C++ | CPU | 准备数据、组织任务、检查结果 | 项目经理 |
| Device kernel | GPU 线程 | 对不同 i 并行做 A[i]+B[i] | 大量执行同一工序的工人 |

编辑器和文件系统此时只把源码字符写入 SSD；还没有 A/B 数值，更没有 GPU 计算。

### 6.2 编译：把源码变成可启动程序

示例发布命令：

~~~bash
nvcc vector_add.cu -o vector_add \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120
~~~

编译阶段处理源码和二进制产物，不处理本次运行中的 A/B 数值。

~~~text
vector_add.cu
      │
      ▼
nvcc：识别 Host C++ 与 Device CUDA
      │
      ├── Host 路径：g++ 编译为 CPU 目标代码
      │
      └── Device 路径：同一个 vector_add kernel 生成
             ├── sm_75 CUBIN：Turing 的 SASS
             ├── sm_89 CUBIN：Ada 的 SASS
             ├── sm_120 CUBIN：Blackwell 的 SASS
             └── compute_120 PTX：JIT 后备
                      │
                      ▼
              FATBIN：装入多个 Device 代码版本
                      │
                      ▼
        .o：Host 目标代码 + 嵌入 FATBIN
                      │
                      ▼
链接器：.o + CUDA Runtime 等库 → vector_add ELF 可执行文件
~~~

| 工具/产物 | 此时做什么 | 输入 → 输出 | 不做什么 |
| --- | --- | --- | --- |
| nvcc | 协调 Host 与 Device 两条编译路径 | .cu → Host 编译输入、PTX/CUBIN | 不执行向量加法 |
| g++ | 编译 Host 控制逻辑 | Host C++ → CPU 机器码/.o | 不生成 GPU SASS |
| CUBIN | 某一架构的 GPU 二进制 | kernel 的 SASS + 元数据 | 不是整代 GPU 的全部指令 |
| PTX | 虚拟 GPU 指令 | kernel 的 JIT 后备表示 | 不是最终原生机器码 |
| FATBIN | 多架构容器 | CUBIN/PTX → 一个容器 | 不选择当前运行 GPU |
| 链接器 | 总装并解析符号 | .o + 库 → ELF | 不启动 kernel |
| CMake/Ninja（若用） | 生成规则/调度命令 | 构建描述 → 调用 nvcc、g++、链接器 | 不处理 Tensor 数值 |

工厂类比：nvcc/g++ 是制造技师，CUBIN 是某型号 GPU 专用零件，FATBIN 是多型号零件箱，链接器是总装部门。

### 6.3 启动程序：驱动选择本机可执行的 Device 代码

执行：

~~~bash
./vector_add
~~~

~~~text
shell
  ↓
操作系统创建进程，加载 ELF 和 CUDA Runtime
  ↓
main() 在 CPU 开始执行
  ↓
Runtime 初始化 CUDA 上下文
  ↓
NVIDIA 驱动检查 GPU、权限和 Compute Capability
  ↓
从 FATBIN 选择可执行的 Device 代码
~~~

当前机器是 RTX 5080 Laptop，Compute Capability 为 12.0。驱动的选择逻辑是：

~~~text
FATBIN
  ├── 有 sm_120 CUBIN
  │     └── 直接加载其中 SASS；通常不需要 JIT
  │
  └── 没有 sm_120 CUBIN
        ├── 有驱动判定兼容的 PTX
        │     └── JIT：PTX → 本机 GPU 的 SASS → 缓存 → 执行
        └── 没有兼容 PTX
              └── 报错：没有可供该 GPU 执行的 Device 代码
~~~

一次运行不会把 sm_75、sm_89、sm_120 全部送去 GPU；只会选择当前硬件能执行的一份。

### 6.4 Host 在 RAM 准备数据，并在 VRAM 预订货架

Host 侧概念代码：

~~~cpp
size_t bytes = N * sizeof(float);  // 32 bytes

float* h_A = ...;  // Host RAM
float* h_B = ...;
float* h_C = ...;

float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, bytes);
cudaMalloc(&d_B, bytes);
cudaMalloc(&d_C, bytes);
~~~

此刻：

~~~text
Host RAM                           Device VRAM
─────────                          ───────────
h_A → [1,2,3,4,5,6,7,8]            d_A → 已分配，内容未初始化
h_B → [10,20,30,40,50,60,70,80]    d_B → 已分配，内容未初始化
h_C → [?, ?, ?, ?, ?, ?, ?, ?]     d_C → 已分配，内容未初始化
~~~

cudaMalloc 的软件角色是 Runtime 向驱动申请显存；硬件结果是 VRAM 中预留 32 字节。它不会自动复制 h_A 的内容。

工厂类比：RAM 是办公室仓库，VRAM 是 GPU 车间仓库；cudaMalloc 只是预订车间货架。

### 6.5 H2D：输入字节从 RAM 搬到 VRAM

~~~cpp
cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
~~~

~~~text
h_A 的 32 字节 ──H2D──→ d_A 的 32 字节
h_B 的 32 字节 ──H2D──→ d_B 的 32 字节
~~~

CUDA Runtime 建立复制请求；驱动安排传输通路和 GPU copy engine。它们只搬运字节，不理解“向量”概念；向量长度 N、dtype 和计算语义由程序解释。

### 6.6 发射 kernel：Host 提交控制命令

~~~cpp
int threads_per_block = 256;
int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
// N = 8，所以 blocks_per_grid = 1

vector_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);
cudaGetLastError();
~~~

~~~text
Grid：1 个 Block
Block：256 个线程
启动线程：256
有效线程：0 到 7
越界线程：8 到 255；if (i < N) 阻止它们读写
~~~

Host 提交的是控制数据：

~~~text
kernel 地址、d_A/d_B/d_C 的 Device 地址、N=8、Grid=1、Block=256、stream
~~~

驱动把 Block 安排到可用 SM；GPU 将 256 个线程按 warp（通常 32 线程）调度。Host 不会创建 256 个 CPU 线程。

### 6.7 Device 真正完成 A+B：每个合法线程一个元素

kernel 的下标规则：

~~~cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) C[i] = A[i] + B[i];
~~~

本例中 blockIdx.x 为 0：

| GPU 线程 | threadIdx.x | i | 读取 | 计算 | 写入 |
| --- | ---: | ---:| --- | --- | --- |
| thread 0 | 0 | 0 | 1、10 | 1 + 10 = 11 | d_C[0] = 11 |
| thread 1 | 1 | 1 | 2、20 | 2 + 20 = 22 | d_C[1] = 22 |
| thread 2 | 2 | 2 | 3、30 | 3 + 30 = 33 | d_C[2] = 33 |
| ... | ... | ... | ... | ... | ... |
| thread 7 | 7 | 7 | 8、80 | 8 + 80 = 88 | d_C[7] = 88 |
| thread 8–255 | 8–255 | 8–255 | 不读取 | 边界判断失败 | 不写入 |

单个合法线程的数据路径：

~~~text
GPU VRAM 中 d_A[i]、d_B[i]
        ↓ load/store 单元与缓存
SM 寄存器
        ↓ 浮点运算单元
A[i] + B[i]
        ↓ store
GPU VRAM 中 d_C[i]
~~~

真正执行浮点加法的是 SM 中的计算单元；CMake、Ninja、Runtime、驱动只负责构建、调度、资源和命令，不替代这一步。

### 6.8 D2H、同步、验证：结果变成人能看到的输出

kernel launch 通常异步。Host 发射后，GPU 可能仍在计算；Host 需要结果时必须等待并复制：

~~~cpp
cudaDeviceSynchronize();
cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

printf("%f\n", h_C[0]);  // 11.0
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
~~~

~~~text
Device VRAM                          Host RAM
───────────                          ─────────
d_C → [11,22,33,44,55,66,77,88] ─D2H→ h_C → [11,22,33,44,55,66,77,88]
                                               ↓
                                    printf / 文件 / 下一算子
~~~

| 动作 | 软件层作用 | 硬件层发生什么 |
| --- | --- | --- |
| cudaDeviceSynchronize | 等待任务并暴露执行错误 | CPU 等待 GPU 完成 |
| cudaMemcpy D2H | 请求复制输出 | 32 字节从 VRAM 经传输通路到 RAM |
| printf | 把 CPU float 格式化为字符 | CPU 产生文本，终端显示 |
| cudaFree | 归还 Device 分配 | VRAM 可被后续任务复用 |

### 6.9 一句话复盘

~~~text
编译时：源码变成“CPU 控制代码 + 多架构 GPU 代码”的 ELF 程序。
运行时：驱动为 RTX 5080 Laptop 选 sm_120 CUBIN；CPU 把 A/B 从 RAM 搬到 VRAM；
GPU 线程各自计算一个 C[i]；C 再从 VRAM 回到 RAM，最后由 CPU 打印。
~~~

若改用 PyTorch extension，VRAM、kernel、驱动和 SM 的底层过程不变；变化只是入口从 ./vector_add 变为 Python import 和 PyTorch Tensor API。

## 7. 对照：PyTorch 内置向量加法的运行路径

第 6 节是自己编译并运行 `vector_add.cu`；本节是调用 PyTorch 已有的加法算子。底层的 Runtime、驱动、VRAM、SM 和数据搬运逻辑仍然存在，但 `nvcc`、CMake、Ninja 已在 PyTorch 发布 wheel 时完成工作，通常不会在本机运行时再次出现。

要计算：

~~~text
输入 x = [1, 2, 3]
计算 y[i] = x[i] + 1
输出 y = [2, 3, 4]
~~~

Python 代码可能是：

~~~python
import torch

x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
y = x + 1.0
print(y.cpu())
~~~

### 第一步：Python 读取源代码

执行：

~~~bash
python example.py
~~~

Python 解释器读取代码，但它自己不负责执行 GPU 加法。它调用虚拟环境中的 PyTorch。

~~~text
example.py
    ↓
Python 解释器
    ↓
.venv 中的 PyTorch
~~~

### 第二步：PyTorch 选择已有算子

PyTorch 根据 Tensor 的 device、dtype、shape 和内存布局，选择对应的已编译算子。这里通常不是现场编译，而是加载 PyTorch 已经准备好的二进制代码。

### 第三步：初始化 GPU

~~~text
PyTorch
  ↓
CUDA Runtime
  ↓
NVIDIA 驱动
  ↓
GPU
~~~

驱动检查 GPU 是否存在、进程是否有权限、版本是否兼容。失败时，PyTorch 就不能使用 CUDA。

### 第四步：数据进入 GPU 显存

~~~text
CPU 内存中的 x = [1, 2, 3]
          ↓ 拷贝
GPU 显存中的 x = [1, 2, 3]
~~~

CPU 和 GPU 通常有不同的内存空间，数据需要通过 Runtime 和驱动传输。

### 第五步：GPU 执行 kernel

~~~text
线程 0：y[0] = x[0] + 1
线程 1：y[1] = x[1] + 1
线程 2：y[2] = x[2] + 1
~~~

这些线程按 grid → block → thread 组织。实际工作由 GPU kernel 完成。

### 第六步：结果返回 CPU

~~~text
GPU 显存 y = [2, 3, 4]
          ↓ y.cpu()
CPU 内存 y = [2, 3, 4]
          ↓
print 输出
~~~

CUDA 操作可能是异步提交的。y.cpu() 通常会等待计算完成，再把结果拷回 CPU。

### 运行阶段总图

~~~text
Python 源代码
    ↓
Python 解释器
    ↓
PyTorch API
    ↓
PyTorch 已编译算子
    ↓
CUDA Runtime
    ↓
NVIDIA 驱动
    ↓
GPU kernel
    ↓
GPU 显存中的结果
    ↓ .cpu()
CPU 内存
    ↓
print / 文件 / 下一步计算
~~~

## 8. 自定义 CUDA 算子的编译和运行数据流

如果你自己写 my_kernel.cu，会多出一个“编译阶段”。

### 编译阶段

~~~text
my_kernel.cu + binding.cpp
          ↓
setup.py 或 CMake
          ↓
Ninja / 构建流程
          ↓
nvcc 编译 GPU 部分
g++ 编译 C++ binding
          ↓
链接 PyTorch、Python、CUDA 开发库
          ↓
生成 my_extension.so
~~~

.so 是 Python 可以加载的二进制扩展。

### 运行阶段

~~~text
Python import my_extension
          ↓
加载 .so
          ↓
C++ binding 检查 Tensor
          ↓
CUDA Runtime 启动 kernel
          ↓
NVIDIA 驱动访问 GPU
          ↓
GPU 执行 kernel
          ↓
返回 PyTorch Tensor
~~~

因此自定义算子同时需要两套条件：

| 阶段 | 需要的东西 |
| --- | --- |
| 编译 | CUDA Toolkit、nvcc、头文件、开发库、g++、CMake/Ninja |
| 运行 | PyTorch、CUDA Runtime、兼容驱动、目标 GPU |

## 9. 当前机器的实测结果

执行：

~~~bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
./scripts/check_environment.sh --purpose=stage-0
~~~

关键输出：

~~~text
torch: 2.11.0+cu128
torch CUDA runtime: 12.8
cuda available: True
GPU 0: NVIDIA GeForce RTX 5080 Laptop GPU
compute capability: 12.0
memory: 15.92 GiB
driver_version: 572.84
nvcc: CUDA 12.8, V12.8.93
CUDA_HOME: /usr/local/cuda-12.8
gcc: 13.3.0
cmake: 3.28.3
ninja: 1.11.1
~~~

白话解释：

- PyTorch 已经能调用 GPU；
- 驱动能访问 RTX 5080；
- 本机还安装了 CUDA Toolkit 和 nvcc；
- 当前环境既能运行，也具备编译自定义 CUDA 的基础。

## 10. 第一题：只安装 PyTorch，能运行但未必能编译

### 题目

纯净 Ubuntu 中只执行：

~~~bash
pip install torch
python -c "import torch; print(torch.cuda.is_available())"
~~~

假设官方 CUDA wheel、NVIDIA 驱动和 GPU 兼容，且系统没有完整 CUDA Toolkit。

### 答案

大概率输出：

~~~text
True
~~~

因为官方预编译 wheel 已带有运行 PyTorch CUDA 算子所需的 Runtime 组件。运行已有二进制代码不要求你现场使用 nvcc。

但是包含自定义 .cu 的：

~~~bash
python setup.py install
~~~

大概率在构建阶段失败，常见位置是：

1. 找不到 CUDA_HOME；
2. 找不到 nvcc；
3. 找不到 cuda_runtime.h；
4. 找不到 CUDA 开发库；
5. 编译或链接失败。

原因是：

~~~text
运行 PyTorch：
已有二进制 + CUDA Runtime + NVIDIA 驱动

编译自定义 CUDA：
源代码 + CUDA Toolkit + nvcc + 头文件 + 开发库 + C++ 编译器
~~~

当前机器不能直接复现“没有 Toolkit”的报错，因为本机已经有 /usr/local/cuda-12.8/bin/nvcc。但它证明了 torch.cuda.is_available() 和 nvcc 是两个独立检查点。

## 11. 第二题：驱动显示 CUDA 12.0，Runtime 是 12.8

### 先给结论

不能简单说“必然是 False”。

更准确的答案是：

~~~text
如果实际驱动版本满足 CUDA 12.x 的兼容要求：
    CUDA 12.8 Runtime 可能仍可运行。

如果驱动确实太旧，或程序使用了驱动不支持的新特性：
    GPU 初始化或第一次执行 CUDA 算子可能失败。
~~~

### 为什么不能只看 12.0 和 12.8

nvidia-smi 显示的 CUDA Version 通常表示驱动支持的 CUDA 版本上限，不是当前 PyTorch 正在使用的 Runtime 版本。

CUDA 11 以后存在同一主版本内的 minor-version compatibility。NVIDIA 文档给出的 CUDA 12.x Linux 驱动门槛是至少 525；CUDA 12.0 Release Notes 给出的具体最低版本是 525.60.13。

所以：

- 如果驱动实际版本达到门槛，CUDA 12.8 不一定失败；
- 如果驱动低于门槛，初始化或执行可能失败；
- 即使能运行，某些新特性也可能受限。

### 失败时可能看到什么

可能出现：

- torch.cuda.is_available() 返回 False；
- warning 提示 NVIDIA driver 太旧；
- 第一次执行 CUDA 算子时出现 CUDA initialization error；
- 出现 driver version insufficient 或 unsupported PTX 等错误。

### 版本检查背后的数据流

~~~text
PyTorch wheel 中的 CUDA Runtime 12.8
          ↓ 调用
宿主机 NVIDIA 驱动 API
          ↓ 检查驱动、GPU 架构和所需能力
成功：初始化 GPU，启动 kernel
失败：返回错误，PyTorch 报告不可用或算子执行失败
~~~

当前机器的实测是：

~~~text
driver_version: 572.84
torch CUDA runtime: 12.8
torch.cuda.is_available(): True
~~~

这只能证明当前驱动和当前 Runtime 兼容，不能证明所有旧驱动都兼容。

## 12. 一张表总结两个问题

| 你要回答的问题 | 主要检查什么 | 检查命令 |
| --- | --- | --- |
| PyTorch 能否使用 GPU？ | Runtime、驱动、GPU、PyTorch | torch.cuda.is_available() |
| 系统能否编译 CUDA？ | Toolkit、nvcc、头文件、开发库、编译器 | nvcc --version |
| 编译出的扩展能否运行？ | 运行时兼容性、ABI、目标 GPU | import extension |
| 结果是否正确？ | reference 和误差阈值 | pytest/数值比较 |
| 是否真的变快？ | warmup、同步、重复采样 | benchmark/profiler |

最重要的记忆：

~~~text
torch.cuda.is_available() 主要回答：能不能运行已有 CUDA 程序？
nvcc --version 主要回答：有没有 CUDA 编译器？
~~~

## 13. 阶段 0 验收

运行：

~~~bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
command -v nvcc
nvcc --version
cmake --version
ninja --version
./scripts/check_environment.sh --purpose=stage-0
~~~

然后用自己的话回答：

1. 为什么只安装 PyTorch CUDA wheel 可能可以运行 GPU，却不能编译 .cu？
2. nvcc 属于哪个组件？
3. 为什么 PyTorch CUDA 可用，仍可能找不到 nvcc？
4. 驱动、CUDA Runtime、CUDA Toolkit 各自负责什么？
5. 输入数据如何从 CPU 到 GPU，再从 GPU 回到 CPU？
6. 为什么 CUDA 12.0 驱动提示不一定意味着 CUDA 12.8 必然不能运行？

如果你能回答这些问题，阶段 0 就完成了。

## 14. 官方参考资料

- NVIDIA CUDA Compatibility: Minor Version Compatibility
  https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- NVIDIA CUDA 12.0 Release Notes
  https://docs.nvidia.com/cuda/archive/12.0.0/cuda-toolkit-release-notes/index.html
