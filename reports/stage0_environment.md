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

### 5.3 nvcc：CUDA 编译器驱动，不只是“编译一个 .cu”

nvcc 是 NVIDIA CUDA Compiler Driver，属于 CUDA Toolkit。它的“driver”在这里是编译器驱动程序，不是显卡驱动。

一个 .cu 文件同时含有 Host 和 Device 代码。nvcc 的关键性质是协调两条编译路径：

~~~text
.cu
 ├─ Host C++ 部分 → 交给宿主 C++ 编译器（例如 g++）
 └─ Device CUDA 部分 → CUDA 编译链 → PTX 和/或目标 GPU 机器码
~~~

Device 侧可能得到：

| 产物 | 定义 | 何时有用 |
| --- | --- | --- |
| PTX | NVIDIA 的虚拟 GPU 指令表示 | 可由驱动在目标 GPU 上 JIT 成实际机器码，利于前向兼容 |
| SASS/cubin | 面向特定 GPU 架构的实际机器码 | 可直接在匹配架构 GPU 上执行 |
| fatbin | 可容纳多个架构代码/PTX 的容器 | 一个二进制支持多种 GPU 架构 |

把 nvcc 想成“双语项目总编”：它识别哪些语句给 CPU，哪些语句给 GPU；CPU 部分交给 g++，GPU 部分生成 NVIDIA GPU 可执行代码，然后把两边结果装配起来。

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

.so 是 Linux 的 shared object，共享对象/动态库。它是 ELF 二进制格式的一种产物，不是文本文件，也通常不是你在终端直接执行的主程序。

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


## 6. 每个环节究竟做什么：计算机科学视角与工厂视角

### 6.1 先分清三种“数据”

“工具如何处理数据”中的数据，不只指 Tensor 数值。整个系统里有三类完全不同的数据：

| 数据类别 | 例子 | 谁处理它 |
| --- | --- | --- |
| 构建数据 | .py/.cpp/.cu 源码、头文件、编译参数、目标架构、.o、.so | CMake、Ninja、nvcc、g++、链接器 |
| 控制数据 | 调用哪个算子、kernel 参数、显存地址、grid/block 大小、stream 顺序 | Python、PyTorch dispatcher、CUDA Runtime、驱动 |
| 业务数据 | 输入 Tensor、权重、中间结果、输出 Tensor | CPU、RAM、PCIe/共享内存通道、GPU 显存、SM |

CMake 和 Ninja 处理的是构建数据，不会读取 Tensor 中的 1、2、3。CUDA Runtime 和驱动主要处理控制数据，同时安排业务数据的分配与搬运。GPU 的计算单元才真正读取 Tensor 数值并执行加法或乘法。

### 6.2 一条统一的人类世界类比

把 GPU 程序想成一家工厂：

| 计算机组件 | 工厂中的角色 |
| --- | --- |
| 你写的 Python/CUDA 源码 | 产品说明和生产工艺设计稿 |
| Python 虚拟环境 | 这家工厂专用的工具柜，避免和别的工厂混用工具 |
| CMake | 工程规划员，把设计稿整理成施工/生产计划 |
| Ninja | 现场调度员，按依赖和顺序安排每一道构建工序 |
| nvcc 和 g++ | 制造机器的技师，把人能读的源码变成机器能执行的二进制 |
| 链接器 | 总装工，把多个零部件和库装成一个 .so 成品 |
| Python 解释器 | 前台业务员，读取用户命令并调用后端能力 |
| PyTorch | 生产管理系统，知道不同 Tensor 应该调用哪台机器、哪道工序 |
| CUDA Runtime | 车间主管，申请仓库空间、安排搬运、提交生产任务 |
| NVIDIA 驱动 | 工厂与具体机器之间的设备控制层，把通用请求变成硬件可执行命令 |
| CPU 与 RAM | 办公区和普通仓库，负责控制流程并保存主机侧数据 |
| PCIe/共享内存通路 | 办公区与 GPU 车间之间的运输通道 |
| GPU 显存 | GPU 车间旁的专用原料仓库 |
| GPU 的 SM | 并行生产车间，大量线程在这里执行同一种工序 |
| 终端 print | 出货窗口，把结果转换成人能看到的文字 |

接下来对每一环都用同样的问题解释：

1. 它收到什么？
2. 它在软件层做什么？
3. 它使用什么硬件？
4. 它输出什么？
5. 它不负责什么？

### 6.3 源代码与编辑器：写下“想做什么”

**计算机科学视角**

源代码是存放在文件系统中的字节。Python 文件描述高级控制逻辑，C++ 文件描述宿主机代码，CUDA 的 .cu 文件还包含 GPU kernel。

保存文件时：

~~~text
键盘输入
  ↓
编辑器在进程内存中维护文本
  ↓
操作系统文件系统
  ↓
SSD/磁盘保存 UTF-8 字节
~~~

**工厂类比**

你是产品设计师，源代码是设计稿。设计稿说明产品怎么做，但纸上的设计稿不会自己开动机器。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | 人键入的字符 |
| 软件手段 | VS Code/编辑器、文件系统 API |
| 硬件手段 | CPU 处理编辑操作，RAM 保存编辑缓冲区，SSD 持久化文件 |
| 输出 | .py、.cpp、.cu、CMakeLists.txt 等源文件 |
| 不负责 | 不执行 Tensor 计算，不启动 GPU |

### 6.4 Python 虚拟环境：决定“使用哪套 Python 工具”

**计算机科学视角**

虚拟环境本质上是一个目录和一组路径规则。激活 .venv 后，shell 修改 PATH，使 python 和 pip 优先指向项目目录中的可执行文件；Python 再从该环境的 site-packages 查找 PyTorch。

它不是虚拟机，也不模拟一台新电脑。

**工厂类比**

它是本项目上锁的工具柜。A 工厂的扳手不会和 B 工厂不同型号的扳手混在一起。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | shell 命令、PATH、已安装 Python 包 |
| 软件手段 | shell 环境变量、Python import 搜索路径 |
| 硬件手段 | CPU 查找路径，SSD 读取解释器和包文件，RAM 装载代码 |
| 输出 | 确定本次运行使用哪个 Python、PyTorch 和依赖版本 |
| 不负责 | 不安装 NVIDIA 驱动，不执行 GPU kernel，不代替 CUDA Toolkit |

### 补充环节 A：pip、pyproject.toml 与 setup.py——启动安装或构建

**计算机科学视角**

pip 是 Python 包安装前端。它读取 pyproject.toml，确定需要哪个 build backend 和哪些构建依赖，然后调用后端完成 wheel 或 editable install。老式项目可能直接运行 setup.py；PyTorch extension 的构建逻辑再去调用 CMake/Ninja 或 torch.utils.cpp_extension。

pip 本身不是 C++ 或 CUDA 编译器。它负责启动、组织和安装：

~~~text
pip install
  ↓ 读取 pyproject.toml
选择 build backend
  ↓
后端生成编译任务
  ↓
CMake/Ninja 或直接调用 nvcc/g++
  ↓
生成 wheel/.so
  ↓
复制或链接到 site-packages
~~~

**工厂类比**

pip 是采购与安装负责人：根据项目清单寻找所需工具，向工程部门下单，最后把制成品登记到本项目的工具柜。setup.py 或 build backend 是具体承办这张订单的项目经理。

| 项目 | 内容 |
| --- | --- |
| 输入 | 包名、源码目录、pyproject.toml/setup.py、安装参数 |
| 软件手段 | 依赖解析、构建隔离、调用 build backend、安装 wheel |
| 硬件手段 | CPU 执行安装逻辑；网络下载包；SSD 保存包、缓存和构建产物 |
| 输出 | site-packages 中的 Python 包、.so、包元数据 |
| 不负责 | 不自己生成 GPU 指令；没有 Toolkit 时不能凭空代替 nvcc |


### 6.5 CMake：把工程描述变成构建计划

**计算机科学视角**

CMake 读取 CMakeLists.txt，检查编译器、库、头文件位置和目标依赖，然后生成 build.ninja 等构建文件。它处理的是“文件与依赖关系”，不是数值计算。

例如它会形成这样的关系：

~~~text
kernel.cu 变化 → 需要重新调用 nvcc
binding.cpp 变化 → 需要重新调用 g++
两个目标完成 → 才能链接 extension.so
~~~

**工厂类比**

CMake 是工程规划员。它看设计稿和现有设备，写出“先造零件 A，再造零件 B，最后总装”的生产计划。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | CMakeLists.txt、编译器位置、库路径、编译选项 |
| 软件手段 | 配置检测、依赖图生成、平台适配 |
| 硬件手段 | CPU 执行 CMake，RAM 保存依赖图，SSD 读写构建文件 |
| 输出 | build.ninja、缓存和构建规则 |
| 不负责 | 通常不直接编译源码，不处理 Tensor，不启动 GPU |

### 6.6 Ninja：按计划调度编译命令

**计算机科学视角**

Ninja 读取 build.ninja，比较输入文件和输出文件的时间戳，决定哪些命令需要执行。它可以并行启动多个编译进程，并在依赖完成后启动链接。

**工厂类比**

Ninja 是现场调度员。它不亲自制造零件，而是告诉不同技师“现在编译 A”“等 A、B 都完成后再链接”。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | build.ninja、文件时间戳、目标名称 |
| 软件手段 | 依赖图遍历、增量构建、并行进程调度 |
| 硬件手段 | CPU 运行调度逻辑，操作系统创建 nvcc/g++ 进程，SSD 读写产物 |
| 输出 | 调用编译器后的 .o、.so 或可执行文件 |
| 不负责 | 不理解 CUDA 数学含义，不处理运行时 Tensor |

### 6.7 nvcc、g++ 与链接器：把源码变成二进制

**计算机科学视角**

一份 CUDA 扩展通常同时包含两部分：

- host code：在 CPU 上运行，由 g++ 等宿主编译器处理；
- device code：在 GPU 上运行，由 CUDA 工具链处理。

nvcc 是编译器驱动。它协调 CUDA 前端和宿主 C++ 编译器，产生目标代码。GPU 部分可能包含目标 GPU 的机器码 SASS，也可能包含可由驱动继续 JIT 的 PTX。链接器再把目标文件与 PyTorch、Python、CUDA 库的符号连接成 .so。

简化流程：

~~~text
文本源码
  ↓ 词法/语法分析
编译器内部表示
  ↓ 优化与代码生成
CPU 机器码 + GPU 代码
  ↓ 链接
.so 动态库
~~~

**工厂类比**

nvcc 和 g++ 是制造技师，把人类设计稿做成机器零件；链接器是总装工，把零件和现成标准件装成可交付设备。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | .cu/.cpp、头文件、宏、优化级别、GPU 架构参数 |
| 软件手段 | 解析、类型检查、优化、代码生成、符号链接 |
| 硬件手段 | 编译主要由 CPU 完成；RAM 保存编译器中间结构；SSD 保存 .o/.so |
| 输出 | CPU 机器码、PTX/SASS、目标文件和动态库 |
| 不负责 | 编译时通常不拿训练 Tensor 去 GPU 计算；安装 Toolkit 也不等于驱动可用 |

### 补充环节 B：操作系统与动态加载器——创建进程并装入二进制

**计算机科学视角**

shell 解析 python example.py 后，请求操作系统创建 Python 进程。操作系统为进程建立虚拟地址空间、线程、文件描述符和权限。执行 import torch 或 import my_extension 时，Python 与动态加载器把 .so 映射到进程地址空间，并解析它依赖的其他共享库和函数符号。

如果缺少共享库、ABI 不匹配或没有 GPU 设备访问权限，程序可能在尚未计算 Tensor 之前就失败。

**工厂类比**

操作系统是园区管理方：分配厂房、供电、道路和门禁。动态加载器是设备安装队：把已经制造好的机器搬进厂房，接好接口，确认所有配套零件都能找到。

| 项目 | 内容 |
| --- | --- |
| 输入 | 可执行文件、.so、依赖库名称、用户权限、环境变量 |
| 软件手段 | 进程/虚拟内存管理、文件映射、符号解析、设备权限检查 |
| 硬件手段 | CPU 执行内核与加载器代码；MMU 建立地址映射；RAM 保存进程页 |
| 输出 | 可运行的 Python 进程，以及已加载的 PyTorch/extension 代码 |
| 不负责 | 不理解神经网络数学，不替代 CUDA Runtime 调度 kernel |


### 6.8 Python 解释器：执行控制流程

**计算机科学视角**

Python 解释器在 CPU 上读取 Python 代码，将其编译为 Python 字节码或内部表示，然后逐条执行。遇到 PyTorch 调用时，它把 Python 对象和参数交给 PyTorch 的原生扩展。

例如 x + 1 在 Python 层先表达“对 Tensor x 调用加法”，真正的大规模数值循环通常在 PyTorch C++/CUDA 后端完成。

**工厂类比**

Python 是前台业务员。它接收订单、检查流程、通知生产系统，但不会亲自站在 GPU 车间做几百万次乘法。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | .py 文件、命令行参数、Python 对象 |
| 软件手段 | 解析/字节码执行、函数调用、异常处理、引用管理 |
| 硬件手段 | CPU 执行解释器，RAM 保存 Python 对象和进程状态 |
| 输出 | PyTorch API 调用、控制流、最终 Python 对象 |
| 不负责 | 不直接调度 GPU warp，不直接解释 GPU 机器码 |

### 6.9 PyTorch：理解 Tensor 并选择算子实现

**计算机科学视角**

PyTorch Tensor 不只是数值数组，还带有 shape、dtype、device、stride、存储地址等元数据。执行 x + 1 时，PyTorch 的 dispatcher 根据这些信息选择 CPU、CUDA 或其他后端实现。

PyTorch 还负责：

- 检查或传播 shape/dtype；
- 管理 Tensor 生命周期；
- 在训练时建立 autograd 计算图；
- 调用已编译算子或自定义 extension；
- 通过 CUDA caching allocator 管理常用显存块。

**工厂类比**

PyTorch 是生产管理系统。它看到订单上写着“CUDA 仓库里的 float32 Tensor”，于是选择 GPU 生产线，而不是 CPU 生产线。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | Tensor 元数据、数据指针、算子名和参数 |
| 软件手段 | dispatcher、ATen 算子、autograd、内存分配器 |
| 硬件手段 | CPU 执行调度代码；Tensor 数据可能位于 RAM 或 GPU 显存 |
| 输出 | 选中的 kernel、输出 Tensor 元数据、运行时调用 |
| 不负责 | 不代替 NVIDIA 驱动直接控制 GPU 寄存器，不提供完整 nvcc Toolkit |

### 6.10 CUDA Runtime：组织一次 GPU 作业

**计算机科学视角**

CUDA Runtime 提供 cudaMalloc、cudaMemcpy、kernel launch、stream、event 等能力。PyTorch 通常通过这些能力申请显存、安排异步拷贝、把 kernel 参数和启动配置提交给驱动。

Runtime 处理的重点是“控制和资源”：

~~~text
在哪块显存放数据？
在哪条 stream 上执行？
启动多少 block、每个 block 多少 thread？
任务之间谁先谁后？
什么时候同步？
~~~

**工厂类比**

Runtime 是车间主管：分配仓库位置、安排运输车、填写生产任务单、规定任务顺序。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | 数据地址、字节数、kernel 句柄、参数、grid/block、stream |
| 软件手段 | Runtime API、stream/event、内存与错误管理 |
| 硬件手段 | CPU 执行 Runtime 库；通过驱动安排 GPU/复制引擎工作 |
| 输出 | 给驱动的分配、拷贝、启动和同步请求 |
| 不负责 | 不亲自执行浮点乘法；Runtime 版本不等于系统 nvcc 版本 |

### 6.11 NVIDIA 驱动：把通用 CUDA 请求变成设备命令

**计算机科学视角**

驱动横跨用户态和内核态。用户态 libcuda 接收 Runtime 请求；内核驱动管理 GPU 上下文、页表、权限、命令提交和硬件通信。必要时，驱动还可能把 PTX JIT 编译为当前 GPU 可执行的机器码。

它也是版本兼容判断的核心位置之一。

**工厂类比**

驱动是拥有设备钥匙和安全权限的设备控制中心。车间主管不能直接拧机器寄存器，必须由控制中心把任务单转换为特定型号设备能执行的命令。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | 显存请求、命令队列、kernel 代码、参数、同步请求 |
| 软件手段 | libcuda、内核驱动、上下文管理、内存映射、可能的 PTX JIT |
| 硬件手段 | CPU 执行驱动代码；操作系统内核与 GPU 通信 |
| 输出 | GPU 命令、内存映射、完成事件或错误码 |
| 不负责 | 不提供 Python API，不等于 CUDA Toolkit |

### 6.12 CPU、RAM 与传输通路：控制和搬运

**计算机科学视角**

CPU 负责运行 Python、PyTorch 调度、Runtime 和大部分驱动代码。RAM 保存 Python 对象、CPU Tensor、源代码和进程状态。独立 GPU 通常通过 PCIe 与主机交换数据；在 WSL 等环境里，虚拟化层会参与设备暴露，但逻辑边界仍是主机内存和 GPU 显存。

数据搬运可能由 GPU copy engine 执行，CPU 负责提交命令，不一定逐字节亲自复制。页锁定内存和异步拷贝可改善传输，但要配合 stream 和生命周期管理。

**工厂类比**

CPU 是办公室，RAM 是普通仓库，PCIe 是运输公路，copy engine 是搬运车辆。

### 6.13 GPU 显存与 GPU 核心：真正处理 Tensor 数值

**计算机科学视角**

GPU 显存保存输入、权重、中间结果和输出。kernel 启动后，GPU 的命令处理器接收任务，SM 将 thread block 分配给 warp。warp scheduler 发射指令，load/store 单元从显存或缓存读取数据，CUDA core/Tensor Core 执行计算，结果再写回寄存器、缓存或显存。

简化的数据路径：

~~~text
GPU 显存
  ↓ load/store 与缓存
SM 中的寄存器/共享内存
  ↓ CUDA Core 或 Tensor Core
计算结果
  ↓
GPU 显存
~~~

**工厂类比**

显存是车间旁的原料仓库；SM 是并行车间；warp 是一组同步工作的工人；CUDA Core/Tensor Core 是执行算术的机器。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | GPU 地址、kernel 指令、Tensor 数值 |
| 软件手段 | 已编译 kernel 决定线程索引、访存和计算规则 |
| 硬件手段 | command processor、SM、warp scheduler、寄存器、缓存、计算核心、显存控制器 |
| 输出 | 写回 GPU 显存的 Tensor 结果 |
| 不负责 | GPU 不读取 Python 语法；CMake/Ninja 也不会在此时参与 |

### 6.14 同步、拷回与 print：结果如何变成人能看到的输出

**计算机科学视角**

kernel launch 往往是异步的：CPU 提交任务后可以继续执行。需要结果时，y.cpu() 会要求把 GPU Tensor 拷回 CPU；如果 GPU 尚未完成，相关操作必须等待。随后 PyTorch 创建 CPU Tensor，Python 再把数值格式化为字符串，终端程序把字符绘制到屏幕。

~~~text
GPU 结果
  ↓ 同步
GPU 显存
  ↓ D2H 拷贝
RAM 中的 CPU Tensor
  ↓ Python/PyTorch 格式化
字符数据
  ↓ 终端与图形系统
屏幕像素
~~~

**工厂类比**

生产完成后，货物从车间仓库运回出货区；文员读取成品、生成发货单，窗口把结果展示给人。

**输入、处理和输出**

| 项目 | 内容 |
| --- | --- |
| 输入 | GPU Tensor 和“我要在 CPU 查看它”的请求 |
| 软件手段 | stream 同步、D2H copy、Tensor 格式化、终端输出 |
| 硬件手段 | copy engine/PCIe 搬运，RAM 保存结果，CPU 转换文本，显示设备呈现 |
| 输出 | CPU Tensor、文件内容或屏幕文字 |
| 不负责 | print 不是 kernel 的输出位置；kernel 的直接输出通常先在 GPU 显存中 |

### 6.15 每个工具在哪个时间段出现

~~~text
写代码时：
编辑器 + 文件系统 + SSD

配置构建时：
CMake

执行构建时：
Ninja → nvcc/g++ → 链接器

启动程序时：
shell → Python 虚拟环境 → Python 解释器 → 动态加载器

执行算子时：
PyTorch → CUDA Runtime → NVIDIA 驱动 → GPU

得到可见结果时：
GPU 显存 → 同步/拷贝 → RAM → Python → 终端
~~~

必须注意：这些工具并不是每次都全部出现。使用 PyTorch 预编译算子时，CMake、Ninja 和 nvcc 已经在 PyTorch 发布者构建 wheel 时工作过，本机运行时通常不会再次调用它们。


## 7. 完整数据流：以向量加一为例

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
