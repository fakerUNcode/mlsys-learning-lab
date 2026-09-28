# B01 / T01：当前设备环境验收

- 日期：2026-09-27
- 仓库 commit：`aa877158ce061e566c73517f8e6e97e2eb4850c`（验收开始时）
- 状态：**通过。**

## 验收目标

记录当前 WSL/GPU/PyTorch 环境；验证 GPU Tensor 加法、最小自定义 GPU kernel 和 profiler。需要区分“运行 PyTorch/Triton 已编译或 JIT kernel”与“系统安装完整 CUDA Toolkit、可用 `nvcc` 编译 `.cu` 文件”。

## 本报告术语与前置知识

- **NVIDIA 驱动**负责让操作系统和程序访问 GPU；WSL 使用 Windows 主机驱动提供的 GPU 接口。
- **CUDA Runtime / Driver API**负责在程序运行时分配设备内存并提交 kernel。
- **CUDA Toolkit / `nvcc`**提供开发工具；`nvcc` 把 `.cu` 源码编译为可执行程序。仅能运行 PyTorch GPU 运算，不足以证明 `nvcc` 已安装。
- `nvidia-smi` 显示的 CUDA 版本代表驱动支持的 CUDA 版本；要确认 Toolkit 是否安装，需另查 `nvcc --version`。


![CUDA 源码编译路径与 PyTorch/Triton 运行路径汇合到 WSL GPU 驱动和设备的示意图](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/t01_cuda_paths.svg)

## 环境快照

| 项目 | 实测 |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS，WSL2；内核 `6.18.33.2-microsoft-standard-WSL2` |
| GPU | NVIDIA GPU |
| GPU 架构 | Compute Capability 12.0 |
| 总显存 | 16,303 MiB（约 15.92 GiB） |
| 检查时显存占用 | 2,932 MiB；该值会随其他进程变化 |
| NVIDIA 驱动 | 572.84（`nvidia-smi` 报告 CUDA 12.8） |
| GCC / G++ | 13.3.0 |
| CUDA Toolkit / `nvcc` | CUDA Toolkit 12.8 已安装；`/usr/local/cuda-12.8/bin/nvcc`，版本 12.8.93 |
| Python（系统） | 3.12.3；没有安装 PyTorch |
| Python（项目 `.venv`） | 3.12；PyTorch 2.11.0+cu128 |
| PyTorch CUDA Runtime | 12.8；`torch.cuda.is_available()` 为 True |
| cuDNN | 91900 |
| Triton | 已安装，可用于 JIT kernel 验证 |
| CMake / Ninja | 3.28.3 / 1.11.1 |

环境检查命令：

```bash
PYTHON_BIN=.venv/bin/python bash scripts/check_environment.sh --purpose=stage-0
nvidia-smi
```

说明：用户的交互式 Bash 在 `~/.bashrc` 中设置了 `/usr/local/cuda-12.8/bin`，交互 shell 可直接运行 `nvcc`。本次自动化工具使用的非交互 shell 未加载 `.bashrc`，所以裸命令 `nvcc` 曾显示 not found；Toolkit 本身并未缺失。可使用绝对路径，或在需要 CUDA 的非交互命令前显式设置 `PATH=/usr/local/cuda-12.8/bin:$PATH`。无需重新安装 Toolkit。

## 验证结果

### GPU Tensor 加法：通过

在项目 `.venv` 中创建 CUDA float32 Tensor，执行 `c = a + b`，显式同步，并与期望值逐元素比较，结果一致。设备为 NVIDIA GPU GPU。

### 最小 GPU kernel：Triton 与原生 CUDA 路径均通过

使用 Triton JIT 编写向量加法 kernel，输入长度 `N=10003`，block 大小 256，覆盖非整 block 的尾部 mask。结果与 PyTorch CUDA 加法严格一致，最大绝对误差为 0。

另用 CUDA Toolkit 12.8 的 `nvcc` 编译并运行独立 `.cu` 向量加法程序。编译参数为 `-O2 -arch=sm_120`，输入长度 `N=10003`、block 大小 256；程序报告设备 `NVIDIA GPU (sm_120)`、`mismatches=0`。编译和 kernel 执行均成功。

复现命令（测试源码本次保存在 `/tmp/b01_t01_vector_add.cu`）：

```bash
/usr/local/cuda-12.8/bin/nvcc -O2 -arch=sm_120 \
  /tmp/b01_t01_vector_add.cu -o /tmp/b01_t01_vector_add
/tmp/b01_t01_vector_add
```

### PyTorch Profiler：通过

用 `torch.profiler.profile` 采集一次 GPU 加法，CPU 与 CUDA activity 均启用。成功捕获 `aten::add` 和底层 CUDA elementwise kernel。该次采样用于工具可用性检查，不作为性能基准。

## 验收结论

- **已通过：** WSL 可见 NVIDIA GPU；项目虚拟环境中的 PyTorch CUDA Runtime 可用；GPU Tensor 加法正确；Triton 与原生 CUDA 自定义 kernel 正确；PyTorch profiler 能采集 CUDA 活动；CUDA Toolkit 12.8 可编译面向 `sm_120` 的 kernel。
- **阻塞：** 无。裸 `nvcc` 在非交互 shell 中不可见是 PATH 初始化差异；绝对路径和交互 shell 均已验证可用。
- **T01 状态：** 通过。可以进入 B01 后续 Tensor 与布局任务。若从非交互脚本调用 CUDA 编译器，使用绝对路径或显式添加 Toolkit 的 `bin` 目录。

## 复现步骤

```bash
PYTHON_BIN=.venv/bin/python bash scripts/check_environment.sh --purpose=stage-0
.venv/bin/python -c 'import torch; x=torch.arange(1024,device="cuda"); y=x+2; torch.cuda.synchronize(); assert torch.equal(y,x+2); print(torch.cuda.get_device_name(0), "PASS")'
/usr/local/cuda-12.8/bin/nvcc --version
/usr/local/cuda-12.8/bin/nvcc -O2 -arch=sm_120 /tmp/b01_t01_vector_add.cu -o /tmp/b01_t01_vector_add
/tmp/b01_t01_vector_add
```

Triton kernel 与 profiler 验证在本次验收中通过临时脚本执行；此处不把临时检查脚本当作长期实验产物。
