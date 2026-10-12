# T10 数值与成本

本章承接 T08 的缩放点积注意力（Scaled Dot-Product Attention，缩放点积注意力）和 T09 的因果掩码，验证 softmax 数值稳定性、全 mask 行约定与 Attention（注意力）资源增长。CPU 实验入口为 [attention_cost.py](attention_cost.py)，GPU 实验入口为 [attention_cost_gpu.py](attention_cost_gpu.py)，对应检查分别在 [test_attention_cost.py](test_attention_cost.py) 和 [test_attention_cost_gpu.py](test_attention_cost_gpu.py)。GPU 路径用 PyTorch CUDA Tensor 运算手写同一数值 reference，不实现自定义 CUDA/Triton kernel；资源账仍用整数公式估算，不测速 FlashAttention。

资料：R04《Attention Is All You Need》§3.2.1；R18《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》§3.1～3.2；R21 服务讲义[04 注意力爆炸](../../../LLM推理服务技术详解/04-注意力爆炸.md)和[05 带宽与算力](../../../LLM推理服务技术详解/05-带宽与算力.md)。R04 原文见 [arXiv](https://arxiv.org/abs/1706.03762)，R18 原文见 [arXiv](https://arxiv.org/abs/2205.14135)。

## 前置基础

### 张量形状

Tensor（张量）是按多维形状组织的数值。单个注意力头中，Query（查询，Q）、Key（键，K）、Value（值，V）形状为 `[B,S,d]`：`B` 是 batch size（批大小），`S` 是 sequence length（序列长度），`d` 是 head dimension（单头维度）。`Q @ Kᵀ` 输出 `[B,S,S]`：每一行对应一个 query 位置，每一列对应一个 key 位置。多个头时分数矩阵形状为 `[B,H,S,S]`，`H` 是头数。

### 点积缩放

R04 的 Scaled Dot-Product Attention 在点积后除以 `sqrt(d)`。若 q、k 的各分量独立、均值为 0、方差为 1，则 `q·k` 是 d 个乘积之和，方差为 d；除以 `sqrt(d)` 后方差为 1。这样可避免维度增大时分数绝对值变大、softmax 权重过早集中。点积缩放控制分数尺度，减最大值控制指数溢出，两者作用不同。

```text
score = (Q @ Kᵀ) / sqrt(d)       # 输出 shape 仍为 [B,H,S,S]
Var(q·k) = d                      # 独立、零均值、单位方差假设
Var((q·k)/sqrt(d)) = d / d = 1   # 缩放后的方差
```

### softmax

Softmax（归一化指数函数）对一行分数逐项计算 `exp(x_j) / sum(exp(x_k))`。有效行的概率非负且总和为 1。`exp` 是 exponential function（指数函数）。浮点数可表示的范围有限；直接求 `exp(1000)` 会变成无穷大，接着出现 `inf / inf = NaN`（Not a Number，非数值）。

### mask

Attention mask（注意力掩码）说明哪些 query-key 配对可参与计算。本文 bool `allowed` 中 `True` 表示允许，`False` 表示屏蔽。允许位置的 logits（未归一化分数）要求是有限浮点数。常见实现把屏蔽分数设为负无穷；若整行都被屏蔽，直接减行最大值会得到 `-inf - (-inf)`，产生 NaN，所以必须单独处理。

### 存储层级

HBM（High Bandwidth Memory，高带宽显存）容量较大，用于存放输入、输出和中间 Tensor；SRAM（Static Random-Access Memory，静态随机存取存储器）位于 GPU 芯片内，容量较小、访问较快。Tensor 字节数等于元素数乘每元素字节数。算术运算量、张量存储量和 HBM 读写量是不同的资源指标。

## 公式推导

### 稳定 softmax

对一行 logits `x`，令 `m=max(x)`。把 softmax 的分子和分母同时除以 `exp(m)`，概率不变：

```text
softmax(x)_j = exp(x_j) / sum_k exp(x_k)
分子分母同时除以 exp(m)：
softmax(x)_j = exp(x_j - m) / sum_k exp(x_k - m)
m = max(x)，所以每个 x_j - m <= 0，最大指数是 exp(0)=1
```

以 `[1000,999,998]` 为例，减最大值后为 `[0,-1,-2]`，概率约为 `[0.665241,0.244728,0.090031]`。float64 下朴素实现产生 `[NaN,NaN,NaN]`，稳定实现则有限且归一化。减最大值不改变概率；它不能修复输入本身已有的 NaN 或无穷大。

### 全 mask 行

实验约定：若一行没有任何允许位置，则该行概率返回全零。脚本先检测是否存在允许 key，再使用安全最大值和安全分母，避免 `-inf - (-inf)` 和除零。

| 行 | logits | allowed | 输出概率 | 行和 |
| --- | --- | --- | --- | --- |
| 有效行 | `[2,1]` | `[True,False]` | `[1,0]` | 1 |
| 全 mask 行 | `[3,4]` | `[False,False]` | `[0,0]` | 0 |

将概率乘 V 后，全 mask 行的注意力加权输出是零向量；若后续有残差连接（Residual Connection，残差连接），残差输入仍可能留在层输出中。正常因果掩码通常允许有效 token 读取自己；padding（填充）或自定义稀疏掩码仍可能产生全 mask 行。

### 在线 softmax

FlashAttention（快速注意力）把 Query（查询，Q）按行分块，把 Key（键，K）和 Value（值，V）按位置分块，在 SRAM 中处理当前分数块，并在线更新归一化状态。对一行 query，令已处理部分的最大分数、指数和、未归一化加权 Value 和分别为 `m`、`l`、`u`。当前块分数是 `x`，对应 Value 块是 `V_block`：

```text
m_new = max(m, max(x))
alpha = exp(m - m_new)
p     = exp(x - m_new)
l_new = alpha * l + sum(p)
u_new = alpha * u + p @ V_block
最终输出 = u_new / l_new
```

新的全局最大值变大时，旧的累计量乘 `alpha`，就能换算到新最大值对应的指数尺度。由于 `m_new >= m`，`alpha` 不大于 1；新块 `p` 中的指数也不大于 1。处理完所有块后，数学结果等价于完整 softmax 再乘 V；不同浮点归约次序可能造成末位舍入差异，不保证逐位相同。

#### 数值例子

用一行分数 `[2,4,3,5]` 和对应 Value `[10,20,30,40]`，分两块计算：块 1 为分数 `[2,4]`、Value `[10,20]`；块 2 为分数 `[3,5]`、Value `[30,40]`。

```text
块 1：m1 = 4
p1 = [exp(2-4), exp(4-4)] = [exp(-2), 1]
l1 = exp(-2) + 1
u1 = exp(-2)*10 + 1*20

块 2 到达后：m2 = max(4, 3, 5) = 5
alpha = exp(m1-m2) = exp(4-5) = exp(-1)
p2 = [exp(3-5), exp(5-5)] = [exp(-2), 1]
l2 = alpha*l1 + exp(-2) + 1
u2 = alpha*u1 + exp(-2)*30 + 1*40
最终加权输出 = u2/l2
```

旧块的权重原先按最大值 4 缩放；乘 `exp(4-5)` 后，与直接按最终最大值 5 计算使用同一尺度。这就是在线更新无需保存完整分数行的原因。

### 资源公式

设 batch 为 `B`、头数为 `H`、序列长度为 `S`、单头维度为 `d`，浮点元素大小为 `b` bytes。score（注意力分数）和 probability（注意力概率）矩阵各含 `B*H*S*S` 个元素：

```text
一个 score 或 probability 矩阵字节数 = B * H * S * S * b
Q、K、V、O 四个 Tensor 总字节数        = 4 * B * H * S * d * b
每层 K、V cache 字节数                  = 2 * B * H * S * d * b
QKᵀ 与 P@V 两次 matmul 的 FLOPs         ≈ 4 * B * H * S * S * d
```

FLOPs（Floating-Point Operations，浮点运算次数）按一次乘加计 2 次。资源账假设标准多头自注意力且 Q/K/V/O 使用相同 dtype（Data Type，数据类型）；不含投影层、反向传播保存值、分配器碎片、临时工作区和其他模型激活。下面的 bool mask 假设按完整 `[B,H,S,S]` 物化且每项 1 byte；实际可以使用广播的小 mask 或由 kernel 隐式生成。

### 序列翻倍

用 `B=1,H=16,d=64,fp16`（16-bit floating-point，16 位浮点数，每元素 2 bytes）比较 `S=4096` 和 `S=8192`。MiB/GiB 使用二进制单位；score 和 probability 分别是一份矩阵。

| 项目 | S=4096 | S=8192 | 变化 |
| --- | ---: | ---: | ---: |
| score 矩阵（fp16） | 512 MiB | 2.0 GiB | 4 倍 |
| probability 矩阵（fp16） | 512 MiB | 2.0 GiB | 4 倍 |
| 完整展开 bool mask（1 byte） | 256 MiB | 1.0 GiB | 4 倍 |
| Q、K、V、O 合计（fp16） | 32 MiB | 64 MiB | 2 倍 |
| 每层 Key-Value cache（键值缓存） | 16 MiB | 32 MiB | 2 倍 |
| 两次 Attention 矩阵乘法 FLOPs | 68,719,476,736 | 274,877,906,944 | 4 倍 |

原因是 score/probability/mask 同时有两个 S 轴，S 翻倍会使元素数乘 4；Q/K/V/O 与 KV cache 只含一个 S 轴，元素数乘 2。完整展开的 mask 是逻辑账，不代表所有程序都会实际分配这么大的 Tensor。

### 搬运对比

未融合的朴素实现可由独立矩阵乘法、softmax、矩阵乘法算子构成。若中间结果物化到 HBM，流程示意如下：

1. 读取 Q、K，计算 score，并将完整 `[S,S]` 分数矩阵写入 HBM。
2. 重新读取 score，计算 probability，并将完整 `[S,S]` 概率矩阵写入 HBM。
3. 重新读取 probability 与 V，计算输出 O，再写回 O。

这类实现会对二次规模中间矩阵产生 HBM 写出和读回。具体后端是否物化完整矩阵取决于采用的算子和融合实现，不能把该流程泛化到所有 Attention 实现。

FlashAttention 通过分块融合矩阵乘法、softmax 与 Value 加权，避免将完整 score/probability 矩阵写入 HBM 后再读回。它仍然要读取 Q/K/V、写出 O，并可能按 tile（分块）调度重复读取某些输入块或暂存部分输出/归一化状态；被消除的是完整 S×S 中间矩阵的 HBM 往返。

| 对比项 | 物化中间矩阵的朴素路径 | FlashAttention 路径 |
| --- | --- | --- |
| score/probability | 完整 S×S 矩阵可能写入并读回 HBM | 分块临时计算，不把完整矩阵写回 HBM |
| 主要运算量 | `O(B*H*S*S*d)` | `O(B*H*S*S*d)`，稠密算术量仍是二次 |
| 额外注意力状态 | 可能随 S² 保存 score/probability | 按输出行保存分块结果和行级归一化统计 |
| 实现方式 | 多个独立 GPU kernel（GPU 计算核）时有中间读写 | 典型融合实现合并相关步骤，块数据在片上复用 |

R18 的 HBM IO（Input/Output，输入/输出）分析在相应 SRAM 容量 `M` 和 head dimension `d` 条件下给出约 `O(S*S*d*d/M)` 的访问复杂度；该量描述不同显存层级间的数据搬运，不是算术量。FlashAttention 保留精确稠密 Attention 的 query-key 配对计算，算术量仍随 `S*S*d` 增长。分块减少中间矩阵读写，但不会令所有输入输出流量消失，也不保证每种 shape 都变成 compute-bound（算力受限）。论文说明其使用分块减少 HBM 与片上 SRAM 之间的读写，并以在线 softmax 处理归一化。[R18 原论文](https://arxiv.org/abs/2205.14135)

### 复杂度边界

R21 讲义 04 用“完整分数矩阵不写入 HBM”解释显存收益；对浮点计算要避免写成“结果逐位相同”。在线分块改变了归约顺序，数学定义等价但浮点舍入可能不同。R18 论文还利用重计算处理反向传播中间值，因此不要把前向 tile 的局部存储表当作训练全过程的所有显存峰值。

## 符号说明

| 符号/术语 | 含义 |
| --- | --- |
| Q / K / V / O | Query（查询）/ Key（键）/ Value（值）/ Output（输出） |
| B / H / S / d | 批大小 / 头数 / 序列长度 / 单头维度 |
| b | 每个浮点元素占用的字节数 |
| m / l / u | 在线 softmax 当前最大值 / 指数和 / 未归一化加权 Value 和 |
| alpha | 新最大值到来后换算旧累计量的因子，`exp(m_old-m_new)` |
| M | R18 IO 复杂度分析中的 SRAM 容量参数 |
| HBM | High Bandwidth Memory，高带宽显存 |
| SRAM | Static Random-Access Memory，静态随机存取存储器 |
| IO | Input/Output，输入/输出；此处为显存层级间读写 |
| FLOPs | Floating-Point Operations，浮点运算次数 |
| NaN | Not a Number，非数值标记 |
| fp16 / float64 | 16 位 / 64 位浮点数据类型 |
| KV cache | Key-Value cache，键值缓存 |
| Kernel Fusion | 算子融合；将多个算子步骤合入一个 GPU kernel |
| Tiling | 分块；把大矩阵运算拆成可驻留片上存储的小块 |
| Memory-bound | 访存受限；数据搬运时间主导工作耗时 |
| Compute-bound | 算力受限；算术计算时间主导工作耗时 |
| `exp` | Exponential function，指数函数 |

## 程序实例

| 顺序 | 入口/函数 | 输入 | 输出与检查 |
| --- | --- | --- | --- |
| 1 | CPU `main()` | logits `[1,3]`，float64 | 对照朴素与稳定 softmax，检查有限性 |
| 2 | CPU `masked_softmax()` | logits 与可广播 bool mask | 屏蔽位置为 0，全 mask 行返回全零概率 |
| 3 | `run_gpu_experiment()` | CUDA logits `[1,3]` 和 `[2,2]`，float64 | 在 GPU 执行 softmax/mask 并检查设备与数值 |
| 4 | `estimate_attention_resources()` | B/H/S/d/元素字节数 | CPU 纯整数资源账，不申请大矩阵 |
| 5 | CPU 与 GPU unittest | 小型 float64 Tensor | 检查稳定性、mask 边界和翻倍比例 |

GPU 执行顺序是：确认 CUDA 可用并选当前设备 → 将固定 logits 与 bool mask 直接创建在 GPU → 在 GPU 做 exp、行最大值、归约和 where → 同步待处理工作并将少量输出复制到 CPU 打印 → 用 Python 整数核算 S=4096/8192 的资源，不创建大矩阵。GPU 上这些 Tensor 由局部变量持有，函数结束后引用释放；CUDA caching allocator 可能仍保留已释放内存，因此程序分别打印 allocated（当前活跃分配）和 reserved（分配器保留）值。这里只用 float64 小张量检查 GPU 正确性，不能推断 fp16/bf16 稳定性或带宽性能。

从仓库根目录运行：

```bash
.venv/bin/python learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/attention_cost.py
.venv/bin/python learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/test_attention_cost.py
.venv/bin/python learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/attention_cost_gpu.py
.venv/bin/python learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/test_attention_cost_gpu.py
```

CPU 和 GPU 路径的资源脚本只做整数乘法，没有申请表格里数百 MiB 或数 GiB 的 Tensor；所以这些数字是资源估算，不是设备峰值显存测量。GPU 脚本运行小型 float64 数值检查，不覆盖 fp16/bf16 softmax、反向传播、融合后端选择或 FlashAttention 性能。

## 直观理解

在线 softmax 像分批按最高温调果汁：后来发现更高温度，就按新旧最大值的指数系数调整旧账，再加上新一批，不必从头重做。FlashAttention 也把中间草稿留在片上，只将最终结果写回显存。
