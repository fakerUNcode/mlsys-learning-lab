# 第 3 章：dtype、device 与存储检查

## 本节术语与前置知识

- **dtype**：每个元素用什么数值格式存储，例如 `float32` 每个元素通常占 4 字节，`float16` 通常占 2 字节。
- **device**：Tensor 所在的计算设备，例如 `cpu` 或 `cuda:0`。
- **storage 槽位**：某一块 storage 内从 0 开始数的元素偏移。CPU、GPU 和转换后新建的 storage 各自编号；不同 storage 的槽位 0 不是同一份内存。
- `.to(dtype=...)` 可改变数值表示；`.to(device=...)` 可把数据转移到另一设备。目标与来源完全相同时，PyTorch 通常可避免复制。

扩展术语见[统一术语表](../../terms.md)。

## 9. dtype 和 device：数据类型与所在设备

### dtype：每个元素如何表示

![float32 转 float16 以及 CPU 到 GPU 往返时值写入各自目标 storage 的示意图](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/t03_dtype_device.svg)

`dtype` 是元素的数据类型，例如 `torch.float32`、`torch.float16`、`torch.int64`。它影响数值范围、精度和每个元素所占字节数：float32 每元素 4 字节，float16 每元素 2 字节，int64 每元素 8 字节。

```python
x = torch.tensor([1.25, 2.5], dtype=torch.float32)
y = x.to(torch.float16)
```

先用“storage 槽位”看它：这里两个 Tensor 都是一维、shape `(2,)`、stride `(1,)`，所以索引 0 对应各自 storage 的槽位 0，索引 1 对应各自 storage 的槽位 1。

```text
x（float32 原 storage）：槽位 0 存 1.25；槽位 1 存 2.5
y（float16 新 storage）：槽位 0 存转换后的 1.25；槽位 1 存转换后的 2.5
```

槽位编号相同，**不代表这两个槽位是同一块内存**。改变 dtype 要把数值编码成另一种格式，通常分配新 storage：float32 每个槽位占 4 字节，float16 每个槽位占 2 字节。若忽略 allocator 的对齐和管理开销，源 storage 中两个元素的字节起点是 0、4，目标 storage 中是 0、2；两个 storage 的基地址也不同。实际地址由运行时分配，不应把这些示意偏移误认成机器的 RAM 物理地址。

`y` 的 shape 不变，dtype 变成 float16，数值会按较低精度表示。若 `.to()` 的目标 dtype/device 与输入完全相同，PyTorch 通常直接返回原 Tensor，不做无谓复制；本例 dtype 不同，所以需要转换和新存储。

<a id="fp16-cast-vs-quantization"></a>

### 和模型量化的联系：都在用更少的位表示数据，但做法不完全相同

把模型参数从 float32 存成 float16，与把一个 Tensor 转成 float16，是同一种浮点格式转换应用在不同对象上：每个值仍然是浮点数，只是可用位数更少。比如下面把两个 float32 元素转成 float16：

```python
p32 = torch.tensor([1.0001, 2.0], dtype=torch.float32)
p16 = p32.to(torch.float16)

print(p32)  # tensor([1.0001, 2.0000])
print(p16)  # tensor([1., 2.], dtype=torch.float16)
print(p32.numel() * p32.element_size())  # 8 字节逻辑元素数据：2 个元素 × 4 字节
print(p16.numel() * p16.element_size())  # 4 字节逻辑元素数据：2 个元素 × 2 字节
```

`1.0001` 在 float16 中无法保持原来的细微差别，会舍入成更接近的可表示值；元素个数和 shape 没变，但这两个元素的逻辑数据字节数减半。这里计算的是元素本身的数据量，不包括分配器、对齐等额外开销。前面的 `1.25`、`2.5` 是特意选的例子，它们能被 float16 精确表示，所以单看那些值不容易观察到精度损失。float16 相对 float32 的可表示范围也更窄，超出范围的值还可能溢出。

这和之前学过的模型量化==共享一个核心权衡：降低每个权重的存储位数==，通常能减少模型占用和权重搬运量，但可能带来数值误差。要区分具体做法：

| 做法 | 值如何表示 | 主要区别 |
|---|---|---|
| `tensor.to(torch.float16)` | 把每个元素转换成 16 位浮点数 | 这里是一个 Tensor 的 dtype 转换；shape 不变，元素数值可能舍入 |
| 模型使用 FP16 权重 | 模型参数以 16 位浮点数保存 | 与上面的浮点格式相同，但对象是模型参数；运算路径还取决于框架、算子和硬件 |
| INT8/INT4 等权重量化 | 把权重映射到更低位数的量化表示 | 通常还要记录比例因子等辅助信息，并使用相应的量化/反量化计算；具体方案不同 |

所以，本节的 `.to(torch.float16)` 可以帮助理解“更少位数会带来更少存储与精度取舍”，但它本身**不等于**把权重压成 INT8/INT4 的量化流程。关于 FP16、BF16 的精度和字节数，回看[《名词地基》：fp16、精度与字节](../../../../LLM推理服务技术详解/01-名词地基.md#fp16-precision-and-bytes)；关于降低权重位数如何影响理想搬运时间，回看[《带宽与算力》：量化为什么提速](../../../../LLM推理服务技术详解/05-带宽与算力.md#model-quantization-memory-bandwidth)。

### device：Tensor 放在哪里

常见 device 是 CPU 内存和 CUDA GPU 显存：

```python
cpu_x = torch.arange(4, dtype=torch.float32)       # 默认在 CPU
gpu_x = cpu_x.to("cuda")                          # 复制/传输到 GPU
back = gpu_x.to("cpu")                            # 传回 CPU
```

按逻辑索引追踪每个值及其所在设备的 storage 槽位：

```text
cpu_x 的值：             [0, 1, 2, 3]
CPU storage 槽位：       [0, 1, 2, 3]

gpu_x 的值：             [0, 1, 2, 3]
GPU storage 槽位：       [0, 1, 2, 3]

back 的值：              [0, 1, 2, 3]
新的 CPU storage 槽位：  [0, 1, 2, 3]
```

每次转换都按索引 0、1、2、3 复制相同值，但目标设备会有自己的 storage。CPU 槽位 0、GPU 槽位 0、返回后新 CPU storage 的槽位 0 属于三块不同存储；它们只是都对应各自 Tensor 的逻辑索引 0。跨 CPU/GPU 转移会在目标设备建立数据，通常需要新存储和传输；shape/dtype 不变（除非同时指定 dtype）。`gpu_x` 和 `cpu_x` 不共享普通 Tensor 存储。GPU 操作可能异步提交；需要在 CPU 读取 GPU 计算结果或检查时间时使用 `torch.cuda.synchronize()`。

支持哪些 dtype/kernel 可能因设备不同而不同。做实验时要打印真实的 `device` 和 `dtype`，不能只根据创建代码猜测。

## 10. 本节操作的存储行为速查

| 操作 | shape/布局效果 | 通常是否新分配存储 | 修改结果是否可能影响输入 |
|---|---|---:|---:|
| `y = x` | 不创建新 Tensor 数据 | 否 | 是，实际上就是同一对象 |
| 基本切片 | 选取部分元素，常改变 shape/offset/stride | 否，通常是 view | 是 |
| `transpose` / `permute` | 交换轴，更新 shape/stride | 否，通常是 view | 是 |
| `view` | 改 shape，要求 stride 兼容 | 否；不兼容则报错 | 是 |
| `reshape` | 请求新 shape | 可能；可 view 时不复制，否则复制 | 可能；不要依赖它一定别名 |
| `contiguous` | 整理为目标连续布局 | 输入已符合时不需要；否则复制 | 复制时否；已连续时可能仍是原存储 |
| `clone` | 数值相同，独立副本 | 是 | 否 |
| `.to(dtype=...)` | 转换元素表示，shape 不变 | dtype 变化时通常是 | dtype 变化时否 |
| `.to(device=...)` | 转移到 CPU/GPU，shape 不变 | 跨设备时通常是 | 跨设备时否 |

表中“通常”是有意保留：对 `reshape`、`contiguous` 和 `.to`，若数据已符合要求，PyTorch 可以避免复制。需要依赖别名关系时要通过实际行为确认或主动使用 `clone()`。

## 11. 复现和检查一个 Tensor

对每个关键中间结果打印这些信息：

```python
def describe(name, tensor):
    print(
        name,
        "shape=", tuple(tensor.shape),
        "stride=", tensor.stride(),
        "contiguous=", tensor.is_contiguous(),
        "dtype=", tensor.dtype,
        "device=", tensor.device,
        "data_ptr=", tensor.untyped_storage().data_ptr(),
    )
```

`data_ptr` 返回运行时使用的指针（CPU 虚拟地址或设备地址），不是可供我们读取的硬件 RAM 物理地址。它用来辅助比较 storage 起始地址；相同地址是共享存储的强证据。不同 view 即使共用 storage，也可能有不同 storage offset，因此还要看偏移。本地实验里用元素槽位编号和 `storage_offset()` 解释映射，用 storage 指针及修改实验验证别名。对于空 Tensor、特殊 backend 或复杂 view，按操作文档判断，不把地址比较当成所有情况的唯一证明。

## 12. 官方资料

本地教程为主线；官方文档供术语核对和深入阅读：

- R02：[Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)，重点看 Tensor View、`view`/`reshape`、`contiguous`、`transpose`。
- R01：[Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)，回顾 `dtype`、`device` 和 Tensor 创建。
