# 第 1 章：存储、stride 与连续性

本章先回答“Tensor 的数字如何放进内存、PyTorch 如何找到它们”。请把 shape 理解为逻辑坐标范围，把 stride 理解为坐标映射到线性存储的步长；下一章再用这套模型解释 transpose、view 和复制。

## 本节术语与前置知识

- **shape** 规定合法逻辑坐标范围；shape `(2,3)` 的坐标如 `(1,2)`。
- **storage** 是一块 Tensor 元素存储；**storage 槽位**是这块存储内从 0 开始的元素编号。它不同于逻辑坐标，也不是运行时地址。
- **stride** 是逻辑坐标沿某轴增加 1 时，storage 槽位增加的元素数；单位是元素，不是字节。
- 具体例子：`x=[[10,20,30],[40,50,60]]` 的 shape 是 `(2,3)`，连续 stride 是 `(3,1)`。若 storage offset 为 0，坐标 `x[1,2]` 映射到槽位 `0 + 1×3 + 2×1 = 5`，该槽位存 `60`。

扩展术语见[统一术语表](../../terms.md)。

## 0. 本节要回答的四个问题

给定一个 Tensor，除了 `shape`，还要能回答：

1. 元素实际存在哪里？两个 Tensor 会不会共用同一块存储？
2. `stride` 如何从逻辑坐标找到实际元素？
3. 为什么 `transpose` 后看起来 shape 变了，但数据没有按新顺序搬家？
4. `view`、`reshape`、`contiguous`、dtype 转换和 CPU/GPU 转换，什么时候复用数据、什么时候产生新存储？

## 1. Tensor 不只是数字：元数据和存储

一个普通 dense Tensor 可以先想成两部分：

- **存储（storage）**：一段线性排列的元素；
- **元数据（metadata）**：shape、stride、storage offset、dtype 和 device 等，告诉 PyTorch 怎样解释存储。

示例：

```python
import torch

x = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])
```

`x.shape == (2, 3)`。可以把底层存储想成一排 `[10, 20, 30, 40, 50, 60]`，而 shape 告诉我们把它读成 2 行 3 列。Tensor 的逻辑下标是 `(行, 列)`；例如 `x[1, 2]` 是 `60`。

不同 Tensor 可以用不同的元数据解释同一块存储。因此，shape 改变不一定意味着元素被复制或重排。

### shape 和 stride 各自回答什么

- **shape 描述逻辑形状**：每个轴有多少个可选索引，也就是哪些逻辑坐标合法。例如 shape `(2, 3)` 表示合法坐标为 `0≤i<2`、`0≤j<3`。
- **stride 描述坐标到 storage 槽位的映射**：某个轴的坐标增加 1 时，storage 槽位增加多少。给定坐标 `(..., i, ...)`，stride 参与计算这个坐标对应的 storage 槽位。

所以，stride 是用来把**逻辑坐标映射到 storage 槽位**的，不是用来表示 shape 的。shape 和 stride 有关系，但不能把二者混为一谈：shape 决定轴长度；实际 stride 还取决于 Tensor 的布局以及它经历过的操作。

例如，shape `(2, 3)` 可以搭配连续布局 stride `(3, 1)`，也可以是一个转置视图的 stride `(1, 3)`。两者 shape 一样，逻辑坐标范围一样，但同一坐标会按各自的 stride 映射到不同的 storage 槽位。因此**只知道 shape，通常不能推断实际 Tensor 的 stride**。

有一个常见特例：若已知 Tensor 使用默认的 row-major 连续布局，就可以从 shape 推算这种布局“应该有”的标准 stride。例如 shape `(2, 3)` 的默认连续 stride 是 `(3, 1)`。这是在额外知道布局条件后推算期望 stride，不表示 shape 单独决定实际 stride。`transpose` 正是会改变 stride、但不必搬动数据的操作。

## 2. stride：从逻辑坐标算出 storage 槽位

`stride`（步长）说明：某一轴的逻辑坐标增加 1 时，storage 槽位要跨过多少个元素。PyTorch 的 stride 单位是**元素个数**，不是字节。**记住：stride 描述逻辑坐标怎样映射到 storage 槽位；它不描述 Tensor 的 shape，也不能只由 shape 推出来。转置会改变坐标到槽位的映射，所以转置后尤其要重新查看 stride。**

仍看 `x`：

```python
x = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])
print(x.shape)   # (2, 3)
print(x.stride())  # (3, 1)
```

为什么 stride 是 `(3, 1)`？

- 行号增加 1，要跳过一整行的 3 个元素，所以第 0 轴 stride 是 3。
- 列号增加 1，只需跳到相邻元素，所以第 1 轴 stride 是 1。

若 storage offset 为 0，逻辑坐标 `(i, j)` 对应的 storage 槽位为：

```text
offset + i * stride[0] + j * stride[1]
```

例如 `x[1, 2]` 对应 storage 槽位 `0 + 1*3 + 2*1 = 5`，也就是这一块 storage 的第 6 个元素 `60`。这个公式将逻辑坐标、stride 与元素所在的 storage 槽位连起来。

对 n 维 Tensor，storage 槽位公式推广为：

```text
storage_offset + Σ(index[d] * stride[d])，d 遍历所有轴
```

一般不需要手动使用这个公式写算子；学会它是为了看懂 transpose 和 view。

## 3. 连续（contiguous）是什么意思

转置图会同时标出逻辑数值和对应的 storage 槽位；先看值的排列，再沿行读槽位，就能直观看到为什么转置视图不连续。

![转置前后 shape、stride 和 storage 槽位映射](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/t03_transpose_stride.svg)

这里说“连续”，指的是：**按照 Tensor 平常的逻辑顺序逐个读取元素时，storage 槽位也一个接一个地前进，中间不跳过其他元素。** 对默认的 row-major（按行排列）布局，先读完第一行，再读第二行。

看一个具体例子：

```python
x = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])
```

`x` 的逻辑读取顺序是 `10, 20, 30, 40, 50, 60`，底层也按这一次序紧挨着存放。shape 是 `(2, 3)`；从一行读到下一行需要跨 3 个元素，从一列读到下一列跨 1 个元素，所以 stride 是 `(3, 1)`。这正是普通二维 row-major 连续布局。

判断连续与否，必须同时看 **shape、stride 和选定的内存格式**；不能仅凭内容打印得像个整齐矩阵来判断。PyTorch 默认的 row-major 连续 stride 由 shape 决定，从右向左计算：

```text
最后一轴 stride = 1
前一轴 stride = 后一轴长度 × 后一轴 stride
```

例：shape `(2, 3, 4)` 的连续 stride 为 `(12, 4, 1)`：

- 最后一轴坐标前进一格，对应槽位增加 1；
- 中间轴前进一格要跨过一行 4 个元素；
- 最前轴前进一格要跨过一个 `3×4` 的块，即 12 个元素。

PyTorch 检查是否符合某种内存格式的标准布局可用 `is_contiguous()`：

```python
x = torch.arange(6).reshape(2, 3)
print(x.stride())         # (3, 1)
print(x.is_contiguous())  # True
```

所以，“连续”不是说 Tensor 的数字必须是一维，也不是说它必须独占存储。它问的是：**按默认行优先顺序遍历这个 Tensor，元素在底层是不是相邻地排列？** 某些轴长度为 1 时，stride 有特殊自由度；初学阶段用 `is_contiguous()` 判断，不必仅凭 stride 手工处理这些边界。
