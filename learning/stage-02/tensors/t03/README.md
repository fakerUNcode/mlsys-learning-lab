# T03 布局实验记录

## 本记录前置知识

- **shape** 是每个轴的长度；shape `(3,4)` 表示合法逻辑坐标 `(0..2, 0..3)`。
- **stride** 是逻辑坐标沿某轴增加 1 时，storage 槽位增加的元素数。shape `(3,4)` 的默认连续 stride 是 `(4,1)`。
- **storage 槽位**是某一块 storage 内的元素偏移，不是逻辑坐标，也不是硬件 RAM 物理地址。例：offset 为 0、stride 为 `(4,1)` 时，坐标 `(2,3)` 对应槽位 `2×4+3×1=11`。
- 两个 Tensor 的槽位编号相同，不一定引用同一块 storage；要结合 storage、storage offset 和修改实验判断别名。

先按顺序阅读第 1 章[存储、stride 与连续性](chapters/01_storage_stride_contiguous.md)、第 2 章[视图、reshape 与别名](chapters/02_views_reshape_aliasing.md)、第 3 章[dtype、device 与存储检查](chapters/03_dtype_device.md)。每个预测先独立填写，再运行脚本核对。完整词义见[统一术语表](../terms.md)。失败题要解释失败原因；对别名题写出源、结果的槽位对应及改值前后的具体内容。

## 实验 A：shape、stride 和连续性

代码起点：

```python
x = torch.arange(12).reshape(3, 4)
t = x.transpose(0, 1)
```

| 对象 | 预测 shape | 预测 stride | 是否连续 | 运行后记录 |
|---|---|---|---|---|
| `x` | | | | |
| `t` | | | | |

解释：`x` 的 `(3,4)` 两个轴各前进一步，storage 槽位分别跨多少个元素？转置后为什么 shape 和 stride 都换了？

## 实验 B：transpose 后的 view

预测 `t.view(12)` 会成功还是报错？先列出 `t` 按行读取时每个元素对应的 storage 槽位，再检查相邻槽位差是否固定。运行脚本记录真实结果与错误原因。

## 实验 C：reshape 是否复制

比较：

```python
a = x.reshape(2, 6)
b = t.reshape(12)
```

预测 a、b 的逻辑数值顺序及是否与来源共用存储。分别列出来源和结果中逻辑坐标对应的 storage 槽位；若结果用了新 storage，从新 storage 槽位 0 开始另列。脚本会修改 reshape 结果的一个元素，观察来源 Tensor 是否随之改变。解释为什么一项能重用存储、另一项需要整理元素顺序。

## 实验 D：contiguous 与 clone

预测 `t.contiguous()` 的 shape、stride、连续性、逻辑数值顺序和别名情况。把 `t` 的旧 storage 槽位映射与新 Tensor 的新 storage 槽位映射分开记录。再预测 `x.clone()` 是否与 x 共用存储。记录实验输出。

## 实验 E：dtype 与 device

CPU 上将 float32 转为 float64：预测哪些属性变化、是否通常新分配存储。跟踪输入槽位 0、1 的值如何写入目标 storage 的槽位 0、1，注意两边槽位编号相同不代表同一块存储。若 CUDA 可用，再执行 CPU→CUDA→CPU：按每个设备分别记录 storage 槽位、device/dtype/shape，验证数值一致，并说明跨设备复制与 view 的区别。CUDA 不可用时记为阻塞并保留 CPU 结果。

## 实验 F：独立复写

不看教程，独立描述下面这条链路，每步写 shape、stride、连续性和是否可能新分配：

```text
创建连续矩阵 → transpose → contiguous → view
```

再用一个两三行的小代码片段演示“view 修改会影响原 Tensor”，说明你如何确认别名。

## 验收记录

- shape/stride/连续性：待完成。
- transpose 后 view 失败：待完成。
- reshape 可能复制及别名行为：待完成。
- view 修改影响原数据：待完成。
- CPU/GPU、dtype 转换：待完成。
- 独立解释和复写：待完成。
- 验收结论：未验收。
