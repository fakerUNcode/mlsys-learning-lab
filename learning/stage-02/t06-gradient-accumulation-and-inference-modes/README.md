# T06：梯度累积、计算图与推理模式

状态：故障对照脚本和 CPU（Central Processing Unit，中央处理器）`float64`（64 位双精度浮点数）运行证据已完成。任务记录：[T06 完成记录](../../../plan/completed/T06.md)；实测环境与输出：[验证报告](../../../reports/stage-02/2026-10-09-t06-validation.md)。本章承接 [T05 链式法则与有限差分](../t05-chain-rule-and-finite-differences/README.md)：T05 练习“梯度怎么算”，T06 解释梯度如何被保存、清理、截断，以及何时不需要记录它。

本章的中心问题是：梯度到底帮训练做什么？答案是，梯度告诉优化器损失在当前参数附近的变化方向和敏感程度；优化器再依据梯度和学习率修改参数。T06 用标量 SGD（Stochastic Gradient Descent，随机梯度下降）演示这条路径，并对照常见的状态错误。推理（inference）是用已训练模型产生预测的阶段，不再通过损失更新参数；后半章区分推理时的模块行为、计算图截断和梯度记录开关。

## 运行与阅读

从仓库根目录运行：

```bash
.venv/bin/python learning/stage-02/t06-gradient-accumulation-and-inference-modes/compare_grad_modes.py
```

入口脚本：[compare_grad_modes.py](compare_grad_modes.py)。脚本只用人工构造的小型 CPU Tensor，不读取数据，也不训练真实模型。当前已观察环境为 PyTorch `2.11.0+cu128`（`cu128` 表示该构建带 CUDA 12.8 支持，但本程序仍在 CPU 上运行）、`torch.float64`；模型中的随机层使用随机种子 17。随机种子是伪随机数生成器的初始状态，同一环境重置为同一种子可复现本次掩码；不同设备或软件版本不保证随机序列逐项相同。所有断言通过时，最后打印 `T06 checks passed`；断言是运行时条件检查，失败会抛出 `AssertionError`（断言失败异常）并使程序退出。

阅读顺序：先理解梯度与参数更新，再看 `.grad` 的累积和清理；接着读 `detach()`，然后比较 `train/eval` 与梯度记录模式，最后比较 `no_grad()` 和 `inference_mode()`。每个实验都对应脚本中的同名函数。

## 前置知识

### Tensor、参数与损失

Tensor 是 PyTorch 保存数值的对象；`torch.tensor(1.0)` 创建零维 Tensor，也就是标量。PyTorch 是一个用于张量计算和机器学习的 Python 库，`torch` 是它的 Python 导入名。`nn.Module` 是 `torch.nn`（neural network，神经网络）中的模块基类，神经网络层和完整模型通常由 Module 组成；Module 负责组织子层、参数、缓冲状态以及训练/评估标志。

`nn.Parameter`（神经网络参数对象）是 `torch.Tensor` 的子类。Tensor 是承载标量、向量或更高维数值的容器；Parameter 保留 Tensor 的运算能力，并增加了“赋给 `nn.Module` 属性时自动登记为可训练参数”的模块规则。登记后，它会进入模块的 `_parameters` 参数表，可由 `model.parameters()` 只取参数值，或由 `model.named_parameters()` 取出“名称与参数值”配对。普通 Tensor 赋给模块属性不会因此自动登记。`requires_grad=True` 默认表示在梯度模式开启时为其记录梯度；它本身不保证优化器会更新它，因为优化器还必须显式接收该 Parameter。本脚本直接把标量 Parameter 放入 SGD 的参数列表，未通过模型收集。

普通 Tensor 赋给 Module 的普通属性不会自动登记成 Parameter。若模型需要保存一个不由优化器训练的状态值，可用 `register_buffer(name, tensor)` 登记为 buffer（缓冲状态）。这类持久 buffer 默认随模块的 `state_dict()` 保存，但它们不由优化器按梯度更新。`state_dict`（state dictionary，状态字典）是由名称映射到参数和持久缓冲状态的字典，常用于保存及重新载入模型状态。

参数参与运算后，PyTorch 的 autograd（automatic differentiation，自动微分）系统在梯度模式下记录运算之间的依赖关系，这些依赖组成计算图。前向计算（forward pass）按输入顺序得到预测和损失；反向传播（backward pass）沿记录的依赖关系应用链式法则。损失 `loss` 是衡量预测与目标差距的标量 Tensor，优化时通常希望让它变小。样本（sample）是一次数据输入及其目标；训练（training）是反复用样本计算损失、梯度并更新参数。T06 的标量实验使用参数 `w`、样本输入 `x` 和目标值 `target`：

```text
y = w * x
e = y - target
L = 0.5 * e * e
```

`w` 是待调整的权重，`x` 是一个样本的输入值，`target` 是该样本希望得到的目标值；`y` 是预测值，`e` 是带符号误差，`L` 是损失。损失越小，表示此例中的预测越接近目标。`0.5` 是为了让平方求导时的系数 2 抵消，便于手算；它不改变最小值所在位置。

### 梯度是什么、用来做什么

梯度是目标值对输入或参数的局部变化率。对标量参数来说，导数的正负表示参数略微增大时损失倾向增大还是减小；绝对值表示在当前点附近，对同样大小的参数改动有多敏感。对一组参数来说，梯度中每个位置对应损失对同位置参数的偏导。

本例对 `w` 求导。这里按复合函数链 `w -> y -> e -> L` 拆开求导。每个等号右侧都对应一条具体规则：

```text
dL/de = e       # d(0.5*e*e)/de = 0.5*(2*e) = e，幂函数求导
de/dy = 1       # e=y-target，对 y 求导，target 固定
dy/dw = x       # y=w*x，对 w 求导，x 固定
dL/dw = (dL/de)*(de/dy)*(dy/dw)   # 链式法则，沿 w -> y -> e -> L 相乘
        = e*1*x                    # 代入前三个局部导数
        = (w*x-target)*x           # 代入 e=y-target 和 y=w*x
```

`dL/dw` 是一个标量，因为 `w` 是标量。若 `w` 是矩阵，梯度也是与 `w` 同形状的矩阵，每个元素表示损失对对应权重的偏导。

当 `target=0`、`w=1` 时：

| 样本 `x` | 预测 `y=w*x` | 损失 `L=0.5*y²` | 梯度 `dL/dw=y*x` |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 0.5 | 1 |
| 2 | 2 | 2 | 4 |

例如 `x=1` 时，梯度为正数 1，说明当前点附近增大 `w` 会增大损失。SGD 是 Stochastic Gradient Descent（随机梯度下降）的缩写：训练通常从随机抽取的单个样本或小批次估计梯度，再用该估计更新参数；本脚本为了可复算固定了样本顺序，所以没有随机抽样。SGD 沿梯度的反方向移动；学习率（learning rate）是每步更新的比例，记作 `η`：

```text
w_new = w_old - learning_rate * (dL/dw)   # 梯度下降更新规则
      = 1 - 0.1 * 1                       # 代入旧参数、学习率和本例梯度
      = 0.9                               # 完成乘法和减法
```

更新后损失是 `0.5*(0.9*1)²=0.405`，小于更新前的 `0.5`。这说明梯度为更新提供局部方向，学习率 `0.1` 决定步长。梯度本身不会改参数；步长过大时，更新可能越过低点，不能保证每一步都降低损失。实际模型使用优化器按每个参数的梯度更新权重和偏置。

T05 的标量 `grad_x` 和线性层 `grad_weight`、`grad_bias` 用于验证导数；T05 没有调用优化器。T06 把梯度接到 `optimizer.step()`，但仍使用小型手工例子，而不是训练模型。

### 三种模块对象

这一节先说明 T06 后面会用到的三种 PyTorch 概念：Dropout（随机失活）是正则化层，Parameter（神经网络参数对象）是待更新状态，BatchNorm（Batch Normalization，批归一化）是归一化层。它们所属层级不同：Dropout 和 BatchNorm 是 `nn.Module`（神经网络模块）子类，Parameter 是 `Tensor` 子类。`model.train()` 把模型及其子模块切到训练标志 `training=True`；`model.eval()` 把它们切到评估标志，等价于 `model.train(False)`。这两个方法改变前两种层的行为，不会改变 Parameter 的值或 `requires_grad` 标志，也不会关闭 autograd。autograd 梯度模式另行决定运算是否记录反向图。

**Dropout（随机失活，PyTorch 类名 `nn.Dropout`）** 是一种正则化层，用来降低网络过度依赖固定激活值组合的风险。正则化（regularization）是在训练时加入约束或随机扰动以帮助减少过拟合；过拟合（overfitting）是模型贴合训练样本细节、在新样本上表现变差。Dropout 没有可学习的权重或偏置参数。

`nn.Dropout(p)` 中的 `p` 是训练态下将元素置零的概率，通常 `0 <= p < 1`。它为每个输入元素采样一个伯努利掩码 `m`：以 `p` 的概率取 0，以 `1-p` 的概率取 1。激活值（activation）是层之间传递的中间数值。训练态的计算为：

```text
y = m * x / (1 - p)
```

这里 `x` 是输入激活值，`y` 是输出，`m` 是只含 0/1 的随机掩码；倒置 Dropout 的缩放让多次随机输出的期望（重复随机时的平均值）满足 `E[y]=x`。例如 `p=0.5` 时，某个值要么为 0，要么放大为原来的 2 倍。单次前向不保证恰好一半元素被置零。评估态下该层直接返回 `x`，不采样掩码也不缩放，因此是恒等映射。随机置零与恒等行为由 `module.training` 控制；autograd 梯度模式仍决定该层的运算是否进入反向图。

神经元协同依赖（co-adaptation）指某些单元过度依赖一组固定搭档；训练时随机屏蔽元素会减少这种依赖。这个动机常用“随机让一部分员工休假，团队不能只靠某几个人”来理解，但它不保证每个模型都因此泛化更好。

**BatchNorm（Batch Normalization，批归一化；PyTorch 类名如 `nn.BatchNorm1d`）** 是按特征通道标准化激活值的神经网络层。`BatchNorm1d` 可接收 `(N,C)` 或 `(N,C,L)`：`N` 是批次中的样本数，`C` 是特征/通道数，`L` 是序列位置数；每个通道分别汇总样本（以及存在时的序列位置）来求均值和方差。`BatchNorm2d` 对图像张量 `(N,C,H,W)` 按通道跨样本和空间位置求统计量。批次（batch）是一次共同送入模型计算的一组样本；`H/W` 是图像的高/宽，`L` 是序列位置，存在时它们也参与相应层的通道统计。

```text
x_hat = (x - mean) / sqrt(var + eps)
y = gamma * x_hat + beta
```

`x` 是输入激活值，`mean`/`var` 是当前用于标准化的均值/方差，`eps` 是 epsilon（数值稳定用的小正数）；`gamma`（希腊字母 gamma）是缩放系数，`beta`（希腊字母 beta）是偏移量。默认 `affine=True` 表示使用可学习的缩放/平移仿射变换，因此 gamma/beta 是 Parameter，由梯度和优化器更新；`affine=False` 时没有这两个可学习参数。标准正态分布指均值 0、方差 1 的高斯分布；有限批次、`eps` 和仿射变换意味着输出不保证仍服从这个分布。

默认 `track_running_stats=True` 时，训练态用当前批统计量标准化，并更新 `running_mean`/`running_var` 两个 buffer；评估态用训练期间积累的 running 统计量，不继续更新。其更新形式可理解为 `new_running = (1-momentum)*old_running + momentum*current_batch_stat`。BatchNorm 自己的 `momentum` 控制新批统计写入 running 平均值的比例，与 SGD 的动量不是同一个状态。训练态用于标准化当前批的方差是有偏估计（分母为观测数 `n`）；写入 `running_var` 的方差是无偏估计（分母为 `n-1`）。这里“无偏”是说重复抽取同样大小的样本时，估计值的平均数等于总体方差；分母 `n-1` 用于补偿用样本均值代替未知总体均值带来的偏小。`track_running_stats=False` 时没有运行统计量，训练和评估都会使用当前批统计量。

BatchNorm 可稳定部分网络中的激活值尺度；训练统计量随批次略有变化，也可能带来轻微正则化效果，但收敛速度、可用学习率或泛化是否改善都取决于模型、数据和训练配置。BatchNorm1d 处理 `(N,C)` 且 `N=1` 的训练输入时，每通道只有一个观测值，PyTorch 会因每通道观测数不足而报错；不能把它泛化成所有 batch size 为 1 都会失败，例如 `(1,C,L)` 且 `L>1` 有多个位置可供统计。评估态若使用已保存的运行统计量，则可处理单个样本。

迁移学习是从已有训练权重出发、再适配新任务。冻结 BatchNorm 的 gamma/beta 梯度与停止 running 统计更新是两件事：前者控制可训练参数，后者控制模块处于何种训练/评估行为。对整个模型调用 `eval()` 还会切换 Dropout，所以若只想冻结 BatchNorm 统计，需按模块分别设置状态。本章代码不实例化 BatchNorm；这里解释它和 Dropout、Parameter、buffer 的区别，不把它列为程序验证结果。

三者的控制关系可压缩成下表。梯度模式（grad mode）是 PyTorch 决定是否记录反向计算图的全局/上下文设置；它与模块的 `training` 标志互不替代。

| 对象 | `train()` / `eval()` 控制什么 | 梯度模式控制什么 | 参数或状态 |
| --- | --- | --- | --- |
| Dropout | 训练时随机屏蔽并缩放；评估时恒等 | 决定本次运算是否进入反向图 | 无可学习参数 |
| Parameter | 不改变其值或 `requires_grad` | 与 `requires_grad` 一起决定是否为其记录梯度 | 优化器收到它且 `.grad` 有值时才可能更新 |
| BatchNorm | 默认训练时用当前批统计并更新 buffer；评估时用运行统计 | 决定 gamma/beta 运算是否参与反向图；`no_grad` 本身不停止训练态 buffer 更新 | gamma/beta 是 Parameter；running mean/variance 是 buffer |

因此 `eval()` 解决“层按何种训练/评估规则运行”，`no_grad()` / `inference_mode()` 解决“是否建立反向求导所需的信息”。推理常同时调用 `model.eval()` 与梯度禁用上下文，因为前者让 Dropout/BatchNorm 采用评估行为，后者减少推理不需要的梯度记录；如果只用其中一个，另一个维度仍按原设置工作。

T06 实际模型由 `nn.Sequential`（顺序组合容器）按顺序连接 `nn.Linear(16,16,bias=False)` 和 `nn.Dropout(p=0.5)`。Linear（线性层）用矩阵乘法把输入特征映射到输出特征；`weight` 是映射矩阵，`bias` 是加到每个输出上的偏置向量。PyTorch Linear 的权重形状为 `(out_features,in_features)`，计算为 `input @ weight.T + bias`；`.T` 表示矩阵转置。代码关闭 bias，再用 `torch.eye(16)` 创建 16 阶单位矩阵（主对角线为 1、其他位置为 0），因此 Dropout 前输出与输入相同。这一设置让后续观察到的差异来自 Dropout 或模块模式。

### 计算图、叶子与梯度保存

直接创建并设置 `requires_grad=True` 的 Tensor（或 `nn.Parameter`）通常是**叶子 Tensor**。通过运算得到的结果是非叶子 Tensor。`grad_fn` 是 PyTorch 用来记录“哪个反向运算产生了这个非叶子结果”的属性；例如乘法结果会关联一个乘法反向节点。对标量损失调用 `loss.backward()` 时，autograd 沿计算图反向应用链式法则；叶子参数的梯度默认累加到它的 `.grad` 字段。`.grad` 是保存数值的梯度缓冲区，不是计算图本身，也不是参数值。

本章的 loss 是标量，所以 `backward()` 可把它的上游梯度当作 1，不需要额外输入。对多元素输出调用 `backward()` 时，需要提供该输出对应的梯度权重，autograd 才知道要计算哪个加权和的导数；本章没有这种情形。若只想拿到指定输入的梯度而不使用 `.grad` 缓冲区，可以像 T05 一样调用 `torch.autograd.grad(output, inputs)`；该函数返回梯度，不会像 `backward()` 那样累加到输入的 `.grad` 字段。

**叶子是计算图的"起点"（参数，尤其是工程中需要更新的参数，如本章的w），非叶子是"中途产物"（中间结果）**。PyTorch 会沿图从 loss 一路回溯，把梯度最终只落回叶子上。

#### 判定规则

先说一个反直觉的点——判定不看"是不是算出来的"，看 `requires_grad` 和 `grad_fn`：

| `requires_grad` | 判定                                               |
| --------------- | -------------------------------------------------- |
| `False`         | **一律是叶子**，哪怕它是运算结果（因为压根没进图） |
| `True`          | `grad_fn is None` → 叶子；否则非叶子               |

所以：

```python
a = torch.tensor([1.0])          # 叶子（用户创建，requires_grad=False）
b = a * 2                        # 也是叶子！b.requires_grad=False，grad_fn=None
c = torch.tensor([1.0], requires_grad=True)   # 叶子
d = c * 2                        # 非叶子，grad_fn=<MulBackward0>
```

`grad_fn` 的具体内部名称（如 `MulBackward0`）会随操作和 PyTorch 版本变化；理解它表示“创建此结果的反向运算节点”即可，不应在程序逻辑中依赖这个字符串。

`torch.no_grad()` 下产生的任何结果也全是叶子（没有 `grad_fn`）。

#### 三处实质差异

|                         | 叶子                             | 非叶子                                                    |
| ----------------------- | -------------------------------- | --------------------------------------------------------- |
| `grad_fn`               | `None`                           | 指向产生它的反向函数，如 `<AddBackward0>`                 |
| `backward()` 后 `.grad` | **自动填充并累加**               | 默认仍为 `None`；梯度会经过此节点传回更早节点，但不保存在这个属性中 |
| 能否改 `requires_grad`  | 可以（`x.requires_grad_(True)`） | 报错：`only change requires_grad flags of leaf variables` |

**为什么 `.grad` 默认只填叶子？** 反向传播经过非叶子中间结果时仍会计算并传递其梯度，以便继续求更早输入的导数；默认不把每个中间梯度长期保存在 `.grad` 属性，以免占用额外内存。若调试时确实要读取某个非叶子的梯度，可在 `backward()` **之前**调用 `inner.retain_grad()`；它要求 PyTorch 在反向后保留这个中间 Tensor 的梯度，之后可通过 `inner.grad` 查看。

# 实验过程

![img](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/t06_overview.svg)

## 实验一：累积和参数更新

![实验①梯度累积与两条 SGD 路径对照](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/t06_exp1_accumulation.svg)

### 反向传播为什么会累积

PyTorch 的 `backward()` 默认把新梯度加到叶子参数已有的 `.grad` 上，不会自动覆盖旧值。这样可以把多个 loss 或多个小批次的梯度合起来，再进行一次参数更新；若每批本来就应独立更新，训练循环必须在新一批反向传播前清理旧梯度。

脚本的 `show_gradient_accumulation()` 先固定 `w=1`，连续处理两个样本，但暂时不调用 `optimizer.step()`：

1. 样本 `x=1` 的梯度是 `(1*1-0)*1=1`，第一次 backward 后 `.grad=1`。
2. 参数仍是 `w=1`；样本 `x=2` 的梯度是 `(1*2-0)*2=4`，第二次 backward 把它加上去，所以 `.grad=1+4=5`。
3. `optimizer.zero_grad(set_to_none=True)` 把优化器管理的梯度字段设为 `None`。此时断言检查 `weight.grad is None`；再对 `x=2` 反向一次，字段得到新的梯度 4。

第一段不更新参数，是为了只观察 `.grad` 的累加。如果在两个 backward 之间调用 step，参数会改变，第二次梯度也会在新参数处计算，难以单独看清“累加”这一件事。

### 为什么每批更新前要清零

优化器保存要更新的参数和更新规则。对独立小批次，常见流程是先清理旧梯度，再前向计算损失、反向计算梯度，最后更新参数：

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(inputs)
loss = loss_fn(prediction, target)
loss.backward()
optimizer.step()
```

`zero_grad(set_to_none=True)` 只清理该优化器管理的参数；`backward()` 写入本轮梯度；`step()` 使用当时 `.grad` 中的数值更新参数。清零放在 backward 之后太晚，无法阻止旧值和新值相加。

脚本把“漏清零”和“正确清零”两条路径都从 `w=1`、学习率 `0.1`、样本顺序 `x=1` 再 `x=2` 开始：

| 阶段 | 漏清零路径 | 正确路径 |
| --- | --- | --- |
| 第一批 `x=1` | 梯度 1，更新 `w=1-0.1*1=0.9` | 梯度 1，更新到 `w=0.9` |
| 第二批 `x=2` | 当前参数下新梯度 `(0.9*2)*2=3.6`；旧的 1 仍在，所以 `.grad=1+3.6=4.6` | 先清零；当前批梯度为 3.6，`.grad=3.6` |
| 第二次更新 | `w=0.9-0.1*4.6=0.44` | `w=0.9-0.1*3.6=0.54` |

两条路径的唯一区别是第二批 backward 前有没有清零；因此结果差异能归因于残留梯度。`.grad=4.6` 不是第二批自身的导数，而是两次 backward 累加后的缓冲区数值。

### `zero_grad` 与有意累积

`set_to_none=True` 表示将 `.grad` 设成 `None`，下次 backward 会建立新梯度。设为 `False` 时，已有梯度会被清成数值为零的 Tensor。二者都能清除旧数值，但状态并不完全相同：读取 `.grad` 时 `None` 与零 Tensor 不同；优化器面对 `None` 时可能跳过该参数，面对零梯度时则可能仍执行包含动量或权重衰减等状态的更新。本章的简单 SGD 不包含这些机制，脚本选择 `True` 是为了让清理后的状态可直接断言。

梯度累积也可能是有意策略：多个小批次各自 backward，暂时不清零、不 step，最后执行一次 step。此时 `.grad` 默认是各批损失梯度之和。若每批 loss 是样本损失的平均值，并希望在可分的逐样本损失下等价于更大的平均批次，就要依据每批样本数做正确加权；等大小的累积批次可把每批 loss 除以累积批次数，不等大时要按样本总数加权。BatchNorm、随机 Dropout 等跨样本或带状态行为还可能让微批次计算与一次大批次计算不同。它与“每批都 step 却忘记清零”不同，因为后者既累加旧梯度又多次更新参数。

## 实验二：`detach` 与梯度边界

![img](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/t06_exp2_3_4.svg)

`value.detach()` 返回一个与旧计算图断开的 Tensor。返回值与原 Tensor 共享底层数值存储，因此一方原地改值会影响另一方；但 autograd 不会从 detached Tensor 沿旧路径回传。对 detached Tensor 调用 `.requires_grad_()` 会在该 Tensor 上重新开启梯度跟踪，让它成为一个新的梯度起点，不会恢复旧图。

脚本 `show_detach()` 的状态顺序是：

1. 创建 `source=2`，并令 `requires_grad=True`。
2. 计算 `intermediate=source*3=6`，此时它仍连接 source 的计算图。
3. 执行 `intermediate.detach().requires_grad_()`，得到数值为 6 的新叶子 `detached_leaf`；梯度从这里开始一段新图。
4. 计算 `downstream_loss=detached_leaf²=36` 并 backward。新叶子的梯度为 `2*6=12`，原 `source.grad` 仍是 `None`。

因此 `detach` 适合明确停止梯度向更早阶段传播的场景；若误用在本应训练的路径上，上游参数就收不到后续损失的梯度。本脚本只验证梯度边界，没有对共享存储做原地写值测试。

## 实验三：模块行为与梯度记录是两类开关

`model.train()` 和 `model.eval()` 改变模块的 `training` 状态；具体影响由各子模块定义。Dropout 在训练态随机置零并缩放保留元素，在评估态不再随机屏蔽。BatchNorm 等模块也会根据状态选择统计行为。`eval()` 不会自动关闭 autograd，因此评估态前向仍可能构建计算图并产生梯度。

`torch.no_grad()` 和 `torch.inference_mode()` 控制一段运算是否记录 autograd 信息；它们不替模型调用 `eval()` 或 `train()`。可把几种开关分开看：

| 设置 | 模块行为状态 | 记录反向计算图 | 运算结果之后能否用于新 autograd 图 |
| --- | --- | --- | --- |
| 默认梯度模式 | 保持当前 train/eval 状态 | 若输入/参数需要梯度则记录 | 可以 |
| `with torch.no_grad():` | 保持当前 train/eval 状态 | 不记录 | 普通 Tensor；可在退出区域后作为常量参与新图 |
| `with torch.inference_mode():` | 保持当前 train/eval 状态 | 不记录，并关闭更多 autograd 跟踪工作 | inference Tensor 有限制；若后续反向要保存它，会报错 |
| `model.eval()` | 切换成评估行为 | 不负责关闭 | 仍按当前梯度模式决定 |

脚本 `make_probe_model()` 建立 `Linear(16,16)` 加 `Dropout(0.5)`，并把线性层权重设为单位矩阵。输入是 16 个 1，因此 Dropout 前输出仍是 16 个 1。训练态保留元素按 `1/(1-0.5)=2` 缩放，故输出元素为 0 或 2；评估态 Dropout 等于恒等映射，输出仍全为 1。

`show_model_mode_and_grad_modes()` 依次验证：

- `model.train()`：Dropout 正常屏蔽，但前向仍处于默认梯度模式，输出 `requires_grad=True`。
- `model.eval()`：Dropout 不再屏蔽，输出等于输入；由于未禁用 autograd，调用 backward 后权重和输入都能收到梯度。
- 再调用 `model.train()` 并把前向放进 `torch.no_grad()`：Dropout 继续屏蔽，但输出不记录梯度。固定种子 17 让本次训练态和 no-grad 训练态使用同一掩码，脚本断言两次零元素数量相同。

评估态这一步调用 `model.zero_grad(set_to_none=True)` 清理模型登记参数的梯度；输入 `values` 不属于模型参数，所以脚本还单独把 `values.grad` 设为 `None`。两者都在本次 backward 前清掉旧观察值，之后再检查新图是否分别连到权重和输入。

验证或测试模型时，若希望行为像评估（关闭 Dropout、停止更新 BatchNorm 运行统计），调用 `model.eval()`；若同时不需要反向图，再套 `torch.no_grad()` 或 `torch.inference_mode()`。两件事解决不同问题，常一起使用但不能互相替代。

## 实验四：`no_grad` 和 `inference_mode`

两种模式都不为区域内运算构建反向图，但产生的 Tensor 后续用途不同。

- `no_grad` 中创建的是普通 Tensor。退出区域后，它可作为常量输入新的梯度计算；区域内的源计算图不会因此恢复。
- `inference_mode` 会创建带 inference 标记的 Tensor，并跳过更多 autograd 跟踪工作。若后续运算要为其他可训练输入求导，且 backward 必须保存该 inference Tensor，PyTorch 会拒绝这次计算。若纯推理数据不再与 autograd 交互，推理模式更合适。
- 若确实要把 inference Tensor 的数值送入后续梯度图，可在 inference mode 外 `.clone()`，复制成普通 Tensor；复制值不会恢复它原来的历史梯度路径。

脚本 `show_no_grad_and_inference_mode()` 使用 `source=[1,2]` 和 `weight=[3,4]`。先在 no-grad 中计算 `features=source*2=[2,4]`。之后计算 `sum(features*weight)`：对 weight 求导需要保存 features，结果梯度是 `[2,4]`；source 梯度仍为 None。接着在 inference mode 生成同值 Tensor，并尝试相同乘法反向；本例的 backward 需要保存 features，因而触发 `RuntimeError`。在区域外 clone 后再次乘以可训练 weight，则能够得到 `[2,4]` 梯度。

这只证明本例这个需要保存输入的乘法会触发限制，不表示 inference Tensor 的每种运算都会失败。脚本也没有测量两种模式的性能差异，不对实际速度做结论。

## 函数职责与执行路径

直接运行脚本时，Python 将 `__name__` 设为 `"__main__"`，入口保护随即调用 `main()`；作为模块导入时不会自动启动实验。源码的 `from __future__ import annotations` 让类型注解延迟求值，`import torch` 引入 PyTorch，`from torch import nn` 用短名称 `nn` 访问神经网络模块。`DTYPE=torch.float64` 统一实验数值类型；Python 的大写名字只是常量命名约定，不会锁定变量。

| 执行顺序 | 函数 | 输入与主要状态变化 | 输出或证据 |
| --- | --- | --- | --- |
| 1 | `main()` | 打印 PyTorch 版本与 dtype，按顺序调用下列函数 | 断言全通过后打印 `T06 checks passed` |
| 2 | `show_gradient_accumulation()` | 创建标量参数、SGD 优化器；比较累加、清零和两条更新路径 | `.grad` 为 1、5、None、4；最终参数为 0.44 和 0.54 |
| 3 | `show_detach()` | 创建 source、intermediate 和 detached 新叶子，再反向 | 新叶子梯度 12；source 梯度 None |
| 4 | `make_probe_model()` 与 `show_model_mode_and_grad_modes()` | 建恒等线性层加 Dropout，切换 train/eval 和 no-grad | 观察 Dropout 输出、requires_grad 及输入/权重梯度 |
| 5 | `show_no_grad_and_inference_mode()` | 分别生成普通 no-grad Tensor 和 inference Tensor，测试后续反向 | no-grad 成功；特定 inference 用法报错；clone 后成功 |

`scalar_loss(weight, x, target)` 是第一组实验共用的损失函数，输入一个标量权重 Tensor、Python 浮点样本和目标，返回保留对 weight 计算图关系的标量 loss。`.item()` 把单元素 Tensor 转成 Python 数值以便打印；`.tolist()` 将向量梯度转为 Python 列表。`assert` 把预期状态变成运行期检查，错误时抛出 `AssertionError`，因此检查失败不会被当作通过。

## 设计与证据

| 设计选择 | 为什么这样做、提供什么保证 | 省略或替换的后果 | 对应证据与边界 |
| --- | --- | --- | --- |
| 累积演示中不调用 step | 固定 `w=1`，把梯度相加单独显示为 `1+4=5` | 若同时更新，后续梯度也因参数改变而改变，不能单独观察累积 | `show_gradient_accumulation()` 断言 1、5、4；单标量例子 |
| 故障和正确路径使用同一初值、样本、学习率 | 让清零时机成为主要变量 | 多个条件一起变化时无法把 0.44 与 0.54 的差异归因于漏清零 | 脚本打印并断言两种最终权重；未覆盖动量、权重衰减等优化器状态 |
| 对独立批次在 backward 前清零 | 避免当前梯度叠加前批梯度 | 漏清零时 step 会使用过期与当前梯度的和 | 运行值 4.6 对比 3.6；有意梯度累积则需延后 step 并正确缩放 loss |
| 用 `detach()` 后重新开启梯度 | 明确展示旧图被切断而新叶子仍可训练 | 只看 requires_grad=False，无法验证新梯度起点和原路径的区别 | 新叶子 grad=12、source grad=None；没有验证共享存储的原地修改 |
| 用 Dropout 对照 train/eval | 让模块模式改变带来可观察的输出差异 | 只用无状态线性层难以观察 eval 的作用 | 固定种子下 train 有 10 个零，eval 输出等于输入；不覆盖 BatchNorm |
| 分别测试 no-grad 与 inference Tensor | 展示二者后续参与新 autograd 图的限制不同 | 只检查 `requires_grad` 无法看出 Tensor 类型和后续可保存性差别 | 特定乘法报错、clone 后梯度成功；不代表所有 inference 操作报错，也没有性能测试 |

这些证据分为三类：公式与手算解释数值为什么应如此；脚本断言验证当前 PyTorch 环境中指定路径的数值和状态；验收记录保存实际运行结果。脚本没有编译期验证（Python 是运行时语言），也没有验证任意模型、设备、优化器或所有算子组合。

## 适用范围

本实验只运行 CPU `float64` 小 Tensor，不覆盖 GPU（Graphics Processing Unit，图形处理器）/CUDA（Compute Unified Device Architecture，NVIDIA 的 GPU 编程平台）、混合精度（在一次计算中使用多种浮点精度）、梯度缩放（低精度训练时缩放损失以降低梯度下溢风险，并在更新前还原比例）、梯度裁剪（限制单个梯度值或梯度范数以约束更新幅度；范数是把多个梯度分量合成的整体大小，例如平方和再开平方）、分布式训练（多进程或多设备协同计算并同步梯度）、真实数据集、复杂优化器状态或运行性能。inference Tensor 的报错仅适用于后续反向确实需要保存它的本例。实际训练时应先决定每次何时更新参数，再安排清零、backward、loss 缩放和 step。

## 参考资料

- R01：[Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：损失、梯度、优化器更新的训练循环。
- R03：[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：计算图、梯度模式、no-grad、inference mode、detach 与 eval 的区别。
- R03：[Zeroing out gradients](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)：`.grad` 默认累加及清零。
- PyTorch API：[Optimizer.zero_grad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)：`None` 梯度与零 Tensor 的行为区别。
- PyTorch API：[Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)：训练态置零/缩放与评估态恒等映射。
- PyTorch 文档：[Parameters, buffers, and modules](https://docs.pytorch.org/docs/stable/notes/modules.html)：Parameter 自动登记、buffer 与模块状态。
- PyTorch API：[BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)：输入形状、仿射参数、运行统计和训练/评估规则。
