# T08：单头缩放点积 Attention

本章把一组位置向量变成另一组位置向量：每个位置先给其他位置打相关性分数，再按分数加权汇总信息。程序入口是 [attention_reference.py](attention_reference.py)，从仓库根目录运行：

```bash
.venv/bin/python learning/batches/b03-attention/t08-scaled-dot-product-attention/attention_reference.py
```

脚本在 CPU（Central Processing Unit，中央处理器）`float64`（64 位双精度浮点数）上用普通矩阵乘法、缩放和稳定 softmax 手写 Attention，再同 PyTorch SDPA 比较。它是小输入 reference，不是高性能 kernel。T08 承接 [T02 Tensor 形状与批量矩阵乘法](../../b01-environment-tensor-layout/t02-tensor-basics/README.md)、[T03 转置与布局](../../b01-environment-tensor-layout/t03-tensor-layout-and-storage/README.md) 和 [T05 链式法则](../../b02-gradients-training-loop/t05-chain-rule-and-finite-differences/README.md)。

## 前置基础

### Tensor 与形状

Tensor 是 PyTorch 保存多维数值的对象。形状 `[B,S,D]` 中，`B` 是 batch（批次）中的样本数，`S` 是序列位置数，`D` 是每个位置的特征数。可以想成一摞 `B` 本点名册；每本有 `S` 行学生，每行有 `D` 个描述学生的数字。轴顺序很重要：`Q[b,i]` 是第 b 个样本的第 i 个位置向量。

### 矩阵与转置

矩阵乘法 `A @ B` 把 A 的列轴与 B 的行轴配对相乘再求和。`[m,k] @ [k,n]` 得到 `[m,n]`。转置（transpose）交换矩阵轴；`K.transpose(-2,-1)` 把 `[B,S,D]` 改成 `[B,D,S]`，让 `Q @ Kᵀ` 得到 `[B,S,S]`。类比把“按学生列特征”翻成“按特征列学生”，从而让同一种特征相乘。

### 注意力与 QKV

Attention（注意力）按相关程度汇总信息。Query（查询，Q）表示当前位置想找什么；Key（键，K）是候选位置用于匹配的线索；Value（值，V）是候选位置提供的内容。生活例子：做题时题目（Q）去查资料目录关键词（K），再按匹配程度阅读对应页面内容（V）。目录决定看哪里，页面内容决定拿到什么。Self-attention（自注意力）表示 Q/K/V 来自同一段输入的不同投影；本脚本从投影后的向量开始，不含投影层。单头表示只做一套匹配与汇总。

### 点积与缩放

点积（dot product）是两个等长向量逐项相乘再求和；它可表示向量匹配程度。特征维 D 越大，累加项越多，分数典型幅度可能变大；softmax 会令大分数过度占优。Scaled Dot-Product Attention（缩放点积注意力）用 `sqrt(D)` 除分数。原论文 §3.2.1 的理由是：若 Q/K 分量独立、均值 0、方差 1，每个乘积项的均值为 0、方差为 1；D 项相加的方差为 D；除以 `sqrt(D)` 后方差约回到 1。具体推导见下方。生活例子：比较长短不同的作文，先按字数归一化，避免项目多被误当成匹配好。这个解释依赖前述统计假设。

### Softmax 稳定性

Softmax（归一化指数函数）对一行分数取指数，再除以指数总和，得到非负且总和为 1 的权重。它像把一行候选资料的相关分变成总计 100% 的阅读时间比例。大正数直接取指数可能溢出；稳定做法先减本行最大值 m：

```text
softmax(x) = exp(x - m) / sum_j exp(x_j - m)
m = max_j(x_j)
```

每项同减 m 不改变比例，因为分子、分母都乘相同的 `exp(-m)`；最大指数变成 1。类比所有物品一起减去同一基准重量，彼此比例不变，但秤不再超量程。

### SDPA 与 FLOPs

SDPA 是 `scaled_dot_product_attention` 的缩写，即 PyTorch 的缩放点积注意力函数。其输入布局含 head（注意力头）轴 `[B,H,S,D]`；本实验单头令 H=1。调用参数 `attn_mask=None` 不屏蔽位置，`is_causal=False` 允许访问后续位置，`dropout_p=0.0` 关闭随机失活，`scale=None` 使用默认 `1/sqrt(D)`。对照时关闭 dropout 才能避免随机差异。这里不是生成模型使用的因果 Attention。

FLOPs（floating-point operations，浮点运算次数）按乘、加各计一次。矩阵乘法 `[m,k]@[k,n]` 约需 `2mkn` FLOPs；QKᵀ 与权重乘 V 各约 `2BS²D`，总计约 `4BS²D`，另有约 `O(BS²)` 的 softmax 逐元素运算。`scores` 和 `weights` 都是 `[B,S,S]`，各占 `B*S*S*e` 字节，e 是每元素字节数。例：`B=2,S=5,D=8,float64` 时，两次矩阵乘法约 1600 FLOPs，一个矩阵占 `2*5*5*8=400` 字节，两个共 800 字节。未计 Q/K/V、输出、临时量、自动微分保存值和分配开销。

大写 **D** 是模型总维度，小写 **d** 是单头维度，关系是 **D = H × d**。

一句话：**d 是拆开算时的单头维度，D 是合起来算时的总维度，两者差一个头数 H。** 你例子里如果 H=1，那 d 就等于 D，两个符号就一样了。

#### 1. 标准矩阵乘法 FLOPs 公式

两个矩阵 $[m,k]\times[k,n]$ 需要 $m\times n$ 个输出元素，每个元素做 $k$ 次乘加（即 $k$ 次乘法 + $k$ 次加法 $\approx 2k$ 个 FLOP），所以总 FLOPs $\approx 2mkn$。

#### 2. QKᵀ 的 FLOPs

对==单个头、单个样本==：

$Q\in\mathbb R^{S\times d}$，$K\in\mathbb R^{S\times d}$，计算 $QK^\top\in\mathbb R^{S\times S}$。

这是 $[S,d]\times[d,S]$，对应 $m=S,\ k=d,\ n=S$，所以单头单样本 FLOPs：

$$2\cdot S\cdot d\cdot S = 2S^2 d$$

有 $H$ 个头、$B$ 个样本：

$$B\cdot H \cdot 2S^2 d = 2 B S^2 (Hd) = 2 B S^2 D$$

这就是 **QKᵀ ≈ $2BS^2D$**（忽略 softmax 等，只算矩阵乘）。

#### 3. 权重（注意力分数）乘 V 的 FLOPs

注意力权重 $A\in\mathbb R^{S\times S}$（softmax 后的分数），乘 $V\in\mathbb R^{S\times d}$：

$[S,S]\times[S,d]$，对应 $m=S,\ k=S,\ n=d$，单头单样本 FLOPs：

$$2\cdot S\cdot S\cdot d = 2S^2 d$$

同样乘上 $B\cdot H$：

$$B H \cdot 2S^2 d = 2B S^2 D$$

所以 **AV（即 attn_weights @ V）≈ $2BS^2D$**。

## 公式推导

### 目标和形状

输入 Q/K/V 均为 [B,S,D]。本练习要求三个输入同形状；输出也是 [B,S,D]。每个样本独立计算，位置之间可以互相读取。真实模型通常从同一个输入经三组不同可学习线性层得到 Q/K/V；本脚本不包含投影。

| 阶段 | 形状 | 含义 |
| --- | --- | --- |
| 输入 Q/K/V | [B,S,D] | 查询、匹配线索、内容向量 |
| 点积与缩放分数 | [B,S,S] | 每个 query 对所有 key 的分数 |
| softmax 权重 | [B,S,S] | 每个 query 分给各 key 的概率 |
| 输出 | [B,S,D] | 权重汇总后的 value 向量 |

### 四步计算

1. **点积。** 固定样本 b、查询位置 i、键位置 j；遍历特征 d，把对应分量相乘后相加：
   $$
   r_{bij}
   \overset{\text{点积定义}}{=}
   \sum_{d=1}^{D} Q_{bid}K_{bjd}
   $$

   Q 的形状是 [B,S,D]，转置 K 的最后两个轴后形状是 [B,D,S]。矩阵乘法沿 D 累加，结果的两个位置轴分别来自 Q 和 K，因此分数矩阵形状为 [B,S,S]。这对应代码中的 query 与转置后的 key 相乘。

2. **缩放。** 每个点积分数除以特征维平方根：
   $$
   s_{bij}
   \overset{\text{缩放点积定义}}{=}
   \frac{r_{bij}}{\sqrt{D}}
   $$

   论文《Attention Is All You Need》第 3.2.1 节用方差解释这个缩放。下面逐步推导其动机。假设 Q 和 K 的每个分量相互独立、均值为 0、方差为 1；这只是推导假设，不保证训练中的实际向量满足它们。暂时省略固定的 b、i、j，令：

   $$
   X_d
   \overset{\text{定义单项乘积}}{=}
   Q_dK_d
   $$

   独立变量乘积的期望可以拆成两个期望的乘积：

   $$
   \mathbb{E}[X_d]
   \overset{\text{代入 }X_d\text{ 定义}}{=}
   \mathbb{E}[Q_dK_d]
   $$

   $$
   \mathbb{E}[Q_dK_d]
   \overset{\text{独立变量乘积期望}}{=}
   \mathbb{E}[Q_d]\mathbb{E}[K_d]
   $$

   $$
   \mathbb{E}[Q_d]\mathbb{E}[K_d]
   \overset{\text{代入零均值假设}}{=}
   0\cdot 0
   $$

   $$
   0\cdot 0
   \overset{\text{乘法计算}}{=}
   0
   $$

   再求乘积项的二阶矩。由方差定义，二阶矩等于方差加均值的平方。对 Q：

   $$
   \mathbb{E}[Q_d^2]
   \overset{\text{方差定义}}{=}
   \operatorname{Var}(Q_d)+\mathbb{E}[Q_d]^2
   $$

   $$
   \operatorname{Var}(Q_d)+\mathbb{E}[Q_d]^2
   \overset{\text{代入单位方差和零均值}}{=}
   1+0^2
   $$

   $$
   1+0^2
   \overset{\text{算术计算}}{=}
   1
   $$

   对 K 使用相同定义和假设：

   $$
   \mathbb{E}[K_d^2]
   \overset{\text{方差定义}}{=}
   \operatorname{Var}(K_d)+\mathbb{E}[K_d]^2
   $$

   $$
   \operatorname{Var}(K_d)+\mathbb{E}[K_d]^2
   \overset{\text{代入单位方差和零均值}}{=}
   1+0^2
   $$

   $$
   1+0^2
   \overset{\text{算术计算}}{=}
   1
   $$

   利用 Q 与 K 的独立性：

   $$
   \mathbb{E}[X_d^2]
   \overset{\text{代入 }X_d\text{ 定义}}{=}
   \mathbb{E}[Q_d^2K_d^2]
   $$

   $$
   \mathbb{E}[Q_d^2K_d^2]
   \overset{\text{独立变量乘积期望}}{=}
   \mathbb{E}[Q_d^2]\mathbb{E}[K_d^2]
   $$

   $$
   \mathbb{E}[Q_d^2]\mathbb{E}[K_d^2]
   \overset{\text{代入刚求出的二阶矩}}{=}
   1\cdot 1
   $$

   $$
   1\cdot 1
   \overset{\text{乘法计算}}{=}
   1
   $$

   用方差定义计算 X_d 的方差：

   $$
   \operatorname{Var}(X_d)
   \overset{\text{方差定义}}{=}
   \mathbb{E}[X_d^2]-\mathbb{E}[X_d]^2
   $$

   $$
   \mathbb{E}[X_d^2]-\mathbb{E}[X_d]^2
   \overset{\text{代入二阶矩为 1、均值为 0}}{=}
   1-0^2
   $$

   $$
   1-0^2
   \overset{\text{算术计算}}{=}
   1
   $$

   原始点积由 D 个相互独立的乘积项相加。独立随机变量求和时，方差可以相加：

   $$
   \operatorname{Var}\left(\sum_{d=1}^{D}X_d\right)
   \overset{\text{独立项方差可加}}{=}
   \sum_{d=1}^{D}\operatorname{Var}(X_d)
   $$

   $$
   \sum_{d=1}^{D}\operatorname{Var}(X_d)
   \overset{\text{每项方差为 1}}{=}
   \sum_{d=1}^{D}1
   $$

   $$
   \sum_{d=1}^{D}1
   \overset{\text{共 D 项}}{=}
   D
   $$

   常数 c 乘随机量时，方差乘以 c 的平方。代入缩放常数：

   $$
   \operatorname{Var}\left(
   \frac{\sum_{d=1}^{D}X_d}{\sqrt{D}}
   \right)
   \overset{\text{常数倍方差规则}}{=}
   \frac{\operatorname{Var}\left(\sum_{d=1}^{D}X_d\right)}
   {(\sqrt{D})^2}
   $$

   $$
   \frac{\operatorname{Var}\left(\sum_{d=1}^{D}X_d\right)}
   {(\sqrt{D})^2}
   \overset{\text{代入未缩放方差为 D}}{=}
   \frac{D}{(\sqrt{D})^2}
   $$

   $$
   \frac{D}{(\sqrt{D})^2}
   \overset{\text{平方根定义}}{=}
   \frac{D}{D}
   $$

   $$
   \frac{D}{D}
   \overset{D>0\text{ 时约分}}{=}
   1
   $$

   所以在上述假设下，未缩放点积的方差随 D 增大；除以平方根后，方差回到 1。分数不容易仅因特征数较大而变大，softmax 也较不容易被极端分数压成近似 one-hot 分布。推导解释设计动机，不是对任意模型激活值分布的保证。

3. **稳定 softmax。** 固定样本 b 和查询位置 i，只沿键位置 j 归一化。先取该行最大分数：
   $$
   m_{bi}
   \overset{\text{该行最大值的定义}}{=}
   \max_j s_{bij}
   $$

   从每个分数减去行最大值，求指数：

   $$
   a_{bij}
   \overset{\text{平移后的指数定义}}{=}
   \exp(s_{bij}-m_{bi})
   $$

   再除以本行指数之和，得到概率权重：

   $$
   w_{bij}
   \overset{\text{softmax 归一化定义}}{=}
   \frac{a_{bij}}{\sum_{r=1}^{S}a_{bir}}
   $$

   减最大值不改变 softmax 结果。逐步看分子、分母共同出现的因子：

   $$
   \frac{\exp(s_{bij}-m_{bi})}
   {\sum_r\exp(s_{bir}-m_{bi})}
   \overset{\text{指数差转为乘积}}{=}
   \frac{\exp(s_{bij})\exp(-m_{bi})}
   {\sum_r\exp(s_{bir})\exp(-m_{bi})}
   $$

   $$
   \frac{\exp(s_{bij})\exp(-m_{bi})}
   {\sum_r\exp(s_{bir})\exp(-m_{bi})}
   \overset{\text{分子分母约去共同非零因子}}{=}
   \frac{\exp(s_{bij})}{\sum_r\exp(s_{bir})}
   $$

   每行权重非负，且沿 j 求和为 1，因为每个指数项都除以这些指数项的总和。减最大值后，指数的输入不大于 0，可避免大正数指数导致数值溢出。分母只遍历键位置 r，不能把 batch、query 或特征轴混进归一化。

4. **加权求和。** 对每个输出位置 i 和特征 d，用注意力权重混合所有键位置的 value：

   $$
   y_{bid}
   \overset{\text{注意力输出定义}}{=}
   \sum_{j=1}^{S}w_{bij}V_{bjd}
   $$

   j 被求和消去，留下每个样本、每个 query 位置的一条 D 维向量，因此输出形状为 [B,S,D]。权重集中在一个位置时，输出接近那个位置的 V；权重分散时，输出是多个 V 的混合。

#### 缩放和稳定softmax的区别

缩放（Scaling）是在进入 Softmax 之前，通过数学手段**抑制输入值的方差**，防止梯度消失；而稳定 Softmax（Stable Softmax）是在 Softmax 计算过程中，通过数值技巧**平移输入值**，防止浮点数溢出（Overflow）。

为了让你更直观地对比，我整理了以下表格：

| 维度                 | 缩放（Scaling / $\frac{1}{\sqrt{d_k}}$）                     | 稳定 Softmax（Stable Softmax / 减去最大值）                  |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **作用阶段**         | **Softmax 之前**（点积计算后）                               | **Softmax 计算之中**（求指数 $e^x$ 之前）                    |
| **核心操作**         | 将 $QK^T$ 的结果除以 $\sqrt{d_k}$                            | 将输入向量减去该向量的最大值 $\max(x)$                       |
| **解决的数学问题**   | **方差爆炸**：当维度 $d_k$ 很大时，点积结果的方差会随之变大，导致部分值过大，使得 Softmax 输出趋近于 One-hot（即退化为几乎只有 1 和 0），造成梯度消失。 | **数值溢出**：如果不减最大值，直接对较大的正数求指数 $e^x$，很容易超出浮点数（如 float16/float32）的最大表示范围，变成 `inf`（无穷大）。 |
| **类比理解**         | **“调音台”**：把整体音量（方差）降到一个合理的基准水平，防止某些乐器声音过大掩盖其他声音。 | **“平移坐标”**：把所有人的身高都减去最高的那个人的身高，确保没有人“高”到超出测量仪器的量程。 |
| **是否改变分布形状** | **是**，改变了数据的尺度和相对比例（但保持相对大小关系）。   | **否**，数学上严格等价。减去常数不改变最终的归一化概率分布。 |

#### 为什么 one-hot 会导致梯度消失？

核心原因一句话：**Softmax 输出接近 one-hot 时，反向传播给 $Q$ 和 $K$ 的梯度几乎为零，参数几乎不更新。**

Softmax 的梯度不是直接对输出求导，而是对**输入 logits** 求导。我们先从 Softmax 的定义出发，推导这两个关键梯度公式。

设向量 $\mathbf{z}=(z_1,\dots,z_S)$，第 $i$ 个分量的 Softmax 输出为 $p_i = \frac{e^{z_i}}{\sum_{k=1}^S e^{z_k}}$。记分母 $Z=\sum_{k=1}^S e^{z_k}$，则 $p_i = e^{z_i}/Z$。

**情形一：对同一个 $i$ 求导 $\frac{\partial p_i}{\partial z_i}$**
使用商的求导法则（注意分母 $Z$ 也依赖 $z_i$，且 $\frac{\partial Z}{\partial z_i}=e^{z_i}$）：
$$
\frac{\partial p_i}{\partial z_i}
= \frac{\partial}{\partial z_i}\left(\frac{e^{z_i}}{Z}\right)
= \frac{e^{z_i}Z - e^{z_i}e^{z_i}}{Z^2}
= \frac{e^{z_i}}{Z} - \left(\frac{e^{z_i}}{Z}\right)^2
= p_i - p_i^2
= p_i(1 - p_i)
$$

**情形二：对不同下标 $j\neq i$ 求导 $\frac{\partial p_i}{\partial z_j}$**
此时分子 $e^{z_i}$ 不含 $z_j$，只有分母 $Z$ 依赖 $z_j$（$\frac{\partial Z}{\partial z_j}=e^{z_j}$）：
$$
\frac{\partial p_i}{\partial z_j}
= \frac{\partial}{\partial z_j}\left(e^{z_i}Z^{-1}\right)
= e^{z_i} \cdot (-1)Z^{-2} \cdot e^{z_j}
= -\frac{e^{z_i}e^{z_j}}{Z^2}
= -\frac{e^{z_i}}{Z} \cdot \frac{e^{z_j}}{Z}
= -p_i p_j
$$

统一写成雅可比矩阵形式，即：
$$
\frac{\partial p_i}{\partial z_j} =
\begin{cases}
p_i(1-p_i), & i=j\\[4pt]
-p_i p_j, & i\neq j
\end{cases}
= p_i(\delta_{ij}-p_j)
$$

当输出接近 one-hot 时，假设第 $k$ 个位置的概率 $p_k \approx 1$，其他所有位置 $p_j \approx 0$。代入上述推导出的两个式子：

- $\frac{\partial p_k}{\partial z_k} = 1 \times (1 - 1) = 0$
- 对所有 $j \neq k$：$\frac{\partial p_k}{\partial z_j} = -1 \times 0 = 0$
- 对其他位置的 $p_j \approx 0$，它们对 $z$ 的梯度也是 $0 \times (1 - 0) = 0$

**所有梯度都趋近于零。** 这意味着无论损失函数怎么变，Softmax 之前的 $QK^T$ 分数几乎接收不到梯度信号，$Q$ 和 $K$ 的参数更新就停滞了。

##### 结合注意力机制的具体链路

注意力权重的梯度要一路反传到 $Q$ 和 $K$。以 $Q$ 为例，梯度链路是：

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial \text{Output}} \cdot \frac{\partial \text{Output}}{\partial W} \cdot \frac{\partial W}{\partial \text{scores}} \cdot \frac{\partial \text{scores}}{\partial Q}$$

其中 $W$ 是注意力权重矩阵（Softmax 输出），$\frac{\partial W}{\partial \text{scores}}$ 就是上面推导出的 $p_i(1-p_i)$ 和 $-p_i p_j$。当 $W$ 接近 one-hot 时，这一项整体趋近于零矩阵，梯度传到 $\text{scores}$ 就几乎断了，更别说继续传到 $Q$ 和 $K$。

##### 直觉类比

Softmax 就像一场**赢家通吃的选举**。正常情况大家票数接近，落选者还有动力改进；一旦某人得票 99.9%，其他人得票 0.1%，落选者觉得"怎么努力都没用"（梯度为零），获胜者也觉得"我已经稳了不用改"（梯度也为零）。系统就固化了，学不动了。

##### 回到缩放的作用

未缩放时，$QK^T$ 的方差随 $d_k$ 线性增长，大方差让 Softmax 输入的绝对值很大，输出就更容易被推向 one-hot。除以 $\sqrt{d_k}$ 把方差拉回 1，Softmax 输出更"平滑"，梯度流通更顺畅，训练才稳定。

### 手算例子

令 B=1、S=2、D=2，并取 Q=K 为二阶单位矩阵。每个 query 与同位置 key 的点积是 1，与另一位置 key 的点积是 0，所以原始分数矩阵为二阶单位矩阵。缩放后，对角线上的值约为 0.7071。第一行 softmax 约为 [0.8044, 0.1956]，第二行为 [0.1956, 0.8044]。取 V 的第一行为 [10,0]、第二行为 [0,20]，第一个输出位置的两个特征分别按权重求和：

$$
y_{1,1,1}
\overset{\text{加权求和定义}}{=}
0.8044\cdot 10+0.1956\cdot 0
$$

$$
y_{1,1,1}
\overset{\text{代入并计算}}{=}
8.044
$$

$$
y_{1,1,2}
\overset{\text{加权求和定义}}{=}
0.8044\cdot 0+0.1956\cdot 20
$$

$$
y_{1,1,2}
\overset{\text{代入并计算}}{=}
3.912
$$

因此第一个输出位置约为 [8.044, 3.912]。这展示输出按相关度混合内容，而非只复制一个位置。

## 程序讲解

本节严格按 attention_reference.py 的执行顺序，把矩阵写出来，并跟踪每一步的形状。例子取 B=1、S=4、D=3：一个样本有 4 个位置，每个位置有 3 个特征。下文中的示例分数用于解释矩阵轴，不是脚本随机输入的实测结果。

### ① 点积

程序输入的 query、key、value 形状都是 [B,S,D]。先看一个样本的 Q；每行是一处位置的 3 个特征：

    Q:
    [[q₀₀, q₀₁, q₀₂],
     [q₁₀, q₁₁, q₁₂],
     [q₂₀, q₂₁, q₂₂],
     [q₃₀, q₃₁, q₃₂]]

K 也是 4 行、每行 3 个特征。程序先转置 K 的最后两个轴：

    key.transpose(-2, -1)

形状从 [1,4,3] 变成 [1,3,4]。转置只交换轴的排列，不改数值。然后执行：

    raw_scores = torch.matmul(query, key.transpose(-2, -1))

形状变化是 [1,4,3] @ [1,3,4] → [1,4,4]。矩阵乘法把中间相同长度的特征轴配起来逐项相乘求和，保留 Q 的位置轴和 K 的位置轴。对固定的样本 b、query 位置 i、key 位置 j：

$$
\text{raw\_scores}_{bij}
\overset{\text{矩阵乘法定义}}{=}
\sum_{d=1}^{3}Q_{bid}K_{bjd}
$$

因此每一行是一个 query 对所有 key 的分数，每一列是一个 key 被所有 query 比较的分数：

| query / key | key 0 | key 1 | key 2 | key 3 |
| --- | --- | --- | --- | --- |
| query 0 | s₀₀ | s₀₁ | s₀₂ | s₀₃ |
| query 1 | s₁₀ | s₁₁ | s₁₂ | s₁₃ |
| query 2 | s₂₀ | s₂₁ | s₂₂ | s₂₃ |
| query 3 | s₃₀ | s₃₁ | s₃₂ | s₃₃ |

例如第 2 行第 1 列表示 query 位置 2 与 key 位置 1 的点积。完整 batch 形状是 [B,S,S]；上表省略了最外层 batch 轴，只展开一个样本。

### ② 缩放

接着程序用特征数 D 的平方根缩放每个点积分数：

    scores = raw_scores / math.sqrt(feature_size)

形状仍是 [B,S,S]。本例 D=3，所以：

$$
\text{scores}_{bij}
\overset{\text{缩放点积定义}}{=}
\frac{\text{raw\_scores}_{bij}}{\sqrt{3}}
$$

raw_scores 是未缩放的点积；scores 是送进 softmax 的 logits（未归一化分数）。除以平方根用于控制分数随特征维增长的尺度，避免 softmax 过早变得极端、让大多数权重接近 0 而梯度变小。它不是简单地“防止梯度消失”的通用保证。

为看清后续如何按行处理，假设某个样本的 scores 如下：

| query / key | key 0 | key 1 | key 2 | key 3 |
| --- | ---: | ---: | ---: | ---: |
| query 0 | 2 | 0 | 1 | 3 |
| query 1 | -1 | 4 | 2 | 0 |
| query 2 | 1 | 1 | 0 | -2 |
| query 3 | 0 | 3 | 2 | 1 |

### ③ 稳定 softmax

attention_reference 把 scores 传给 stable_softmax；该函数的参数名是 logits，所以函数内的 logits 就是上表的缩放分数。调用流程是：

    weights = stable_softmax(scores)

函数沿最后一维取最大值：

    row_max = logits.max(dim=-1, keepdim=True).values

对于 [B,S,S]，三个轴依次是 batch、query、key。最后一轴是 key，因此对 dim=-1 求最大值，就是固定一个样本和一个 query 行，只比较该行所有 key 的分数；不同 query 行和不同样本不会混在一起。上表示例逐行最大值依次是 3、4、1、3。

keepdim=True 保留被归约的 key 轴，只把长度 S 变成 1，因此 row_max 的形状从 [1,4,4] 变成 [1,4,1]：

    [[[3],
      [4],
      [1],
      [3]]]

这样每个 query 行有自己的最大值。接着程序从每行分数减去对应的最大值：

    shifted = logits - row_max

PyTorch 广播会沿 key 轴把每个最大值用于该行的 4 个分数。结果是：

    [[[-1, -3, -2,  0],
      [-5,  0, -2, -4],
      [ 0,  0, -1, -3],
      [-3,  0, -1, -2]]]

随后逐项求指数，并沿 key 轴求和作为分母：

    exponentials = torch.exp(shifted)
    denominator = exponentials.sum(dim=-1, keepdim=True)
    weights = exponentials / denominator

denominator 形状为 [1,4,1]，除法时广播回每行的 4 项；weights 形状恢复为 [1,4,4]。示例得到的注意力权重约为：

| query / key | key 0 | key 1 | key 2 | key 3 |
| --- | ---: | ---: | ---: | ---: |
| query 0 | 0.2369 | 0.0321 | 0.0871 | 0.6439 |
| query 1 | 0.0058 | 0.8618 | 0.1166 | 0.0158 |
| query 2 | 0.4136 | 0.4136 | 0.1522 | 0.0206 |
| query 3 | 0.0321 | 0.6439 | 0.2369 | 0.0871 |

每行非负且和为 1，可以把它看成该 query 对所有 key 的注意力分配比例。减去行最大值不会改变这些比例，却让每行送入指数函数的数都不大于 0，降低指数溢出的风险。

### ④ 乘 V

最后程序用权重矩阵乘 value：

    output = torch.matmul(weights, value)

weights 形状是 [B,S,S]，value 形状是 [B,S,D]。中间两个长度为 S 的 key 轴配对并求和，输出形状回到 [B,S,D]。固定样本 b、query 位置 i 和特征 d，输出为：

$$
\text{output}_{bid}
\overset{\text{矩阵乘法定义}}{=}
\sum_{j=1}^{4}\text{weights}_{bij}V_{bjd}
$$

对 query 位置 0 来说，就是拿 weights 的第 0 行分别乘 V 的 4 行，再把 4 个结果相加。权重决定“从每个位置取多少”，V 决定“取到什么内容”。如果一个权重接近 1，输出接近对应位置的 V；如果权重分散，输出混合多个位置的 V。

### 形状总览

| 顺序 | 程序操作 | 形状 | 含义 |
| --- | --- | --- | --- |
| 输入 | query, key, value | [B,S,D] | 每个位置各有 D 个特征 |
| ① | query @ key.transpose(-2, -1) | [B,S,D] @ [B,D,S] → [B,S,S] | 每个 query-key 位置对的点积分数 |
| ② | raw_scores / sqrt(D) | [B,S,S] | 缩放后的 logits |
| ③ | stable_softmax(scores) | [B,S,S] | 沿 key 轴逐 query 行归一化 |
| ④ | weights @ value | [B,S,S] @ [B,S,D] → [B,S,D] | 用权重混合各位置的 value |

本 reference 的单头布局是 [B,S,D]。对照 PyTorch SDPA 时，sdpa_reference 会插入长度为 1 的 head 轴，把输入暂时变为 [B,1,S,D]；SDPA 返回后再移除该轴，恢复 [B,S,D]。

## 符号说明

- `B`：批次大小，即独立样本数。
- `S`：序列长度，即每个样本的位置数。
- `D`：每个 Q/K/V 位置向量的特征数；本实验三者相同。
- `Q`（Query，查询）：当前位置用来寻找相关信息的向量。
- `K`（Key，键）：候选位置用来被匹配的向量。
- `V`（Value，值）：候选位置中实际被汇总的内容向量。
- `i,j,d`：分别为 query 位置、key 位置、特征分量的索引。
- `m[b,i]`：固定样本和 query 位置时，该行最大的 key 分数。
- `scores/weights`：分数矩阵/概率权重矩阵，形状均为 `[B,S,S]`。
- `e`：一个元素占用字节数；float64 的 e 为 8。
- `O(BS²D)`：输入尺寸放大时主运算量的渐近增长记号。
- `E[X]`：随机量 X 的期望，即大量同条件重复时的平均值。
- `Var(X)`：随机量 X 的方差，即数值围绕期望的离散程度。
- `sqrt(D)`：D 的平方根；`A @ B` 中的 `@` 是 Python/PyTorch 矩阵乘法运算符。

## 程序实例

程序路径：固定种子 → 生成三组 Q/K/V → 手写 QKᵀ、缩放、稳定 softmax、乘 V → 增加长度为 1 的 head 轴调用 SDPA → 检查误差、形状、权重行和、有限值 → 打印 FLOPs 和中间矩阵字节数。尺寸为 `(1,3,2)`、`(2,5,8)`、`(3,17,32)`。

从仓库根目录运行：`.venv/bin/python learning/batches/b03-attention/t08-scaled-dot-product-attention/attention_reference.py`。核心函数 `attention_reference` 接受同 device、同 dtype、同形状的 `[B,S,D]` 浮点 Tensor，返回输出及用于检查的分数/权重；只读输入并分配结果，不启动线程。调用期间不可并发原地修改输入。

自动断言说明这些三组 CPU float64 输入下，手写结果与 PyTorch SDPA 接近，且稳定 softmax 对超大 logits 保持有限。容差为 `rtol=1e-10, atol=1e-12`，其他设备或 dtype 应另测。该程序未覆盖 GPU、mask、因果语义、多头、dropout、反向传播和性能。

## 直观理解

把每个位置想成查资料的学生：Q 是问题，K 是目录关键词，点积给相关度，softmax 把相关度变成总和 100% 的阅读时间比例，V 是资料内容。学生按比例汇总资料得到新理解；缩放与减最大值则像校准评分并避免计算器超量程。
