# T07：训练循环

本章从二维合成分类数据开始，逐步写出一个多层感知机（MLP）的 `forward → loss → backward → step`。目标是读懂并能脱离答案手写这个闭环，理解每个 Tensor 的形状、损失梯度怎样到达参数、参数在哪一步改变，以及如何验证训练和保存/加载。任务要求损失目标必须在观察结果前定好；本实验预先设为“最终训练集交叉熵 < 0.12，且不超过初始损失的 10%”。不要求高质量大模型或泛化结论。

程序入口：[train_mlp.py](train_mlp.py)。运行证据：[T07 验证报告](../../../reports/stage-02/2026-10-10-t07-validation.md)。本章承接 [T05 链式法则](../t05-chain-rule-and-finite-differences/README.md) 和 [T06 梯度清零、反向与评估模式](../t06-gradient-accumulation-and-inference-modes/README.md)。

![image-20261010102853638](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/image-20261010102853638.png)

## 前置基础

本程序用到的基础只有以下几项。

1. **矩阵乘法与维度。** 若一批输入为 $X$，形状 $[B,I]$ 表示 B 个样本、每个 I 个特征。线性层权重 $W$ 的形状是 $[O,I]$，偏置 $b$ 是 $[O]$，则 $XW^{\mathsf{T}}+b$ 的形状为 $[B,O]$。第 n 行是第 n 个样本，第 j 列是该样本的第 j 个输出。
2. **链式法则。** 先看标量复合函数 `f(u(x))`：外层 f 依赖 u，u 又依赖 x。对 x 的变化率要先算 f 对 u 的变化率，再乘 u 对 x 的变化率：

   $$
   \frac{df}{dx}
   \overset{\text{标量链式法则}}{=}
   \frac{df}{du}\frac{du}{dx}.
   $$

   神经网络的 Tensor 链也逐坐标应用链式法则，并对中间索引求和。autograd 在 `backward()` 时沿实际执行过的 Tensor 运算完成这些乘法和求和；它负责求导，不负责更新参数。
3. **梯度下降。** 参数 $\theta$ 的局部下降方向是 $-\partial L/\partial\theta$。基础 SGD 更新为：

   $$
   \theta_{\mathrm{new}}
   \overset{\text{SGD 更新定义}}{=}
   \theta_{\mathrm{old}}-\eta\frac{\partial L}{\partial\theta}.
   $$

   学习率 $\eta$ 决定步长。梯度方向来自当前损失和参数值；过大的学习率可能越过低点，所以单步并不保证损失下降。
4. **批次平均。** `CrossEntropyLoss` 默认先对一个批次的样本损失取平均。不同批次大小时，epoch 平均值应按样本数加权，而不是把每批均值简单平均。

## 公式推导

### 公式背景

分类任务给模型一个输入样本 $x_n$，并要求它在 C 个类别中选一个标签 $y_n$。模型最后一层输出 C 个实数分数；这些分数称为 logits。logit 可以是任意正数或负数，不能直接当概率，因为它们不保证非负，也不保证总和为 1。

本章选择 softmax，把一组实数分数转换为一组合法概率。指数变换不是从“概率必须归一化”唯一推出来的，而是一个常用参数化选择：指数严格递增，所以保留分数大小顺序；指数值恒正，所以可作为概率权重；除以总和后概率和为 1。softmax 对所有 logits 同加同一个常数不变，因为分子、分母会同时乘上同一个指数因子。本节假设普通类别索引标签、不使用 class weight 和 label smoothing；这与脚本中的 `nn.CrossEntropyLoss()` 配置一致。

对一个已观测标签 $y_n$，分类分布给这个观测的似然就是 $p_{n,y_n}$。最大化这个似然会偏好给正确类别更大概率的参数。对数函数严格递增，因此最大化 $\log p_{n,y_n}$ 与最大化 $p_{n,y_n}$ 的最优参数相同；前面加负号后，最大化变成最小化：

$$
\ell_n
\overset{\text{负对数似然定义}}{=}
-\log p_{n,y_n}.
$$

因为概率在 $(0,1]$ 内，负对数损失非负；正确类概率越大，损失越小。它是观测到的真实类别的自信息，不等同于整个预测分布的 Shannon 熵。默认 batch 损失再对样本取平均，这是 `reduction="mean"` 的约定；它表示“每个样本平均承担多少损失”，不是由概率归一化本身推出来的。

### 逐式推导

#### softmax

先给类别 j 的 logit 取指数，得到正权重：

$$
a_{n,j}
\overset{\text{指数函数为正}}{=}
\exp(z_{n,j}).
$$

为这个样本的全部类别权重求和，得到归一化分母：

$$
S_n
\overset{\text{累加所有类别}}{=}
\sum_{k=0}^{C-1}a_{n,k}.
$$

用单类权重除以总权重，得到类别概率：

$$
p_{n,j}
\overset{\text{概率归一化定义}}{=}
\frac{a_{n,j}}{S_n}
\\
p_{n,j}
\overset{\text{代入 }a,S}{=}
\frac{\exp(z_{n,j})}
     {\sum_{k=0}^{C-1}\exp(z_{n,k})}.
$$

分母对 j 不变，所有类别概率相加时分子正好重现分母，因此总和为 1。每个分子都为正，所以每项概率都大于 0。对 T07 来说，类别数 $C\overset{\text{脚本类别数}}{=}3$，代码张量 `logits` 形状 `[B,3]` 的第二轴就是这里的类别索引 j；公式中的 $p_{n,j}$ 对应沿 `dim=1` 归一化后第 n 行第 j 列的值。训练代码把 logits 直接交给交叉熵函数，由其内部按稳定方式计算等价的 log-softmax；并不需要额外显式 softmax。

#### 单样本损失

模型给真实类别 $y_n$ 的概率是 $p_{n,y_n}$。负对数似然的定义就是对该概率取负对数：

$$
\ell_n
\overset{\text{负对数似然定义}}{=}
-\log p_{n,y_n}.
$$

将 softmax 定义代入。分子是正确类别 logit 的指数，分母是全部类别指数的和：

$$
\ell_n
\overset{\text{代入 softmax}}{=}
-\log\!\left(
\frac{\exp(z_{n,y_n})}
     {\sum_k\exp(z_{n,k})}
\right).
$$

用对数商法则 $\log(a/b)\overset{\text{对数商法则}}{=}\log a-\log b$ 拆开，并用 $\log(\exp(u))\overset{\text{指数与对数互逆}}{=}u$ 化简：

$$
\ell_n
\overset{\text{对数商法则}}{=}
-\log\exp(z_{n,y_n})
+\log\!\left(\sum_k\exp(z_{n,k})\right)
\\
\ell_n
\overset{\log(\exp(u))=u}{=}
-z_{n,y_n}
+\log\!\left(\sum_k\exp(z_{n,k})\right).
$$

第一项奖励正确类别 logit 变大；第二项由所有类别分数共同决定。若正确类分数相对其他类别越高，正确类概率就越大，负对数损失就越小。

单个样本的负对数似然变小，等价于模型给真实类别分配了更大的概率。对整个 batch 来说，平均损失下降只说明平均预测改善，不保证每条样本都变好，也不保证每个样本的预测熵每一步都下降。只有当某个样本的损失趋近于 0 时，正确类概率才趋近于 1，该样本的预测分布才趋近 one-hot，预测熵才趋近于 0。以下关系说明这些量如何联系。

1. 预测分布的熵在减小（更确定）

   对分类任务，softmax 输出是一个概率分布 `p_n`。它的（香农）熵为

   $$
   H(p_n)
   \overset{\text{Shannon 熵定义}}{=}
   -\sum_k p_{n,k}\log p_{n,k}.
   $$

   负对数似然 $-\log p_{n,y_n}$ 是

   真实类别 `y_n` 的自信息（surprisal）

   ，它衡量“模型给真实标签分配的概率有多小”。当 $\ell_n$ 趋近于 0 时，$p_{n,y_n}$ 趋近于 1，此时分布趋于 one-hot，熵 $H(p_n)$ 趋近于 0，模型对这个样本越来越确信。有限范围内损失变小并不保证熵每一步都单调下降，因为错误类别之间的概率也可能重新分配。

2. 交叉熵趋近真实分布的熵（这里真实分布是确定性的 one-hot，熵为 0）



   交叉熵定义为：

   $$
   H(q,\hat p)
   \overset{\text{交叉熵定义}}{=}
   -\sum_k q_k\log\hat p_k.
   $$

   其中 $q$ 是真实分布（one-hot，$q_{y_n}\overset{\text{标签定义}}{=}1$）。代入后，只有真实类别那一项非零：

   $$
   H(q,\hat p)
   \overset{\text{one-hot 标签}}{=}
   -\log\hat p_{y_n}
   \overset{\hat p=p}{=}
   \ell_n.
   $$

   KL 散度定义为真实分布到预测分布的加权对数比：

   $$
   D_{\mathrm{KL}}(q\Vert\hat p)
   \overset{\text{KL 定义}}{=}
   \sum_k q_k\log\frac{q_k}{\hat p_k}.
   $$

   把它与真实分布熵 $H(q)\overset{\text{熵定义}}{=}-\sum_kq_k\log q_k$ 相加，并使用对数商法则整理：

   $$
   H(q)+D_{\mathrm{KL}}(q\Vert\hat p)
   \overset{\text{代入熵和 KL 定义}}{=}
   -\sum_kq_k\log q_k
   +\sum_kq_k(\log q_k-\log\hat p_k)
   \\
   H(q)+D_{\mathrm{KL}}(q\Vert\hat p)
   \overset{\text{合并相反项}}{=}
   -\sum_kq_k\log\hat p_k
   \\
   H(q)+D_{\mathrm{KL}}(q\Vert\hat p)
   \overset{\text{交叉熵定义}}{=}
   H(q,\hat p).
   $$

   对 one-hot 标签，只有 $q_{y_n}\overset{\text{标签定义}}{=}1$，其他类别的 $q_k\overset{\text{one-hot 定义}}{=}0$。按极限约定 $0\log0\overset{\text{连续延拓}}{=}0$，所以 $H(q)\overset{\text{one-hot 熵}}{=}0$，且交叉熵只留下真实类别一项：

   $$
   \ell_n
   \overset{\text{one-hot 真实分布}}{=}
   H(q,\hat p)
   \overset{H(q)=0}{=}
   D_{\mathrm{KL}}(q\Vert\hat p).
   $$

   因此对固定样本，$\ell_n$ 减小等价于 $D_{\mathrm{KL}}(q\Vert\hat p)$ 减小，也就是预测分布更接近真实 one-hot 分布。

3. **信息论解释**
   $-\log p_{n,y_n}$ 使用自然对数时表示以 nat 为单位的自信息；若使用以 2 为底的对数，单位才是 bit。它可解释为按模型分布编码真实标签的理想码长，但单个标签损失下降本身不能推出模型携带了更多关于标签的信息。

总结：对单个样本，负对数损失越小，正确类概率越大，且与 one-hot 真实分布的 KL 散度越小；损失趋近 0 时，预测分布趋近 one-hot、预测熵趋近 0。对 batch 平均值，只能据此判断平均负对数似然下降，不能断言每个样本都朝同一方向变化。

#### 批次平均

一个 batch 有 B 条样本，对每条样本分别算出 $\ell_n$。默认 `reduction="mean"` 把这些样本损失相加，再除以 B：

$$
L
\overset{\text{算术平均定义}}{=}
\frac{1}{B}\sum_{n=1}^{B}\ell_n.
$$

这是平均值而不是总和，所以在样本内容相近时，改变 batch size 不会仅因样本数量变多就按比例放大 loss。T07 的 `loss_fn(logits, batch_labels)` 正是这个默认行为，返回零维标量 Tensor；`loss.backward()` 从此标量反传。若改为 `reduction="sum"`，loss 和梯度会变成当前平均值的 B 倍，通常还需相应调整学习率或损失缩放。

#### 梯度

训练还需要知道损失如何推动 logits。先对单样本损失 $\ell_n\overset{\text{负对数似然}}{=}-\log p_{n,y_n}$ 关于类别 j 的 logit 求导。softmax 对 logit 的导数来自商法则；结果可写为“本类概率乘上本类指示值减去本类概率”：

先写正确类概率的分子、分母。对分数 `z[n,j]` 求导时，分子只有当 j 正好是正确类时才变化；分母中的第 j 项总会变化：

$$
\frac{\partial p_{n,y_n}}{\partial z_{n,j}}
\overset{\text{商法则}}{=}
\frac{
\mathbf{1}[j=y_n]\exp(z_{n,y_n})S_n
-\exp(z_{n,y_n})\exp(z_{n,j})
}{S_n^2}.
$$

将 $\exp(z_{n,y_n})/S_n$ 识别为正确类概率，将 $\exp(z_{n,j})/S_n$ 识别为第 j 类概率：

$$
\frac{\partial p_{n,y_n}}{\partial z_{n,j}}
\overset{\text{提取两个概率因子}}{=}
p_{n,y_n}
\left(\mathbf{1}[j=y_n]-p_{n,j}\right).
$$

链式法则把它代入负对数的导数：

$$
\frac{\partial\ell_n}{\partial z_{n,j}}
\overset{\text{链式法则}}{=}
-\frac{1}{p_{n,y_n}}
 \frac{\partial p_{n,y_n}}{\partial z_{n,j}}
\\
\frac{\partial\ell_n}{\partial z_{n,j}}
\overset{\text{代入 softmax 导数}}{=}
-\left(\mathbf{1}[j=y_n]-p_{n,j}\right)
\\
\frac{\partial\ell_n}{\partial z_{n,j}}
\overset{\text{展开负号}}{=}
p_{n,j}-\mathbf{1}[j=y_n].
$$

批次损失是 B 个单样本损失的平均，求导时常数 $1/B$ 保留在每项前面：

$$
\frac{\partial L}{\partial z_{n,j}}
\overset{\text{对批次均值求导}}{=}
\frac{p_{n,j}-\mathbf{1}[j=y_n]}{B}.
$$

这就是 `CrossEntropyLoss` 产生并继续沿 MLP 计算图传播的 logits 梯度。PyTorch 再用链式法则把它传过输出 Linear、ReLU、隐藏 Linear，最终累加到各参数的 `.grad`。

### 符号说明

| 符号 | 含义 | T07 中对应 |
| --- | --- | --- |
| $x_n$ | 第 n 个样本的输入特征向量 | `features[n]`，形状 `[2]` |
| $X$ | 当前批次的输入矩阵 | `batch_features`，形状 `[B,2]` |
| $W_1,b_1$ | 第一层 Linear 的权重和偏置 | `model.hidden.weight` `[16,2]` 与 `.bias` `[16]` |
| $H$ | 第一层仿射变换的隐藏值 | `hidden_values`，形状 `[B,16]` |
| $A$ | ReLU 输出的隐藏特征 | `activated`，形状 `[B,16]` |
| $W_2,b_2$ | 输出层 Linear 的权重和偏置 | `model.output.weight` `[3,16]` 与 `.bias` `[3]` |
| $Z$ | 输出类别 logits 的矩阵 | 代码变量 `logits`，形状 `[B,3]` |
| $n$ | batch 内样本索引，范围 $1$ 到 $B$ | `logits` / `labels` 的第 0 轴位置；代码下标从 0 开始 |
| $j,k$ | 类别索引，范围 $0$ 到 $C-1$ | `logits` 的第 1 轴；本例取 0、1、2 |
| $C$ | 类别总数 | `CLASS_COUNT=3` |
| $B$ | 当前 batch 的样本数 | `batch_labels.shape[0]`，通常为 48 |
| $y_n$ | 第 n 个样本的真实类别整数 | `batch_labels[n]` |
| $z_{n,j}$ | 第 n 个样本第 j 类的原始分数 | `logits[n, j]` |
| $a_{n,j}$ | 指数映射后的正权重 | softmax 分子中的 `exp(logits[n, j])` |
| $S_n$ | 样本 n 的全部类别指数权重总和 | softmax 归一化分母 |
| $p_{n,j}$ | softmax 后第 n 个样本第 j 类概率 | 对 `logits` 沿 `dim=1` 归一化的结果 |
| $\ell_n$ | 第 n 个样本的负对数似然 | 单条样本交叉熵 |
| $L$ | 当前 batch 平均损失 | `loss`，零维 Tensor |
| $\mathbf{1}[j=y_n]$ | 条件成立时为 1，否则为 0 | 公式中的正确类别选择器；非标签 Tensor |
| $p_n$ | 样本 n 的整个预测概率向量 | softmax 沿类别轴计算的第 n 行 |
| $q$ | 单样本真实标签的 one-hot 概率分布 | 推导交叉熵/KL 关系用；代码用整数标签而不构造它 |
| $\hat p$ | 用模型预测的类别概率分布 | 理论记号；代码传 logits 给交叉熵，未单独存放 |
| $H(p_n)$ | 预测概率向量的 Shannon 熵 | 本脚本未计算，仅用于区分不确定度与真实标签损失 |
| $D_{\mathrm{KL}}(q\Vert\hat p)$ | 真实分布到预测分布的 KL 散度 | 本脚本未单独计算；one-hot 标签下等于单样本 NLL |
| $\theta,\eta$ | 待更新参数与正学习率 | 任一模型 Parameter 与 `LEARNING_RATE` |

## 程序实例

这是T07这套代码在实际运行时，每一个核心步骤输入和产出的具体数据形状与含义：

**1. 数据生成阶段 (`make_dataset`)**



- **输入：** 随机种子 `SEED` (20261010)。
- **产出：**
  - `features`：形状为 `[384, 2]` 的张量。代表 384 个样本，每个样本有 x 和 y 两个坐标值（例如 `[1.34, -0.72]`）。
  - `labels`：形状为 `[384]` 的整数张量。代表这 384 个样本分别属于哪一个类别（取值为 0、1 或 2）。

**2. 批次加载阶段 (`DataLoader`)**



- **输入：** 全部的 384 个样本数据。
- **产出：** 每次循环吐出一个批次（Batch）。
  - `batch_features`：形状为 `[48, 2]`。从总数据里抽出的 48 个坐标点。
  - `batch_labels`：形状为 `[48]`。这 48 个点对应的正确类别。

**3. 前向传播阶段 (`model(batch_features)`)**

这一步是模型真正在“做预测”，它分为三层依次进行：



- **第一层（隐藏层 `self.hidden`）：**
  - 输入：`[48, 2]` 的坐标数据。
  - 产出：`[48, 16]` 的张量。模型用它的权重，把二维坐标升维，提取成了 16 个隐藏特征。
- **第二层（激活函数 `self.activation`）：**
  - 输入：`[48, 16]` 的隐藏特征。
  - 产出：依然是 `[48, 16]`。把所有负数特征变成 0（ReLU），保留正数，引入非线性。
- **第三层（输出层 `self.output`）：**
  - 输入：经过激活的 `[48, 16]` 特征。
  - 产出：最终的 `logits`，形状为 `[48, 3]`。代表这 48 个样本，每个样本对应 3 个类别的“原始得分”（数字越大，模型越认为属于该类）。

**4. 误差计算阶段 (`loss_fn(logits, batch_labels)`)**



- **输入：** 模型给出的得分 `[48, 3]` 和 正确答案 `[48]`。
- **产出：** 一个0维的标量（例如 `1.15`）。程序内部会先把得分转化为概率，然后用交叉熵公式算出这 48 个样本的平均预测误差。这个数字越小，说明预测越准。

**5. 反向传播阶段 (`loss.backward()`)**



- **输入：** 刚才算出的误差标量（如 `1.15`）。
- **产出：** 无直接返回值，但它会**暗中修改**模型内部状态。它顺着计算图往回推，算出每一层权重对最终误差的“责任大小”（也就是梯度），并把这些梯度保存在参数的 `.grad` 属性里（例如，第一层权重的梯度形状与权重本身一样，都是 `[16, 2]`）。

**6. 参数更新阶段 (`optimizer.step()`)**



- **输入：** 模型中储存的梯度 `.grad`，以及我们在开头设定的学习率 `LEARNING_RATE = 0.08`。
- **产出：** 无返回值。优化器会根据输入的数据，把模型里所有的权重和偏置数值稍微调整一下。调整后，下一轮输入同样的数据时，模型算出的误差就会变小一点。

### 任务形状

每个合成样本是二维平面中的点，标签为三个类别之一。数据围绕三个固定中心生成，少量高斯噪声让点不完全重合。样本标签不是模型输入的一部分。

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `features` | `[N, 2]` | N 个二维样本 |
| `labels` | `[N]` | 每个样本的类别整数 `0/1/2`，类型为 int64 |
| `hidden.weight` | `[16, 2]` | 输入特征到隐藏特征的权重 |
| `hidden.bias` | `[16]` | 隐藏层偏置 |
| `output.weight` | `[3, 16]` | 隐藏特征到类别分数的权重 |
| `output.bias` | `[3]` | 每个类别的偏置 |
| `logits` | `[B, 3]` | 一个批次 B 个样本各自的三个未归一化类别分数 |

这里的 MLP 有两次仿射映射，中间有 ReLU：

$$
\begin{aligned}
H &\overset{\text{第一层 Linear}}{=} XW_1^{\mathsf{T}}+b_1,
&H&\in\mathbb{R}^{B\times16},\\
A &\overset{\text{逐元素 ReLU}}{=} \max(H,0),
&A&\in\mathbb{R}^{B\times16},\\
Z &\overset{\text{第二层 Linear}}{=} AW_2^{\mathsf{T}}+b_2,
&Z&\in\mathbb{R}^{B\times3}.
\end{aligned}
$$

$\max(H,0)$ 是逐元素操作：每个负元素变为零，正元素保持原值。隐藏层因此能构造分段线性的非线性边界；如果删掉 ReLU，连续线性映射的复合仍是一个线性映射，隐藏层不会带来非线性表达能力。$Z$（代码变量 `logits`）没有 softmax，因为交叉熵损失直接接收 logits，并在内部以数值稳定的方式计算对数概率。

### 随机种子

`SEED=20261010` 是本章记录的种子。`torch.manual_seed(SEED)` 控制 PyTorch 使用的全局随机数流；`torch.Generator().manual_seed(seed)` 则创建单独的随机数生成器。脚本用专用 generator 生成数据，也为 DataLoader 的 shuffle 单独设 generator。这样数据生成和批次顺序可重复，并减少调用其他随机代码对这两处的影响。固定种子不是跨 PyTorch 版本、设备、算子都逐位相同的普遍保证；本次只记录实际环境与结果。

### 批次数据

`TensorDataset(features, labels)` 把两个沿第 0 维一一对应的 Tensor 组成数据集；访问索引 `i` 会取出 `(features[i], labels[i])`。`DataLoader(dataset, batch_size=48, shuffle=True, ...)` 是可迭代对象：每轮迭代产出一对小批次 Tensor，而不是一次把 384 个样本作为一个更新步骤。数据集大小能被 48 整除，本例每轮恰有 8 批。

Python 的 `for batch_features, batch_labels in train_loader:` 会反复取下一批，并把返回 tuple 的两项分别绑定到两个变量。小批量更新让每一步只处理 48 个样本；打乱顺序让每个 epoch 的批次次序变化。该脚本的 CPU 张量已在内存中，不涉及文件解码、多进程加载或 GPU 传输。

### 模型与前向

`class SmallMLP(nn.Module):` 用 Python 类定义一个模型类型；括号中的 `nn.Module` 表示继承 PyTorch 模块基类。`__init__` 是创建对象时调用的初始化方法。`super().__init__()` 初始化基类内部登记机制，随后赋给 `self.hidden`、`self.output` 的 Linear 子模块会被登记。Linear 内的 `weight`、`bias` 是可训练 Parameter，因此 `model.parameters()` 能找到它们，优化器能更新它们，`state_dict()` 能保存它们。

`self` 是当前模型对象，`self.hidden(features)` 的括号调用该子模块的 `forward`。当写 `model(features)` 时，PyTorch Module 的调用路径会执行 `forward(features)`，并在模块调用过程中管理钩子等框架行为。因此通常调用 `model(x)`，而不是直接调用 `model.forward(x)`。

本模型的 forward 实际顺序如下：

1. `hidden_values = self.hidden(features)`：对每一行计算 `x @ W1.T + b1`，由 `[B,2]` 得到 `[B,16]`。
2. `activated = self.activation(hidden_values)`：ReLU 负数归零，形状仍是 `[B,16]`。
3. `return self.output(activated)`：第二个 Linear 产生 `[B,3]` logits，返回给调用者。

`def forward(self, features: torch.Tensor) -> torch.Tensor:` 中冒号开启函数体，缩进定义函数范围；类型标注提示输入和输出类型，但 Python 通常不靠它自动强制类型。`return` 把结果交给 `model(features)` 的调用位置。`self` 是方法调用时自动传入的当前实例。

### 数值实例

![image-20261010103110770](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/image-20261010103110770.png)

下面用项目实际产生的首个训练 batch 的第一条样本。条件固定为脚本记录的 seed `20261010`、shuffle seed `20261011`、CPU、初始随机模型（尚未执行任何 step）。输入点约为 $[-1.510805,-0.633204]$，标签是类别 0；模型输出的三个 logits 为：

$$
z_{1,:}
\overset{\text{模型 forward，四舍五入}}{=}
[-0.128450,\;-0.042385,\;0.193429].
$$

先逐类取指数。三个值和分母按顺序是：

$$
\exp(z_{1,:})
\overset{\text{逐类取指数}}{\approx}
[0.879458,\;0.958500,\;1.213403]
\\
S_1
\overset{\text{三类权重相加}}{\approx}
0.879458+0.958500+1.213403
\\
S_1
\overset{\text{求和，四舍五入}}{\approx}
3.051361.
$$

例如正确类 0 的概率，是类别 0 权重除以总权重；其余两项按相同方法得到：

$$
p_{1,:}
\overset{\text{逐项相除，四舍五入}}{\approx}
\left[
\frac{0.879458}{3.051361},
\frac{0.958500}{3.051361},
\frac{1.213403}{3.051361}
\right]
\\
p_{1,:}
\overset{\text{计算各分数}}{\approx}
[0.288218,\;0.314122,\;0.397660].
$$

三项相加约为 1。标签为 0，因此单样本损失只取第一项概率的负对数：

$$
\ell_1
\overset{\text{负对数似然}}{\approx}
-\log(0.288218)
\\
\ell_1
\overset{\text{计算并四舍五入}}{\approx}
1.244038.
$$

交叉熵展开式也能独立核对同一个数：

$$
\ell_1
\overset{\text{代入正确类 logit 与分母}}{\approx}
-(-0.128450)+\log(3.051361)
\\
\ell_1
\overset{\log(3.051361)\approx1.115588}{\approx}
0.128450+1.115588
\\
\ell_1
\overset{\text{相加，四舍五入}}{\approx}
1.244038.
$$

训练脚本的 float32 输出为 `1.2440375` 左右；与手算末位略有不同来自十进制展示舍入和 PyTorch 稳定实现的浮点舍入，不是公式差异。

首批 B=48 条样本各自都有一个 $\ell_n$。脚本此时尚未训练，首批平均交叉熵约为 $1.097396$：

$$
L_{\mathrm{batch}}
\overset{\text{批损失总和除以 48}}{\approx}
\frac{52.675032}{48}
\\
L_{\mathrm{batch}}
\overset{\text{脚本实算，四舍五入}}{\approx}
1.097396.
$$

其中 `52.675032` 是 48 条样本各自交叉熵的和（由脚本对每条样本的 loss 求和得到）。因此这一条样本的 `1.244038` 不等于整批 `loss`；它是 48 个加数中的一个。代码里 `CrossEntropyLoss` 负责对整批 logits/labels 算这个均值，训练循环再对该标量调用 `backward()`。

同一条样本的概率还能直接核对 backward 的起点。单样本 logits 梯度先是概率减正确类 one-hot 指示向量；放进 B=48 的批均值后再除以 48：

$$
\nabla_{z_1}L
\overset{\text{概率减正确类指示，再除以 48}}{=}
\frac{[0.288218,\;0.314122,\;0.397660]-[1,\;0,\;0]}{48}
\\
\nabla_{z_1}L
\overset{\text{逐项相减并除以 48}}{\approx}
[-0.014829,\;0.006544,\;0.008285].
$$

第一类的梯度为负，梯度下降会提高正确类 logit；另两类梯度为正，梯度下降会降低错误类 logit。PyTorch 从这个 logits 梯度继续沿 MLP 的计算图反传到 weight 和 bias。

每一步下降都在减小模型分布与真实分布之间的 KL 散度；$(-1/48)(p-q)$ 就是 KL 散度下降最快的反方向在该点的值。

### 手写四步

下面是每个 batch 都要亲手重写的核心。阅读时把每行与完整脚本中的训练循环对照；做独立练习时先遮住右侧说明，逐行写完再解释每个变量形状。

```python
for batch_features, batch_labels in train_loader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch_features)                  # forward
    loss = loss_fn(logits, batch_labels)            # loss
    loss.backward()                                 # backward
    optimizer.step()                                # step
```

### 清理梯度

`optimizer.zero_grad(set_to_none=True)` 找到该优化器管理的参数，并把各参数的 `.grad` 设成 `None`。`.grad` 是保存上次反向结果的缓冲区。PyTorch 的 `backward()` 默认将新梯度累加进已有缓冲区；如果独立批次前不清理，第二批更新使用的就是前批梯度与本批梯度之和。

`set_to_none=True` 用 `None` 表示当前没有梯度，下一次 backward 建立本轮梯度。使用普通 `zero_grad()` 也会清理梯度，但具体默认值可能受 PyTorch 版本 API 影响；本脚本显式给出参数，也让“无梯度”状态可读。若刻意做梯度累积，才会跨多个 batch 延后清理和 step。

### 前向计算

`logits = model(batch_features)` 读取当前参数和本批输入，得到 `[B,3]`。这时参数没有被改写。因为模型参数需要梯度，PyTorch 会记录这些 Linear、ReLU 等操作间的依赖，形成之后 backward 要遍历的计算图。

forward 是一次函数求值：输入是当前样本和当前权重，输出是当前类别分数。把它放在 step 之后会用更新后的参数计算当前批预测；把 loss 放在不同 forward 上则会让梯度对应到不同的模型状态。

### 损失与反传

`loss_fn(logits, batch_labels)` 将 logits 与真值类别比较，产生一个 0 维标量 Tensor。标量是一个数而不是长度为 1 的向量。调用 `loss.backward()` 相当于从这个标量开始反向求导；由于上游标量系数默认为 1，autograd 沿 forward 计算图应用链式法则，把每个叶子参数的 $\partial L/\partial\theta$ 累加到 `.grad`。

令最后一层 logits 为 $z_{n,j}$，softmax 概率为 $p_{n,j}$，批次平均交叉熵对 logits 的导数是：

$$
\frac{\partial L}{\partial z_{n,j}}
\overset{\text{softmax 交叉熵求导}}{=}
\frac{p_{n,j}-\mathbf{1}[j=y_n]}{B}.
$$

$\mathbf{1}[j=y_n]$ 在 j 是正确类别时取 1，否则取 0。这个式子告诉我们：正确类分数的梯度为 $(p_{n,j}-1)/B$，错误类分数梯度为 $p_{n,j}/B$。梯度再经过输出 Linear、ReLU 和隐藏 Linear 传播到所有 weight 与 bias。ReLU 正值区间局部导数为 1、负值区间为 0；恰为 0 的点采用框架选定的次梯度约定，连续随机输入落在精确零上的概率通常很低，但这不是本实验单独验证的结论。

以单个样本、正确类别 0 为例，若 softmax 概率是 $[0.7,0.2,0.1]$，则 logits 梯度是：

$$
[0.7-1,\;0.2-0,\;0.1-0]
\overset{\text{逐项相减}}{=}
[-0.3,\;0.2,\;0.1].
$$

正确类 logit 往上推会降低 loss，所以它的梯度为负；两个错误类 logit 往上推会增加 loss，所以梯度为正。实际批次再除以 B 并对样本求和。

**backward 只算梯度，不改权重。** 此时你可以查看 `model.hidden.weight.grad`；若要查看具体值，需要在 backward 后读取。模型参数的值要等 optimizer.step 才更新。

### 参数更新

`optimizer.step()` 让 SGD 读取每个参数当前的 `.grad`，并按下式更新权重和偏置：

$$
\theta
\overset{\text{SGD 参数更新}}{\leftarrow}
\theta-\eta\nabla_{\theta}L.
$$

Python 代码无需逐参数手写，因为创建 `torch.optim.SGD(model.parameters(), lr=...)` 时，优化器已取得模型参数的引用并登记更新规则。

执行后，模型对象中的 Parameter 数值变化；前一批的 logits、loss 是更新之前算出来的旧结果。下一批会用更新后的参数重新 forward，并构建新计算图。训练循环的时间顺序是：清梯度 → 用当前权重前向 → 计算损失 → 反向得梯度 → 优化器改权重 → 下一批。

### 损失非单调

SGD 用当前梯度的一阶局部方向更新。非凸神经网络损失面可能弯曲，固定学习率的一步可能越过低点；Mini-batch 的梯度也只是当前批样本的平均，不一定等于全体样本的方向。因此本任务采取可复现的固定合成数据和容易的分类边界，训练多轮，然后检查预定阈值。这个证据说明这次设置达到目标，不证明任意模型、随机种子或学习率都会下降。

训练集评估也不是独立泛化评测。脚本用同一合成样本算初始和最终损失，目的只是证明训练闭环能优化一个已知小任务。若要评估泛化，应另外留出未训练的数据；本 T07 不把这作为验收要求。

### 评估保存

`model.eval()` 切换会依据训练/评估状态改变行为的子模块。本模型只有 Linear 与 ReLU，它们当前没有状态差异；仍明确调用 eval，让保存后比较 logits 的流程具备清楚的评估语义。`torch.no_grad()` 禁止该代码块创建反向图，因为评估只需数值，不需梯度。这是两个不同开关，T06 有专门对照。

`model.state_dict()` 返回从参数名称到 Tensor 状态的映射。本章 `torch.save(state_dict, path)` 保存这组参数；加载时先创建结构相同的 `SmallMLP()`，然后用 `load_state_dict(state)` 将 Tensor 复制进对应参数。权重状态文件不自动包含 Python 类的实现，也不包含本实验要恢复训练时可能需要的优化器状态、epoch 或随机数状态。

脚本在保存前用 eval/no-grad 算完整数据的 logits；加载到新模型后用相同输入再算一次，并用 `torch.testing.assert_close(..., rtol=0, atol=0)` 检查逐元素完全相等。两次都是同设备、相同结构、相同参数和确定性 Linear/ReLU 运算，所以本例预期最大差为 0。此检查证明 checkpoint 恢复了这次推理所需的模型参数；不证明文件跨软件版本兼容，也不验证断点续训恢复优化器状态。

`torch.load(..., map_location="cpu", weights_only=True)` 把权重映射到 CPU，并启用仅权重安全加载路径。checkpoint 在临时目录中创建，训练完成后随 `TemporaryDirectory` 上下文退出而删除；这样复跑不在仓库留下随机生成的二进制文件。若希望永久保存，需自行把路径改成明确的产物位置，并一并考虑模型配置和优化器状态。

### 手写练习

请按顺序在新文件或纸上完成，每一步先写再对照 `train_mlp.py`。

1. **形状标注。** 写出输入 `[B,2]` 经 `Linear(2,16)`、ReLU、`Linear(16,3)` 后每步形状，并说明 bias 如何沿批次维广播。
2. **独写 forward。** 不复制本页代码，写 `SmallMLP.forward` 三步；逐行说明 `self`、点号属性、括号调用、局部变量和 `return`。
3. **写 loss。** 给出 logits `[B,3]` 与 long labels `[B]`，写 `nn.CrossEntropyLoss()` 调用；解释为何不手工 softmax、为什么 loss 是标量。
4. **写完整 batch 更新。** 从 `zero_grad(set_to_none=True)` 开始写四步；遮住答案后，逐句说出执行前后的 `.grad`、计算图和参数状态。
5. **解释反向方向。** 对概率 `[0.7,0.2,0.1]`、标签 0，写出单样本 logits 梯度；说明梯度的符号如何影响正确/错误类分数。
6. **加上训练外壳。** 写 epoch 循环和 DataLoader 批次循环；说出 shuffle、生成为何可复现、epoch 平均损失为什么乘回批大小再除样本总数。
7. **写验收与 checkpoint。** 在看最终数值前先声明损失阈值；保存 `state_dict`，新建同结构模型、载入、eval、no-grad 前向并比较 logits。



### 执行路径

| 顺序 | 函数/阶段 | 输入与变化 | 输出/证据 |
| --- | --- | --- | --- |
| 1 | `main()` 固定种子 | 初始化全局随机数状态 | 控制可复现实验条件 |
| 2 | `make_dataset()` | seed；生成 N 个二维点与整数标签，再共同打乱 | `features [384,2]`、`labels [384]` |
| 3 | `TensorDataset`、`DataLoader` | 按样本索引配对；按 48 个一批迭代 | 每 epoch 8 个批次 |
| 4 | `SmallMLP.__init__` | 创建并登记两层 Linear、ReLU | 16 维隐藏表示到 3 类 logits |
| 5 | `evaluate_loss()` | eval/no-grad 遍历全部样本 | 加权平均交叉熵，作为初始/最终值 |
| 6 | 训练循环 | 每批清梯度、forward、loss、backward、step | 参数反复更新；周期性打印批次加权损失 |
| 7 | 预定目标断言 | 比较最终损失与绝对阈值及初始值比例 | 未达标时抛 AssertionError 并以失败退出 |
| 8 | 保存/重载核对 | state_dict 写入临时 checkpoint，再加载到新模型 | 相同 eval 输入上的 logits 最大误差为 0 |

从入口输入到输出的形状流是：`features [384,2] → batch [48,2] → hidden [48,16] → logits [48,3] → scalar loss → 参数 .grad → step 更新参数`。最后一批是否较小由样本数决定；本例 384 能被 48 整除，因此批次大小均为 48。

### 设计证据

| 设计选择 | 提供的保证与原因 | 省略/替换后果 | 当前证据与边界 |
| --- | --- | --- | --- |
| 固定样本、模型种子和 shuffle generator | 让数据、初始化、批次顺序在本环境可重放，便于复核损失目标 | 随机流变化可能改变初始损失与收敛轨迹 | 输出记录 seed、PyTorch 版本；跨版本/设备不承诺逐位一致 |
| 小型三类二维合成数据 | 可直接看到输入、类别和 logits 的形状，快速运行闭环 | 大数据会引入无关的下载、预处理和设备问题 | 全量训练集 loss 通过阈值；不证明泛化效果 |
| ReLU 隐藏层和原始 logits 输出 | 提供非线性决策边界；CE 接收稳定计算所需的原始类别分数 | 去掉 ReLU 会退化为整体线性映射；预先 softmax 不符合 CE 的预期输入 | 训练达到损失门限；没有与其他网络结构比较 |
| 手写 `zero_grad → forward → loss → backward → step` | 让梯度缓冲区、计算图和参数更新的时间关系显式可读 | 次序错或漏清梯度会用旧梯度、无对应图梯度，或在错误参数状态上优化 | 实际训练脚本逐批执行；不单独展示故障分支 |
| 预先固定双重 loss 门限 | 同时要求绝对损失够低，并相对起点明显下降 | 只打印损失不能自动让验收失败 | 两个 Python assert 运行时强制检查；目标仅适用于该合成数据设定 |
| 保存 `state_dict` 后新建模型重载 | 验证权重保存恢复能重现模型推理输出 | 只保存文件而不重载比较，无法证明加载后行为相同 | eval logits `atol=rtol=0`；不测试断点续训或跨版本格式 |

证据层次：Python 解释器在运行时解析语法；Tensor 运算在运行时检查形状与 dtype；训练损失断言核对本次优化目标；保存/加载断言核对本次 checkpoint 推理一致。这里没有编译期检查，也没有独立测试集、GPU、多进程 DataLoader、梯度有限差分复核或性能测量。

### 运行方式

从仓库根目录执行：

```text
.venv/bin/python learning/stage-02/t07-mlp-training-loop/train_mlp.py
```

成功时末尾出现 `save_load_max_logit_error=0.0` 和 `T07 checks passed`。若损失断言失败，错误信息指出绝对阈值或相对下降比例未达到；若 checkpoint 输出不一致，`assert_close` 会抛出异常。脚本不下载数据，CPU 和已安装的 PyTorch 即可运行。

## 直观理解

可以把 logits 想成三匹赛马的原始成绩：softmax 把成绩换成总和为 1 的获胜概率；交叉熵只看真实获胜者分到多少概率，分得越少惩罚越大；batch 平均则把一场比赛中每条样本的惩罚取平均。

## 参考资料

- R01：[Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)：Dataset / DataLoader 分工、批次迭代。
- R01：[Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)：Module、子模块、forward 与 Parameter。
- R01：[Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：清梯度、反向传播和 optimizer step 的训练循环。
- PyTorch API：[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：logits 与类别索引目标的形状、类型契约以及 `reduction` 行为。
- R01：[Save and Load the Model](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)：state_dict、torch.save/load 与推理。
- 延续材料：[T05 链式法则与有限差分](../t05-chain-rule-and-finite-differences/README.md)、[T06 梯度累积与推理模式](../t06-gradient-accumulation-and-inference-modes/README.md)。
