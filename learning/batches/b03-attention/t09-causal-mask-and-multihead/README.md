# T09 因果多头

本章从 `[B,S,D]` 的 token 表示出发，依次做 Q/K/V 投影、拆分注意力头、屏蔽未来位置、合并各头并投影回 `[B,S,D]`。程序入口是 [multihead_attention.py](multihead_attention.py)，自动验收在 [test_multihead_attention.py](test_multihead_attention.py)。这是一份可读 reference，使用 PyTorch 的 scaled dot-product attention（SDPA，缩放点积注意力）函数；不包含 token embedding、位置编码、残差、层归一化和前馈网络。

资料：[Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) 及 [PyTorch SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。原论文多头图展示“线性投影 → 并行注意力头 → 拼接 → 输出投影”；本实现把多个头合批放在一个张量中调用 SDPA，计算含义相同。

## 前置基础

### 张量与布局

Tensor（张量）是装多维数值的对象。形状 `[B,S,D]` 表示 B 个样本，每个样本有 S 个 token（词元/序列位置），每个 token 用 D 个特征数表示。轴顺序是数据解释的一部分：`x[b,s,d]` 指第 b 个样本、第 s 个位置的第 d 个特征。

本章内部把 D 拆成 H 个头、每头 d 个特征，要求 `D = H*d`。头轴让不同投影子空间能够分别计算注意力。它不是把 token 序列切成 H 段，而是把每个 token 的特征投影拆成 H 组。

### QKV 与注意力

Query（查询，Q）提供当前位置寻找信息的线索；Key（键，K）提供各位置供匹配的线索；Value（值，V）携带被汇总的内容。Scaled Dot-Product Attention（缩放点积注意力）先算 Q 与各 K 的点积，除以单头维度的平方根，然后用 softmax（归一化指数函数）得到权重，并按权重求和 V。Self-attention（自注意力）表示 Q/K/V 都由同一输入序列投影得到。

Multi-Head Attention（多头注意力）对 Q/K/V 使用不同的可学习线性投影，在多个头中并行做注意力，再拼接各头输出并做一次输出投影。原论文指出，不同头可以关注不同表示子空间和位置关系；具体学到什么关系由训练决定，不能仅凭头数断言。

### 因果与 mask

Causal mask（因果掩码）规定位置 i 只能读取位置 j 不大于 i 的信息。矩阵可写成：

```text
query 行 i / key 列 j：允许当 j <= i
S=4 时：
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

在 PyTorch `scaled_dot_product_attention` 的布尔 `attn_mask` 中，`True` 表示允许该 query-key 配对参与注意力，`False` 表示屏蔽。注意这与 `nn.MultiheadAttention` 的布尔 `key_padding_mask` 语义相反：后者 `True` 表示要屏蔽。不要把两种 API 的布尔 mask 直接互换。

SDPA 的 `is_causal=True` 会应用因果约束，本实现因此传 `attn_mask=None`；当前 API 不允许同时设置 `attn_mask` 与 `is_causal=True`。因果条件的关键不是删掉未来 token 的计算，而是让其对过去位置的注意力权重为零。

### Dropout模式

Dropout（随机失活）在训练时按概率随机将部分注意力权重置零并重新缩放。`torch.nn.functional.scaled_dot_product_attention` 是函数，不会读取外层模块的 `training` 状态。即使调用 `module.eval()`，传入非零 `dropout_p` 仍会执行 dropout。因此 forward 必须按 `self.training` 显式传概率；本实现的 eval 路径传 `0.0`。

#### 训练标志

在 PyTorch 里，**每个 `nn.Module`（模型/层）都有一个布尔属性叫 `training`**，表示“当前是不是在训练”。



| 状态          | `training` 值 | 含义                                           |
| ------------- | ------------- | ---------------------------------------------- |
| 训练模式      | `True`        | 模型正在训练，dropout、BN 等要生效             |
| 推理/评估模式 | `False`       | 模型在预测/测试，dropout 要关闭，BN 用移动平均 |

**默认情况下**，你 `module.train()` 会把 `training` 设为 `True`，`module.eval()` 会设为 `False`。

> 可以把它理解成一个**全局开关**，告诉模型里的所有层：“现在是训练还是测试？”

#### eval行为

```python
module.eval()
```

它的作用是：**递归地把 module 及其所有子模块的 `training` 标志设为 `False`**。

**但它本身不会直接改变任何计算逻辑**，只是改了一个标志位。真正起作用的，是各个层在 forward 里**自己去读这个标志**，然后决定行为。

#### 随机失活

**Dropout（随机失活）** 是训练时用来防止过拟合的技术：

- **训练时**：按概率 `p` 随机把一些神经元/权重**置零**，剩下的**除以 `(1-p)` 放大**（这就是“重新缩放”），保持期望不变。
- **推理时**：**不随机置零**，所有神经元都参与计算，直接用完整输出。

> 你可以把 dropout 想象成“训练时随机让一部分人休息，推理时全员上岗”。

#### SDPA函数

`torch.nn.functional.scaled_dot_product_attention`（简称 SDPA）是 **函数**，不是 `nn.Module` 子类。

**关键区别**：

| 类型                                     | 会不会自动读 `training` 状态                               |
| ---------------------------------------- | ---------------------------------------------------------- |
| `nn.Dropout`（模块）                     | ✅ 会自动根据 `module.training` 决定做不做 dropout          |
| `F.scaled_dot_product_attention`（函数） | ❌ **完全不看 `training`**，你传什么 `dropout_p` 它就做什么 |

也就是说：

```python
F.scaled_dot_product_attention(Q, K, V, dropout_p=0.1)
```

- 不管你是 `train()` 还是 `eval()`，**只要 `dropout_p > 0`，它都会执行 dropout**。
- 这在推理时是**灾难性的**：预测结果会随机抖动，且数值被错误缩放。

#### 模式标志

在 `nn.Module` 的子类里，`self.training` 就是**当前模块自己的 `training` 标志**。

```python
class MyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1

    def forward(self, Q, K, V):
        # self.training 是 True 还是 False，取决于外部有没有调用 model.train() / model.eval()
        dp = self.dropout_p if self.training else 0.0
        return F.scaled_dot_product_attention(Q, K, V, dropout_p=dp)
```

**逻辑图解**：

```
model.train()  →  self.training = True  →  dropout_p = 0.1  →  训练时做 dropout
model.eval()   →  self.training = False →  dropout_p = 0.0  →  推理时关闭 dropout
```



> Dropout 在训练时按概率随机将部分注意力权重置零并重新缩放。

✅ 训练时：随机置零 + 缩放，防止过拟合。

> `torch.nn.functional.scaled_dot_product_attention` 是函数，不会读取外层模块的 `training` 状态。

✅ 它是函数，不看 `self.training`，你传什么 `dropout_p` 它就做什么。

> 即使调用 `module.eval()`，传入非零 `dropout_p` 仍会执行 dropout。

✅ `eval()` 只改标志，不改函数行为，所以 dropout 依然会执行，这是错的。

> 因此 forward 必须按 `self.training` 显式传概率；本实现的 eval 路径传 `0.0`。

✅ 正确做法：在 `forward` 里手动判断 `self.training`，训练时传 `dropout_p`，推理时传 `0.0`。



**`module.eval()` 只是改了个标志，`F.scaled_dot_product_attention` 是个“死脑筋”函数，不看标志。所以你必须自己用 `self.training` 判断，在推理时把 `dropout_p` 手动设成 0。**

## 公式推导

### 目标与形状

令输入 `X` 的形状为 `[B,S,D]`，头数为 H，单头宽度 `d=D/H`。投影层把每个 token 的 D 个输入特征变成 3D 个 Q/K/V 特征；之后 reshape 和 transpose 只改变轴的组织方式，不改变数值总数。

| 阶段 | 形状 | 说明 |
| --- | --- | --- |
| 输入 X | `[B,S,D]` | 每个位置一个 D 维向量 |
| 联合 QKV 投影 | `[B,S,3D]` | 末轴依次放 Q、K、V |
| 拆分并换轴 | `[3,B,H,S,d]` | Q/K/V 每个都成为 `[B,H,S,d]` |
| 每头注意力 | `[B,H,S,d]` | 头、样本彼此独立，位置满足因果约束 |
| 合并头 | `[B,S,H*d]=[B,S,D]` | 把头特征接回末轴 |
| 输出投影 | `[B,S,D]` | 混合各头结果并回到模型宽度 |

### 逐步计算

以下等式按定义展开，并写明使用的规则：

1. **联合投影。** 每个位置的输入向量乘一个可学习矩阵，再加偏置。矩阵乘法把末维 D 映射成 3D：

   ```text
   QKV = Linear(X)                         [线性层定义]
   shape(QKV) = [B,S,3D]                  [Linear 只映射最后一轴]
   ```

   将末轴按 Q、K、V 三块切分，每块宽度 D；再把每块 D 拆成 H 份、每份 d：

   ```text
   [B,S,3D] -> [B,S,3,H,d]                [3D = 3*H*d，因 D=H*d]
   permute(3,B,S,H,d -> 3,B,H,S,d)        [交换头轴 H 与序列轴 S]
   ```

2. **单头分数。** 固定样本 b、头 h、query 位置 i、key 位置 j，单头点积沿 d 个特征求和：

   ```text
   raw[b,h,i,j] = sum(k=0..d-1, Q[b,h,i,k] * K[b,h,j,k])   [点积定义]
   scores[b,h,i,j] = raw[b,h,i,j] / sqrt(d)                [缩放点积定义]
   ```

   因而分数矩阵形状为 `[B,H,S,S]`：两个 S 轴分别是 query 位置 i 和 key 位置 j。SDPA 内部负责缩放、softmax 和对 V 加权；这里不重复手写 T08 已覆盖的基础算法。

3. **因果限制。** 定义布尔允许矩阵 `M[i,j] = (j <= i)`。不允许的位置在 softmax 前等价于加负无穷：

   ```text
   masked_scores[b,h,i,j] = scores[b,h,i,j]   若 M[i,j] 为 True
   masked_scores[b,h,i,j] = -infinity        若 M[i,j] 为 False
   ```

   `exp(-infinity)=0`，所以未来位置权重为 0；对每一行做 softmax 后，只在 `j<=i` 的 key 上归一化。这里通过 `is_causal=True` 请求 SDPA 执行这条规则。

4. **加权值与合头。** 每个头在允许的 key 位置上对 V 求加权和：

   ```text
   head[b,h,i,:] = sum(j=0..i, weight[b,h,i,j] * V[b,h,j,:]) [加权和定义]
   shape(heads) = [B,H,S,d]                                   [保留 batch/head/query/特征轴]
   ```

   transpose 把 `[B,H,S,d]` 变成 `[B,S,H,d]`，使同一个 token 的 H 个头相邻；contiguous 复制为连续存储；reshape 把相邻的 H 和 d 合并为 D，因为 `H*d=D`。若跳过轴换位就直接 reshape，内存线性顺序对应的元素可能被错误解释为不同 token 的头特征。

5. **输出投影。** 拼接后每个位置仍是 D 维，末尾线性层混合头输出：

   ```text
   merged[b,i,:] = concat(head[b,0,i,:], ..., head[b,H-1,i,:]) [按头拼接]
   output = Linear(merged)                                    [可学习输出投影]
   shape(output) = [B,S,D]                                    [输出投影保持 D]
   ```

这个 reference 将自回归“只能看当前位置和过去”的保证放在 SDPA 的因果参数中。若改成 `is_causal=False`，未来 token 会进入过去位置的加权和，训练时并行预测就会泄漏答案。

## 符号说明

- `B`：batch（批次）大小，即一次处理的样本数量。
- `S`：sequence length（序列长度），即每个样本的 token 数量。
- `D`：model dimension（模型特征维度），每个 token 的总宽度。
- `H`：number of heads（注意力头数）。
- `d`：head dimension（单头特征维度），满足 `d=D/H`。
- `i`：query token 的序列位置索引；`j`：key/value token 的序列位置索引。
- `W^Q/W^K/W^V`：query/key/value projection（查询/键/值投影）的可学习权重；`W^O`：输出投影权重。
- `M[i,j]`：布尔因果允许关系，成立表示 key 位置 j 可供 query 位置 i 读取。
- `softmax`：把一行实数分数变成总和为 1 的非负权重。

## 程序实例

### 调用方式

从仓库根目录运行完整检查：

```bash
.venv/bin/python learning/batches/b03-attention/t09-causal-mask-and-multihead/test_multihead_attention.py
```

最小推理示例：

```python
import torch
from multihead_attention import CausalMultiHeadSelfAttention

layer = CausalMultiHeadSelfAttention(d_model=16, num_heads=4, dropout_p=0.1)
layer.eval()                         # 推理模式；forward 会传 dropout_p=0.0
x = torch.randn(2, 5, 16)           # 2 个样本、每个 5 个 token、每个 token 16 维
y = layer(x)                         # [2,5,16]
print(y.shape)
```

在仓库根目录先切换到本章节目录，再将上面代码交给 `.venv/bin/python` 运行。`multihead_attention.py` 因此会从当前目录导入；仓库目录名使用连字符，不需要为了这个单文件示例再添加 Python 打包配置。

以下命令会从仓库根目录执行同一示例：

```bash
cd learning/batches/b03-attention/t09-causal-mask-and-multihead
../../../../.venv/bin/python - <<'PY'
import torch
from multihead_attention import CausalMultiHeadSelfAttention

layer = CausalMultiHeadSelfAttention(d_model=16, num_heads=4, dropout_p=0.1).eval()
y = layer(torch.randn(2, 5, 16))
print(y.shape)
PY
```

### 执行路径

`forward` 从 `[B,S,D]` 开始；联合线性层生成 `[B,S,3D]`，reshape 出 Q/K/V 和 H 头，permute 成 SDPA 的 `[B,H,S,d]` 布局；SDPA 在每个样本、每个头内应用因果注意力；transpose 与 contiguous 将注意力结果组织成每个位置的完整头向量；输出线性层返回 `[B,S,D]`。输入由调用方持有，线性层和 SDPA 创建中间 Tensor，forward 返回新输出；模块参数由 `nn.Module` 持有。

### 验收对应

| 验收点 | 程序证据 |
| --- | --- |
| 不同 S、batch、head 数 | `test_output_shapes_across_batch_sequence_and_heads` 覆盖 `(B,S,H)=(1,1,1),(2,5,2),(3,7,4)` |
| 修改未来 token 不影响过去输出 | `test_future_token_changes_do_not_affect_past_outputs` 修改位置 4 及之后，检查位置 0～3 输出逐元素一致 |
| Boolean mask 的 True/False | `test_sdpa_boolean_true_means_allowed` 用两个等分 key，分别只允许一个 value，期望输出直接等于对应 value |
| 显式三角 mask 与因果参数等价 | `test_boolean_lower_triangle_matches_is_causal` 在相同 Q/K/V 上对照两种因果表达 |
| eval 推理 dropout 为零 | `test_inference_always_passes_zero_dropout_to_functional_sdpa` 监视 SDPA 实际收到的参数，并检查重复输出完全一致 |
| D 能被 H 整除 | `test_invalid_head_partition_is_rejected` 检查非法头配置报错 |

这些测试是 CPU 小尺寸语义检查，不覆盖 GPU 内核选择、混合精度误差、反向传播数值、长序列资源开销或训练质量。SDPA 具体选择融合后端由 PyTorch、设备、dtype 和输入布局决定，本章不声称某个后端或性能收益。


## 直观理解

把每个 token 想成小组讨论中的一位同学：H 个头是 H 张不同主题的讨论桌，每桌只依据当前发言和此前发言，不能偷看后面的发言。最后把各桌记录拼起来，再由输出投影整合成每位同学的新笔记。
