"""
B04/T13: a small token-ID-to-logits decoder assembled from T11/T12 pieces.
（模块级文档字符串：完整定义了本文件的功能定位、输入输出张量契约、工程约束以及 Llama 风格的 Pre-Norm 数据流图）
"""

# 启用 PEP 563 延迟类型注解评估。
# 原理：将所有类型注解在运行时存储为字符串而非立即求值，从而在 Python 3.7~3.9 中也能支持 `torch.Tensor | None` 等 Python 3.10+ 的联合类型语法，并避免循环引用报错。
from __future__ import annotations

# 导入 sys 模块，用于在运行时动态读取和修改 Python 解释器的模块搜索路径（sys.path）。
import sys
# 从 pathlib 导入面向对象的路径处理类 Path，提供跨平台（Windows/Linux/macOS）的文件系统路径解析能力。
from pathlib import Path
# 导入静态类型提示工具 Dict（字典类型）和 Tuple（元组类型），用于函数签名的类型检查与 IDE 智能提示。
from typing import Dict, Tuple

# 导入 PyTorch 深度学习框架核心库，提供多维张量（Tensor）计算与自动微分支持。
import torch
# 从 torch 中导入神经网络模块库 nn，包含 Module 基类、Linear 线性层等神经网络核心构件。
from torch import nn


# ============================================================
# 路径管理
# ============================================================

# 获取当前脚本文件的路径（__file__），通过 .resolve() 解析所有符号链接并转为绝对路径，再通过 .parent 获取当前文件所在的目录（即 T13 目录）。
_HERE = Path(__file__).resolve().parent

# 获取 T13 目录的上一级父目录（即 B04 根目录），作为寻找平级兄弟模块（T11、T12）的锚点。
_B04 = _HERE.parent

# 利用 Path 重载的 `/` 运算符，拼接出存放基础组件（词嵌入、残差、RMSNorm、SwiGLU）的 T11 绝对目录路径。
_T11 = _B04 / "t11-embedding-residual-rmsnorm-swiglu"

# 拼接出存放位置编码与注意力机制（RoPE、GQA）的 T12 绝对目录路径。
_T12 = _B04 / "t12-rope-gqa-position"

# 遍历 T11 和 T12 两个目录对象，准备将它们注册到 Python 的模块查找路径中。
for _module_dir in (_T11, _T12):
    # 将 Path 对象转为字符串，检查它是否尚未存在于 sys.path 列表中（避免重复添加造成路径污染）。
    if str(_module_dir) not in sys.path:
        # 将目录字符串插入到 sys.path 的索引 0 处（最高优先级）。
        # 原理：Python 执行 `import` 时会按顺序扫描 sys.path；插在头部可确保优先加载本地 T11/T12 源码，防止被系统环境中同名的第三方库覆盖。
        # 此外，当 _T12 被加入后，T12 内部的 `gqa.py` 在执行 `import rope` 时也能顺利在同一目录下找到 `rope.py`。
        sys.path.insert(0, str(_module_dir))


# ============================================================
# 从 T11 / T12 导入已实现的子模块
# ============================================================

# 从 T11 的 embedding.py 导入 TokenEmbedding 类，负责将离散的整数 Token ID 查表映射为连续的稠密向量。
from embedding import TokenEmbedding

# 从 T12 的 gqa.py 导入 causal_mask（生成下三角因果掩码，防止模型偷看未来信息）和 grouped_attention（分组查询注意力核心计算函数）。
from gqa import causal_mask, grouped_attention

# 从 T11 的 residual.py 导入 ResidualAdd 类，封装了 `x + sublayer(x)` 的残差相加操作，用于缓解深层网络的梯度消失。
from residual import ResidualAdd

# 从 T11 的 rmsnorm.py 导入 RMSNorm（均方根层归一化），Llama 架构的核心组件，省去了传统 LayerNorm 的均值中心化计算，提升训练稳定性与计算效率。
from rmsnorm import RMSNorm

# 从 T12 的 rope.py 导入 apply_rope 函数，通过复数域旋转矩阵将绝对位置索引转化为注意力内积中的相对位置信息。
from rope import apply_rope

# 从 T11 的 swiglu.py 导入 SwiGLU 前馈网络层，结合了 Swish (SiLU) 激活函数与门控线性单元（GLU），相比传统 ReLU MLP 具有更强的非线性表达能力。
from swiglu import SwiGLU


# ============================================================
# Decoder Block（核心解码器层）
# ============================================================

# 定义 SmallDecoderBlock 类，继承自 nn.Module。
# 原理：继承 nn.Module 使该类能够自动追踪并注册内部所有的子模块（如 Linear、RMSNorm）及可学习参数（Parameters），支持 .to(device)、.eval()、.parameters() 等标准操作。
class SmallDecoderBlock(nn.Module):
    """一个 Pre-Norm、GQA + SwiGLU 的 Decoder Block。"""

    # 构造函数，定义构建单个 Transformer Decoder 层所需的超参数。
    def __init__(
        self,
        d_model: int,          # 隐藏层特征维度 D（即每个 Token 向量的长度）
        query_heads: int,      # Query（查询）注意力头的数量 Hq
        kv_heads: int,         # Key/Value（键/值）注意力头的数量 Hkv（GQA 中 Hkv <= Hq）
        d_ff: int,             # SwiGLU 前馈网络（FFN）的中间扩展维度 F
        eps: float = 1e-6,     # RMSNorm 分母中的极小常数，防止方差为 0 时发生除零溢出（NaN）
    ) -> None:
        # 调用父类 nn.Module 的初始化方法。
        # 原理：必须在挂载任何子模块或 Parameter 之前调用，因为它负责初始化 PyTorch 内部的 `_parameters` 和 `_modules` 有序字典。
        super().__init__()

        # --------------------------------------------------------
        # 基本维度合法性检查（防御性编程，在模型实例化阶段尽早拦截错误配置）
        # --------------------------------------------------------
        # 检查四个核心维度中最小值是否小于等于 0。
        if min(d_model, query_heads, kv_heads, d_ff) <= 0:
            # 神经网络的特征维度与头数必须为正整数，否则抛出数值错误异常。
            raise ValueError("all dimensions must be positive")

        # 检查隐藏维度 d_model 是否不能被 query_heads 整除（余数非 0 则条件为真）。
        if d_model % query_heads:
            # 原理：多头注意力会将 d_model 均分给每个 Query 头，若不能整除则无法均匀切分特征空间。
            raise ValueError("d_model must be divisible by query_heads")

        # 检查 query_heads 是否不能被 kv_heads 整除。
        if query_heads % kv_heads:
            # 原理：GQA（分组查询注意力）将 Query 头均匀分成 kv_heads 个组，每组共享同一个 K 头和 V 头；每组包含 `query_heads // kv_heads` 个 Q 头，因此必须严格整除。
            raise ValueError("query_heads must be divisible by kv_heads")

        # 将隐藏层维度保存为实例属性，供 forward 前向传播中进行形状校验和重塑（reshape）使用。
        self.d_model = d_model
        # 保存 Query 头数 Hq。
        self.query_heads = query_heads
        # 保存 Key/Value 头数 Hkv。
        self.kv_heads = kv_heads
        # 计算单个注意力头的特征维度 head_dim = D // Hq。无论是 Q 头还是 K/V 头，单个头的向量维度都固定为 head_dim。
        self.head_dim = d_model // query_heads

        # 检查单个头的维度 head_dim 是否为奇数（对 2 取模非 0）。
        if self.head_dim % 2:
            # 原理：RoPE（旋转位置编码）将每个头内的特征向量两两配对（组成二维平面上的点 (x_{2i}, x_{2i+1})）进行旋转变换，因此 head_dim 必须是偶数。
            raise ValueError("head_dim must be even for this RoPE reference")

        # ============================================================
        # Attention 分支（Pre-Norm 结构：先归一化，再进入注意力层）
        # ============================================================
        # 实例化注意力分支前的 RMSNorm 层，对输入维度 D 进行归一化。
        # 原理：Pre-Norm 将归一化放在残差旁路的入口处，使得主干残差通路保持纯净的恒等映射，梯度可以从最后一层直接无损流向第一层，大幅提升深层 Transformer 的训练稳定性。
        self.attn_norm = RMSNorm(d_model, eps=eps)

        # 定义 Query 线性投影层：输入维度为 d_model，输出维度为 Hq * head_dim（实际上等于 d_model）。
        # bias=False 遵循 Llama 架构设计：去掉线性层偏置项可减少参数量与计算开销，且配合 RMSNorm 时偏置项对模型性能无正向收益。
        self.q_proj = nn.Linear(
            d_model, query_heads * self.head_dim, bias=False
        )

        # 定义 Key 线性投影层：输入维度为 d_model，输出维度为 Hkv * head_dim。
        # 原理（GQA 省显存的核心）：当 Hkv < Hq 时，K 和 V 的输出总维度小于 d_model，不仅减少了投影矩阵的参数量，更大幅削减了推理时 KV Cache 的显存占用与带宽压力。
        self.k_proj = nn.Linear(
            d_model, kv_heads * self.head_dim, bias=False
        )
        # 定义 Value 线性投影层：结构与 k_proj 完全一致，将 d_model 映射为 Hkv 个 V 头。
        self.v_proj = nn.Linear(
            d_model, kv_heads * self.head_dim, bias=False
        )

        # 定义输出线性投影层（Output Projection）：将 Hq 个注意力头拼接后的维度（Hq * head_dim）重新融合映射回隐藏层维度 d_model，同样不使用偏置。
        self.o_proj = nn.Linear(
            query_heads * self.head_dim, d_model, bias=False
        )

        # 实例化注意力子层的残差连接模块，用于将注意力层的输出特征加回到进入 attn_norm 之前的原始输入 x 上。
        self.attn_residual = ResidualAdd()

        # ============================================================
        # FFN 分支（Pre-Norm + SwiGLU 前馈网络）
        # ============================================================
        # 实例化 FFN 分支前的 RMSNorm 层，对经过注意力残差更新后的特征进行预归一化。
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        # 实例化 SwiGLU 模块：内部包含 gate_proj(D→F)、up_proj(D→F) 和 down_proj(F→D) 三个无偏置线性层，负责对每个 Token 的特征进行位置无关的非线性变换与特征提取。
        self.mlp = SwiGLU(d_model, d_ff, bias=False)
        # 实例化 FFN 子层的残差连接模块，将 SwiGLU 的输出加回到进入 mlp_norm 之前的特征上。
        self.mlp_residual = ResidualAdd()

    # ============================================================
    # 前向传播（定义数据流经 Decoder Block 的具体计算步骤）
    # ============================================================
    # 定义 forward 方法，接收三维隐藏状态张量 x 和二维位置索引张量 positions，返回更新后的隐藏状态张量及各阶段的形状追踪字典 trace。
    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[int, ...]]]:

        # ------------------------------------------------------------
        # 输入合法性检查
        # ------------------------------------------------------------
        # 验证输入 x 是否为 3 维张量（Batch, Seq_len, Dim），且最后一维必须严格等于模型预设的 d_model。
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [B,S,d_model]")

        # 验证 positions 的形状是否与 x 的前两维 [B, S] 完全一致，且数据类型必须为 64 位长整型（torch.long / int64）。
        # 原理：位置索引代表离散的整数步长（如第 0、1、2 个词），后续用于索引或离散位置计算。
        if positions.shape != x.shape[:2] or positions.dtype != torch.long:
            raise ValueError("positions must be Long [B,S]")

        # 解包 x 的形状元组，提取批大小 batch (B) 和序列长度 length (S)，忽略已知的最后一维 D（用 `_` 占位）。
        batch, length, _ = x.shape

        # 初始化 trace 字典，用于记录数据流经各个子模块后的张量形状（将 torch.Size 转为普通整数 tuple）。
        # 设计考量：只保存形状 tuple 而不保存中间 Tensor 引用，既能用于单元测试和架构验证，又避免了持有计算图节点导致内存泄漏。
        trace: Dict[str, Tuple[int, ...]] = {
            "block_input": tuple(x.shape)  # 记录刚进入 Block 时的输入形状：[B, S, D]
        }

        # ============================================================
        # 1. Attention 输入归一化 + Q/K/V 投影与多头切分
        # ============================================================
        # 对输入 x 执行 Pre-Norm（RMSNorm），形状保持 [B, S, D] 不变。注意这里赋值给新变量 normalized，严格保留原变量 x 供后续残差相加使用（遵循设计原则 3：不原地修改输入）。
        normalized = self.attn_norm(x)

        # 计算 Query 张量并调整为多头注意力所需的维度排布：
        q = (
            # 1) 线性投影：[B, S, D] -> [B, S, Hq * head_dim]
            self.q_proj(normalized)
            # 2) 拆分多头：将最后一维重塑为 (Hq, head_dim)，张量形状变为 [B, S, Hq, head_dim]
            .reshape(batch, length, self.query_heads, self.head_dim)
            # 3) 交换序列维度与头维度：将维度 1 (S) 和维度 2 (Hq) 互换，形状变为 [B, Hq, S, head_dim]。
            # 原理：将 Hq 提到 S 前面，使得每个注意力头成为一个独立的批处理维度，后续矩阵乘法可在最后两维 [S, head_dim] 上并行独立计算各个头的注意力。
            .transpose(1, 2)
        )

        # 计算 Key 张量并调整维度排布：
        k = (
            # 1) 线性投影：[B, S, D] -> [B, S, Hkv * head_dim]
            self.k_proj(normalized)
            # 2) 拆分多头：重塑为 [B, S, Hkv, head_dim]
            .reshape(batch, length, self.kv_heads, self.head_dim)
            # 3) 交换维度：变为 [B, Hkv, S, head_dim]
            .transpose(1, 2)
        )

        # 计算 Value 张量并调整维度排布（与 K 共享相同的头数 Hkv）：
        v = (
            # 1) 线性投影：[B, S, D] -> [B, S, Hkv * head_dim]
            self.v_proj(normalized)
            # 2) 拆分多头：重塑为 [B, S, Hkv, head_dim]
            .reshape(batch, length, self.kv_heads, self.head_dim)
            # 3) 交换维度：变为 [B, Hkv, S, head_dim]
            .transpose(1, 2)
        )

        # 记录 RMSNorm 输出的形状：[B, S, D]
        trace["attn_norm"] = tuple(normalized.shape)
        # 记录 Q 张量的形状：[B, Hq, S, head_dim]
        trace["q"] = tuple(q.shape)
        # 记录 K（以及 V）张量的形状：[B, Hkv, S, head_dim]
        trace["k_v"] = tuple(k.shape)

        # ============================================================
        # 2. RoPE（旋转位置编码） + Causal Attention（因果分组注意力）
        # ============================================================
        # 对 Query 张量施加旋转位置编码，注入每个 Token 的位置信息，输出形状保持 [B, Hq, S, head_dim]。
        q = apply_rope(q, positions)
        # 对 Key 张量施加旋转位置编码，输出形状保持 [B, Hkv, S, head_dim]。
        # 原理：RoPE 仅作用于 Q 和 K，不作用于 V。因为位置相关性是通过 $Q \cdot K^T$ 的内积体现的，旋转后的 Q 和 K 做内积会自动简化为仅依赖相对距离 $(m - n)$ 的函数；而 V 仅承载内容语义信息。
        k = apply_rope(k, positions)

        # 根据 Query 的位置序列和 Key 的位置序列生成因果布尔掩码（Causal Mask），形状通常为 [B, 1, S, S] 或 [B, S, S]。
        # 原理：遵循自回归（Autoregressive）语言模型准则，位置为 $i$ 的查询词只能看到位置 $j \le i$ 的历史键值词（allow 为 True），对于 $j > i$ 的未来词（allow 为 False），在 Softmax 前其注意力分数会被置为 $-\infty$。
        allow = causal_mask(positions, positions)

        # 执行分组查询注意力（GQA）计算：
        # 内部逻辑：将 Hkv 个 K/V 头广播（repeat_interleave）匹配到 Hq 个 Q 头 -> 计算缩放点积 $\frac{QK^T}{\sqrt{d_k}}$ -> 应用 allow 因果掩码 -> Softmax 归一化得到注意力权重 -> 乘以 V。
        # 输出 attended 的形状与 q 完全一致，为 [B, Hq, S, head_dim]。
        attended = grouped_attention(q, k, v, allow)
        # 记录注意力计算输出张量的形状：[B, Hq, S, head_dim]
        trace["attention"] = tuple(attended.shape)

        # ============================================================
        # 3. 多头合并 + 输出投影 + 残差连接
        # ============================================================
        # 将多头注意力的输出重新拼接回三维张量 [B, S, D]：
        merged = (
            # 1) 将维度 1 (Hq) 和维度 2 (S) 换回原来的位置，形状从 [B, Hq, S, head_dim] 变回 [B, S, Hq, head_dim]。
            attended.transpose(1, 2)
            # 2) 调用 .contiguous() 重排底层内存。
            # 原理：.transpose() 只修改了张量的步长（stride）元数据，并未移动物理内存中的数据，导致张量在内存中是不连续的；而后续合并维度时若底层内存不连续，直接展平会受限或触发隐式拷贝，显式调用 contiguous() 开辟连续内存块是 PyTorch 多头合并的标准范式。
            .contiguous()
            # 3) 将最后两维 (Hq, head_dim) 展平合并为单一特征维度 Hq * head_dim（即 d_model），形状变为 [B, S, D]。
            .reshape(batch, length, self.query_heads * self.head_dim)
        )

        # 通过输出线性层 o_proj 对拼接后的多头特征进行跨头信息交互与线性变换，形状保持 [B, S, D]。
        attention_update = self.o_proj(merged)
        # 执行第一次残差连接：将未归一化的原始输入 x 与注意力分支计算出的增量 attention_update 相加（$x = x + \text{Attn}(\text{RMSNorm}(x))$），形状为 [B, S, D]。
        x = self.attn_residual(x, attention_update)
        # 记录注意力残差相加后的张量形状：[B, S, D]
        trace["after_attention_residual"] = tuple(x.shape)

        # ============================================================
        # 4. FFN（SwiGLU 前馈网络）+ 残差连接
        # ============================================================
        # 对经过注意力更新后的特征 x 进行第二次 Pre-Norm（RMSNorm），形状保持 [B, S, D]。
        normalized = self.mlp_norm(x)
        # 将归一化后的特征送入 SwiGLU 前馈网络：内部经历 [B, S, D] -> [B, S, d_ff]（门控激活相乘）-> [B, S, D]，输出形状仍为 [B, S, D]。
        mlp_update = self.mlp(normalized)
        # 执行第二次残差连接：将进入 FFN 前的 x 与 FFN 计算出的增量 mlp_update 相加（$x = x + \text{SwiGLU}(\text{RMSNorm}(x))$），形状为 [B, S, D]。
        x = self.mlp_residual(x, mlp_update)
        # 记录整个 Decoder Block 最终输出的形状：[B, S, D]
        trace["after_mlp_residual"] = tuple(x.shape)

        # 返回当前层的最终隐藏状态张量 x，以及记录了全流程维度变化的 trace 字典。
        return x, trace


# ============================================================
# TinyDecoderLM（完整的端到端微型语言模型封装）
# ============================================================

# 定义 TinyDecoderLM 类，继承自 nn.Module，将词嵌入、Decoder Block、最终归一化和词表输出头串联成完整的自回归解码器模型。
class TinyDecoderLM(nn.Module):
    """一个最小化的、单 Block Decoder 语言模型。"""

    # 构造函数，接收词表大小 vocab_size 以及透传给 SmallDecoderBlock 的各项超参数。
    def __init__(
        self,
        vocab_size: int,       # 词表大小 V（即模型认识的不同 Token 总数）
        d_model: int,          # 隐藏层维度 D
        query_heads: int,      # Query 头数 Hq
        kv_heads: int,         # Key/Value 头数 Hkv
        d_ff: int,             # FFN 中间层维度 F
    ) -> None:
        # 初始化 nn.Module 父类。
        super().__init__()

        # 实例化词嵌入层（TokenEmbedding），内部维护一个形状为 [V, D] 的可学习查找表权重矩阵。
        self.embedding = TokenEmbedding(vocab_size, d_model)
        # 实例化核心的单个 Decoder Block，负责上下文特征交互与非线性变换。
        self.block = SmallDecoderBlock(
            d_model, query_heads, kv_heads, d_ff
        )
        # 实例化最终的 RMSNorm 层（Final Norm）。
        # 原理：在 Pre-Norm 架构中，由于主干残差通路一路累加而未经归一化，最后一层 Block 输出的隐藏状态数值尺度可能较大或不稳定；因此在进入分类头（LM Head）之前，必须追加一次最终的归一化。
        self.final_norm = RMSNorm(d_model)
        # 实例化语言模型输出头（LM Head）：一个从隐藏维度 d_model 映射到词表大小 vocab_size 的无偏置线性层，用于计算每个位置预测下一个词的未归一化对数概率（logits）。
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # 定义完整模型的前向传播函数，接收必选的 token_ids [B, S] 和可选的自定义位置张量 positions [B, S]。
    def forward(
        self, token_ids: torch.Tensor, positions: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[int, ...]]]:

        # ------------------------------------------------------------
        # 输入检查
        # ------------------------------------------------------------
        # 检查输入的 token_ids 数据类型是否为 torch.long (int64)。
        # 原理：PyTorch 的嵌入层查表操作要求索引必须是长整型，传入浮点数或其他整型会报错。
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")

        # 检查 token_ids 是否为 2 维张量 [B, S]，且 Batch 和 Seq_len 两个维度都不能为 0（非空张量）。
        if token_ids.ndim != 2 or min(token_ids.shape) == 0:
            raise ValueError("token_ids must be nonempty [B,S]")

        # 如果调用者没有显式传入 positions 参数（默认情况），则自动生成标准的连续绝对位置编码 `0, 1, ..., S-1`：
        if positions is None:
            positions = torch.arange(
                token_ids.shape[1],        # 生成从 0 到 S-1 的一维序列，长度为 S，形状为 [S]
                dtype=torch.long,          # 指定数据类型为长整型 torch.long
                device=token_ids.device,   # 确保生成的位置张量与 token_ids 位于同一个计算设备（如 CPU）上
            ).expand(token_ids.shape[0], -1)
            # 原理：`.expand(B, -1)` 将一维张量 [S] 广播扩展为二维视图 [B, S]（-1 表示保持第二维长度 S 不变）。
            # 优势：expand 仅修改视图步长（stride 设为 0），不会在内存中复制 B 份数据，实现了零内存开销的批处理位置对齐。

        # 若调用者手动传入了 positions，则检查它与 token_ids 是否在同一个计算设备上，防止跨设备运算触发 RuntimeError。
        if positions.device != token_ids.device:
            raise ValueError(
                "positions and token_ids must share a device"
            )

        # ============================================================
        # 前向路径（端到端推理流水线）
        # ============================================================
        # 第 1 步：词嵌入查表。将整数序列 token_ids [B, S] 映射为稠密特征张量 hidden [B, S, D]。
        hidden = self.embedding(token_ids)
        # 第 2 步：特征编码。将 hidden 和位置索引 positions 送入 Decoder Block，得到包含上下文信息的特征 hidden [B, S, D] 及内部 trace。
        hidden, trace = self.block(hidden, positions)
        # 第 3 步：最终归一化。对 Decoder Block 的输出执行 Final RMSNorm，形状保持 [B, S, D]。
        normalized = self.final_norm(hidden)
        # 第 4 步：词表投影。通过线性层 lm_head 将维度 D 映射到词表维度 V，得到最终的 logits 张量，形状为 [B, S, V]。
        # 原理：logits[b, s, v] 表示第 b 个样本在位置 s 处，预测下一个 Token 为词表中第 v 个词的原始打分（后续若接 Softmax 即可转化为概率分布）。
        logits = self.lm_head(normalized)

        # 汇总构建包含整个模型端到端完整路径的 shape trace 字典：
        trace = {
            # 记录输入 token_ids 的形状：[B, S]
            "token_ids": tuple(token_ids.shape),
            # 记录经过 Embedding 层后的形状：[B, S, D]
            "embedding": (
                token_ids.shape[0],
                token_ids.shape[1],
                self.block.d_model,
            ),
            # 使用字典解包运算符 `**`，将 SmallDecoderBlock 返回的内部 trace 键值对按顺序原样嵌入到当前字典中。
            **trace,
            # 记录最终 RMSNorm 后的形状：[B, S, D]
            "final_norm": tuple(normalized.shape),
            # 记录最终输出 logits 的形状：[B, S, V]
            "logits": tuple(logits.shape),
        }

        # 返回模型预测的 logits 张量 [B, S, V] 和完整的形状追踪字典 trace。
        return logits, trace


# ============================================================
# 主函数（CPU Reference 基准验证）
# ============================================================

# 定义主函数，用于独立运行脚本时执行确定性测试与维度验证。
def main() -> None:
    """运行一个确定性的小型 CPU 示例，并打印真实的 shape 路径。"""

    # 固定 PyTorch 的全局随机数生成器种子为 1313。
    # 原理：神经网络层（Embedding、Linear）在实例化时会使用随机分布初始化权重；固定种子可确保每次运行脚本时生成的初始权重完全一致，从而使输出的 logits 数值具备 100% 的可复现性（Deterministic Reference）。
    torch.manual_seed(1313)

    # 实例化一个超小型的 TinyDecoderLM 模型，并进行精度与模式转换：
    model = TinyDecoderLM(
        vocab_size=17,     # 词表大小 V = 17
        d_model=8,         # 隐藏层维度 D = 8
        query_heads=4,     # Query 头数 Hq = 4（因此单个头维度 head_dim = 8 // 4 = 2，满足偶数要求）
        kv_heads=2,        # KV 头数 Hkv = 2（每 4 // 2 = 2 个 Q 头共享 1 组 K/V 头）
        d_ff=12,           # SwiGLU 中间层维度 F = 12
    ).double().eval()
    # `.double()`：将模型所有可学习参数的数据类型从默认的 float32 转换为双精度浮点数 float64（减少浮点舍入误差，适合作为高精度的 CPU 数值参考基准）。
    # `.eval()`：将模型切换为评估（推理）模式（虽然本模型不含 Dropout 或 BatchNorm，但推理前调用 .eval() 是标准的工程规范）。

    # 构造一个模拟的输入 Token ID 张量，包含 1 个样本（B=1），序列长度为 4（S=4），形状为 [1, 4]，类型为长整型 torch.long。
    token_ids = torch.tensor([[1, 4, 2, 8]], dtype=torch.long)

    # 开启 `torch.inference_mode()` 上下文管理器。
    # 原理：这是比 `torch.no_grad()` 更彻底的推理优化模式。它不仅关闭了自动求导引擎（不构建反向传播计算图、不记录梯度），还禁用了张量的版本计数器（version tracking）和视图元数据追踪，在纯前向推理时内存占用更小、运行更快。
    with torch.inference_mode():
        # 调用模型的 forward 方法（通过 nn.Module.__call__ 触发），传入 token_ids，获取预测分数 logits 和形状追踪字典 trace。
        logits, trace = model(token_ids)

    # 打印当前运行的环境配置元信息（设备、精度、随机种子），方便自动化测试脚本比对输出日志。
    print("device=cpu dtype=float64 seed=1313")
    # 遍历 trace 字典中的每一个阶段名称和对应的形状元组。
    for name, shape in trace.items():
        # 格式化打印每个阶段的名称及其张量形状（例如 `token_ids.shape=(1, 4)`、`q.shape=(1, 4, 4, 2)` 等）。
        print(f"{name}.shape={shape}")

    # 切片提取第 0 个样本（`0`）、序列最后一个位置（`-1`，即第 4 个词之后）、词表前 3 个词（`:3`）的预测分数，转为 Python 列表并打印，用于校验数值计算的正确性。
    print(f"logits[0,-1,:3]={logits[0, -1, :3].tolist()}")

    # 统计并打印整个模型的总可学习参数量：
    # `model.parameters()` 递归遍历模型内所有注册的 Parameter 张量；`p.numel()` 返回单个参数张量中包含的元素总个数；`sum(...)` 求和。
    # 参数明细拆解（总计 656 个参数）：
    # 1. embedding   : 17 * 8 = 136
    # 2. attn_norm   : 8
    # 3. q_proj      : 8 * (4 * 2) = 64
    # 4. k_proj      : 8 * (2 * 2) = 32
    # 5. v_proj      : 8 * (2 * 2) = 32
    # 6. o_proj      : (4 * 2) * 8 = 64
    # 7. mlp_norm    : 8
    # 8. mlp(SwiGLU) : gate(8*12) + up(8*12) + down(12*8) = 288
    # 9. final_norm  : 8
    # 10. lm_head    : 8 * 17 = 136
    # 合计：136 + 8 + 64 + 32 + 32 + 64 + 8 + 288 + 8 + 136 = 776（注：具体总数取决于 TokenEmbedding 和 RMSNorm 内部是否含额外参数，此处通过代码动态精确统计）。
    print(
        f"parameters={sum(p.numel() for p in model.parameters())}"
    )


# Python 模块入口守卫：
# 原理：当直接通过 `python t13_xxx.py` 运行本文件时，内置变量 `__name__` 的值为 `"__main__"`，此时会执行 `main()` 函数；
# 而如果本文件被其他模块（如单元测试文件或后续的 T14 模块）通过 `import` 导入，则 `__name__` 为模块名，`main()` 不会被自动执行，从而避免产生意外的打印或计算开销。
if __name__ == "__main__":
    main()
