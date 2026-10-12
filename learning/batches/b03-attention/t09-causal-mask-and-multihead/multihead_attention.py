"""T09 因果多头自注意力的最小实现。

本模块位于 B03 Attention 学习流程：
输入 [B, S, D] 的 token 表示，经 Q/K/V 线性投影、拆分 H 个头、
执行因果缩放点积注意力（SDPA）、合并头并投影，返回 [B, S, D]。
这是便于检查形状和语义的 PyTorch reference，不含位置编码、残差、归一化或 FFN。
"""

from __future__ import annotations  # 允许类型注解中使用尚未定义的类型（如 CausalMultiHeadSelfAttention）

import torch  # PyTorch 核心库
from torch import nn  # 神经网络模块，提供 Layer、Module 等基类
from torch.nn import functional as F  # 函数式接口，含 scaled_dot_product_attention


class CausalMultiHeadSelfAttention(nn.Module):
    """对 [B, S, D] 输入计算因果多头自注意力。

    参数:
        d_model: 每个 token 的总特征宽度 D（模型维度）。
        num_heads: 并行注意力头数 H，要求 D 可被 H 整除。
        dropout_p: 训练时 SDPA 对注意力权重使用的 dropout 概率。
    返回:
        [B, S, D] 张量；输入和输出都按 batch、序列、特征排列。
    并发要求:
        模块不保存逐次 forward 的临时张量；遵守 PyTorch 模块常规规则，
        不要在另一个线程执行 forward 时原地修改参数或输入。训练态 dropout 使用随机数。
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0) -> None:
        """初始化层：校验参数、定义线性投影。"""
        super().__init__()  # 调用 nn.Module 的初始化，注册子模块

        # ---- 参数合法性校验 ----
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")  # 维度/头数必须为正
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")  # 确保每个头能均分特征
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")  # dropout 概率必须在合理范围

        # ---- 保存配置 ----
        self.d_model = d_model      # 模型总维度 D
        self.num_heads = num_heads  # 头数 H
        self.head_dim = d_model // num_heads  # 每个头的维度 d = D / H
        self.dropout_p = float(dropout_p)     # 训练时 dropout 概率

        # ---- 定义子层 ----
        # 一次线性层同时产出 Q、K、V，输出末轴按 [Q全部特征, K全部特征, V全部特征] 排列。
        # nn.Linear(in_features, out_features) 只作用于最后一轴：
        #   [B, S, D] -> [B, S, 3*D]，且不混合 batch 维度和位置维度。
        self.qkv_projection = nn.Linear(d_model, 3 * d_model)

        # 合并头后再学习一次 D -> D 映射，对应论文中的输出投影 W^O。
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行投影、拆头、因果注意力、合头和输出投影。

        参数:
            x: 浮点 Tensor [B, S, D]，其中 B、S、D 均非零且 D 等于构造时 d_model。
        返回:
            与 x 同形状、device 和 dtype 的浮点 Tensor。
        并发要求:
            同类的不同 forward 可并行；单次调用期间调用方不得原地修改 x。
        """
        # ========== 1. 输入校验 ==========
        if x.ndim != 3:
            raise ValueError("x must have shape [B,S,D]")  # 必须是 3 维张量
        # Python 读取 x.shape发现它是一个长度为 3 的可迭代对象，按从左到右的顺序，依次把元素赋值给左边的变量：batch = 2，sequence = 5，features = 64
        batch, sequence, features = x.shape  # 解包形状：B=batch, S=sequence, D=features
        if batch == 0 or sequence == 0 or features == 0:
            raise ValueError("B, S, and D must be non-zero")  # 任一维度不能为 0
        if features != self.d_model:
            raise ValueError(f"last dimension must equal d_model={self.d_model}")  # 特征维必须匹配
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")  # 必须是浮点类型

        # ========== 2. QKV 联合投影 ==========
        # Linear 返回 [B, S, 3*D]。
        # reshape 把连续的 3*D 特征重新解释为 (3, H, d) 三个轴：
        #   元素顺序不变，只改变视图，形状变为 [B, S, 3, H, d]
        # 其中 3 对应 Q/K/V，H 是头数，d 是每个头的维度。
        qkv = self.qkv_projection(x).reshape(
            batch, sequence, 3, self.num_heads, self.head_dim
        )

        # permute(2, 0, 3, 1, 4) 交换轴次序：
        #   原轴: (B, S, 3, H, d) -> 新轴: (3, B, H, S, d)
        # 目的：把 Q/K/V 轴（原第2轴）移到最前，方便用 unbind 拆分。
        # unbind(dim=0) 沿第0维拆开，即沿 Q/K/V 那一维拆成三个独立的张量 q, k, v，每个形状都是 [B, H, S, d]——正好是 F.scaled_dot_product_attention 想要的布局。
        # .permute将原来的轴顺序(0, 1, 2, 3, 4)重排为(2, 0, 3, 1, 4)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # ========== 3. 缩放点积注意力（SDPA） ==========
        # SDPA 期望输入布局为 [B, H, S, d]（batch, head, sequence, head_dim）。
        # is_causal=True 自动应用因果 mask：query 位置 i 只能读取 key 位置 j <= i。
        #   （即当前 token 不能看到未来 token）
        # SDPA 的布尔 mask 语义：True 表示允许参与注意力，False 表示屏蔽。
        # 此处使用内建因果约束，不另传 mask。
        #
        # 注意：函数式 F.scaled_dot_product_attention 不会自行检查 module.training，
        # 因此推理时必须显式把 dropout 设为 0，否则会错误地应用 dropout。
        dropout_p = self.dropout_p if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            q,                   # Query: [B, H, S, d]
            k,                   # Key:   [B, H, S, d]
            v,                   # Value: [B, H, S, d]
            attn_mask=None,      # 不额外传 mask，由 is_causal 控制
            dropout_p=dropout_p, # 训练时按设定概率 dropout，推理时为 0
            is_causal=True,      # 启用因果注意力（下三角 mask）
        )  # attended 形状: [B, H, S, d]

        # ========== 4. 合并多头并输出投影 ==========
        # transpose(1, 2) 交换 H 和 S 轴：[B, H, S, d] -> [B, S, H, d]
        # 但 transpose 通常只改 stride、返回非连续视图；reshape 仍会保留逻辑元素顺序，
        # 只是可能暗中复制。这里显式 contiguous()，让后续合头前的连续化步骤可见。
        # reshape 将 (H, d) 合并回 D： [B, S, H, d] -> [B, S, H*d] = [B, S, D]
        merged = attended.transpose(1, 2).contiguous().reshape(batch, sequence, features)

        # 最后通过输出投影 W^O 做线性变换，返回 [B, S, D]。
        return self.output_projection(merged)
