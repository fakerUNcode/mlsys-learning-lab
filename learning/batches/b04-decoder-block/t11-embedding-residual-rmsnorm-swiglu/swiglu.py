"""T11 / B04: Llama-style gated MLP (SiLU SwiGLU), runnable from repository root.

这是模块的文档字符串，用来说明本模块的功能。
作用：
- 实现 Llama 风格的门控 MLP（SwiGLU）
- 可以从仓库根目录直接运行
"""

import torch
from torch import nn
from torch.nn import functional as F

# 导入 PyTorch 及相关模块
# torch：张量计算
# nn：神经网络模块（Linear、Module）
# F：函数式 API（silu 等）


class SwiGLU(nn.Module):
    """Compute down(SiLU(gate(x)) * up(x)); default bias matches Llama (False)."""

    # 定义一个 SwiGLU 模块，继承自 nn.Module
    # 计算流程：
    #   down(SiLU(gate(x)) * up(x))
    # 这是 Llama 中使用的 FFN（前馈网络）结构


    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        # 构造函数
        # d_model：输入 / 输出特征维度（模型的隐藏维度）
        # d_ff：中间层维度（通常比 d_model 大）
        # bias：是否使用偏置项，Llama 默认不使用偏置

        super().__init__()
        # 调用父类 nn.Module 的初始化方法

        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        # 安全检查：d_model 和 d_ff 必须是正整数

        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        # gate 投影层：
        # 将输入从 [..., d_model] 投影到 [..., d_ff]
        # 用于生成“门控信号”

        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        # up 投影层：
        # 同样是 [d_model -> d_ff]
        # 用于生成“值（value）”

        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        # down 投影层：
        # 将门控相乘后的结果从 [..., d_ff] 投影回 [..., d_model]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播函数
        # x：输入张量，形状为 [..., d_model]

        if x.ndim == 0 or x.shape[-1] != self.gate_proj.in_features:
            raise ValueError("x must have shape [..., d_model]")
            # 安全检查：
            # 1. 输入不能是标量
            # 2. 最后一维的大小必须等于 d_model

        gated = F.silu(self.gate_proj(x))
        # 1. 对输入做 gate 投影：x -> [..., d_ff]
        # 2. 使用 SiLU（也叫 Swish）激活函数
        # SiLU(x) = x * sigmoid(x)
        # 生成“门控信号”

        value = self.up_proj(x)
        # 对输入做 up 投影：x -> [..., d_ff]
        # 生成“值”

        return self.down_proj(gated * value)
        # 1. 门控信号和值逐元素相乘（gating）
        # 2. 通过 down 投影层映射回 d_model
        # 输出形状：[..., d_model]


def main() -> None:
    # 主函数：验证 SwiGLU 是否正确实现

    torch.manual_seed(1104)
    # 固定随机种子，保证结果可复现

    layer = SwiGLU(d_model=4, d_ff=6, bias=False).double()
    # 创建一个 SwiGLU 层
    # d_model=4：输入 / 输出维度
    # d_ff=6：中间层维度
    # bias=False：不使用偏置（与 Llama 一致）
    # .double()：使用 float64 精度

    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 10 - 1
    # 构造输入张量：
    # arange(24)：0~23
    # reshape(2, 3, 4)：形状 [B=2, S=3, D=4]
    # /10 - 1：映射到 [-1, 1.3] 附近，避免过大数值

    actual = layer(x)
    # 调用 SwiGLU 层，计算真实输出

    # Independent reference uses the same stored weights, expanded as matrices.
    # 下面用“独立公式”重新实现一次 SwiGLU，用于验证结果

    gate = x @ layer.gate_proj.weight.T
    # gate 投影：
    # x 乘以 gate_proj 的权重矩阵
    # 等价于 gate_proj(x)（在不使用 bias 时）

    up = x @ layer.up_proj.weight.T
    # up 投影：
    # x 乘以 up_proj 的权重矩阵

    expected = (F.silu(gate) * up) @ layer.down_proj.weight.T
    # 手工计算期望结果：
    # 1. SiLU(gate)
    # 2. SiLU(gate) * up
    # 3. 乘以 down_proj 的权重矩阵
    # 输出形状：[2, 3, 4]

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    # 断言：实际结果和期望结果必须非常接近
    # 允许极小的浮点误差（1e-12）

    count = sum(p.numel() for p in layer.parameters())
    # 统计模型参数总量

    print(f"input.shape={tuple(x.shape)} hidden.shape={tuple(gate.shape)}")
    # 打印输入和中间层（hidden）的形状
    # hidden.shape = [2, 3, 6]

    print(f"output.shape={tuple(actual.shape)} bias=False parameters={count}")
    # 打印输出形状和参数数量

    print(f"max_abs_error={(actual - expected).abs().max().item():.3e}")
    # 打印实际和期望结果之间的最大绝对误差


if __name__ == "__main__":
    # 当直接运行该文件时，执行 main()
    main()