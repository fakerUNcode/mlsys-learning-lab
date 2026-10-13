"""T11 / B04: Llama-style RMSNorm, runnable from the repository root.

这是模块的文档字符串，用来说明本模块的功能。
作用：
- 实现 Llama 风格的 RMSNorm（Root Mean Square Layer Normalization）
- 可以从仓库根目录直接运行
"""

import torch
from torch import nn

# 导入 PyTorch
# torch：张量计算、数学函数（如 rsqrt）
# nn：神经网络模块，用来定义 nn.Module 和 nn.Parameter


class RMSNorm(nn.Module):
    """Normalize only the last axis of [..., D], then apply a learned scale."""

    # 定义一个 RMSNorm 类，继承自 nn.Module
    # 只针对“最后一个维度”做归一化
    # 归一化后再乘上一个可学习的缩放参数（weight）

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        # 构造函数
        # d_model：输入特征的维度（最后一个维度的大小）
        # eps：一个很小的数，用于防止除以 0（数值稳定性）

        super().__init__()
        # 调用父类 nn.Module 的初始化方法

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        # 安全检查：特征维度必须是正整数

        if eps <= 0:
            raise ValueError("eps must be positive")
        # 安全检查：eps 必须是正数

        self.weight = nn.Parameter(torch.ones(d_model))
        # 定义一个可学习的缩放参数，形状为 [D]
        # 初始值全为 1
        # nn.Parameter：告诉 PyTorch 这是需要训练的参数
        # 所有前导维度（batch、sequence 等）共享这一个向量

        self.eps = eps
        # 保存 eps，供 forward 计算时使用

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播函数
        # x：输入张量，形状为 [..., D]
        # 返回值：归一化后的张量，形状不变

        if x.ndim == 0 or x.shape[-1] != self.weight.numel():
            raise ValueError("x must have shape [..., d_model]")
            # 安全检查：
            # 1. 输入不能是标量（至少要有最后一个维度）
            # 2. 最后一维的大小必须等于 d_model

        input_dtype = x.dtype
        # 记录输入的数据类型（如 float16 / float32 / float64）
        # 后面要把结果转换回原来的类型

        x_fp32 = x.to(torch.float32)
        # 将输入转换成 float32
        # RMSNorm 在 Llama 中通常在 float32 下计算统计量
        # 这样可以提高数值稳定性，尤其是使用低精度训练时

        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        # 计算均方值：
        # x_fp32.pow(2)：每个元素平方
        # .mean(dim=-1, keepdim=True)：在最后一个维度求平均
        # keepdim=True：保持维度，便于后续广播计算
        # 结果形状：[..., 1]

        normalized = x_fp32 * torch.rsqrt(mean_square + self.eps)
        # RMSNorm 的核心公式：
        # rsqrt = 1 / sqrt(...)
        # x / sqrt(mean(x^2) + eps)
        # 注意：没有减去均值（这是 RMSNorm 与 LayerNorm 的区别）

        return self.weight * normalized.to(input_dtype)
        # 1. 把归一化结果转换回原始数据类型
        # 2. 乘上可学习的缩放参数 self.weight
        # 3. 返回最终结果


def main() -> None:
    # 主函数：用来验证 RMSNorm 是否实现正确
    # 只有直接运行本文件时才会执行

    layer = RMSNorm(d_model=4, eps=1e-6)
    # 创建一个 RMSNorm 层
    # 特征维度为 4
    # eps = 1e-6

    x = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.0, 0.0, 0.0, 0.0]])
    # 构造输入张量
    # 形状：[2, 4]
    # 两行数据，每行 4 个特征

    actual = layer(x)
    # 调用 RMSNorm 层，计算归一化结果
    # actual 是模型的真实输出

    # Independent expression makes the normalized axis and epsilon explicit.
    # 下面用“独立公式”重新实现一次 RMSNorm
    # 用来和模型输出做对比，验证实现是否正确

    x_fp32 = x.float()
    # 转换成 float32

    expected = (
        x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).to(x.dtype) * layer.weight
    # 手工计算期望结果：
    # 1. square()：平方
    # 2. mean(dim=-1, keepdim=True)：在最后一维求平均
    # 3. + 1e-6：防止除零
    # 4. rsqrt()：求倒数平方根
    # 5. 乘回输入，完成归一化
    # 6. 转回原 dtype
    # 7. 乘上可学习参数 weight

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # 断言：模型输出和手工计算结果必须完全一致
    # rtol=0, atol=0：绝对相等，不考虑误差

    print(f"input.shape={tuple(x.shape)} output.shape={tuple(actual.shape)}")
    # 打印输入和输出的形状
    # 都是 [2, 4]

    print(f"normalized_axis=-1 eps={layer.eps} parameters={layer.weight.numel()}")
    # 打印归一化的轴、eps 和参数数量
    # 参数数量 = d_model = 4

    print(f"output={actual.tolist()}")
    # 打印最终归一化后的输出值


if __name__ == "__main__":
    # 当直接运行这个文件时，执行 main()
    main()