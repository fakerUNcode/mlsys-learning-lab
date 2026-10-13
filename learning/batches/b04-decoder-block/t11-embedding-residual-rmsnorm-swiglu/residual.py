"""T11 / B04: shape-preserving residual addition, runnable from the repository root.

这是模块的文档字符串，用来说明本模块的功能。
作用：
- 实现“形状保持不变的残差加法”
- 可以从仓库根目录直接运行
"""

import torch
from torch import nn

# 导入 PyTorch
# torch：张量计算
# nn：神经网络模块（用于继承 nn.Module）


class ResidualAdd(nn.Module):
    """Add same-shaped activations; input/output shape is [B, S, D]."""

    # 定义一个残差加法模块，继承自 nn.Module
    # 输入：
    #   residual：残差路径（skip connection）的激活值
    #   branch：经过变换后的分支激活值
    # 输出：
    #   residual + branch，形状不变，通常是 [B, S, D]


    def forward(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor
    ) -> torch.Tensor:
        # 前向传播函数
        # residual：残差路径的输入（不经过变换）
        # branch：经过某个变换（如 Linear / Attention / MLP）后的输出
        # 返回值：两个张量逐元素相加的结果

        if residual.shape != branch.shape:
            # 安全检查：两个张量的形状必须完全一致
            # 否则无法进行逐元素相加
            raise ValueError(
                f"residual and branch must match exactly, got "
                f"{tuple(residual.shape)} and {tuple(branch.shape)}"
            )
            # 如果形状不一致，直接抛出错误并说明两者的形状

        return residual + branch
        # 核心操作：残差连接
        # 对应公式：
        #     output = x + f(x)
        # 这是 Transformer / ResNet 中最基础的残差结构


def main() -> None:
    # 主函数：用来验证 ResidualAdd 是否正确

    skip = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.float64
    )
    # 构造残差路径（skip connection）的输入
    # 形状：[1, 2, 2]
    # 可以看成：
    #   B=1（batch size）
    #   S=2（sequence length）
    #   D=2（feature dimension）

    update = torch.tensor(
        [[[0.5, -1.0], [2.0, 3.0]]],
        dtype=torch.float64
    )
    # 构造变换分支（branch）的输出
    # 形状同样必须是 [1, 2, 2]

    output = ResidualAdd()(skip, update)
    # 创建一个 ResidualAdd 的临时实例，并调用它
    # 等价于：
    #   layer = ResidualAdd()
    #   output = layer(skip, update)
    # 执行的是 skip + update

    expected = torch.tensor(
        [[[1.5, 1.0], [5.0, 7.0]]],
        dtype=torch.float64
    )
    # 手工计算的期望结果：
    # [[
    #   [1.0 + 0.5, 2.0 + (-1.0)],
    #   [3.0 + 2.0, 4.0 + 3.0]
    # ]]
    # = [[[1.5, 1.0], [5.0, 7.0]]]

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # 断言：实际输出和期望结果必须完全一致
    # rtol=0, atol=0：绝对相等，不允许误差

    print(f"input shapes={tuple(skip.shape)} + {tuple(update.shape)}")
    # 打印两个输入张量的形状（用于调试 / 验证）

    print(f"output={output.tolist()}; parameters=0")
    # 打印最终输出结果
    # parameters=0：
    # 残差加法没有任何可学习参数


if __name__ == "__main__":
    # 当直接运行该文件时，执行 main()
    main()