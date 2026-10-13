"""T11 / B04: token ID -> learned vector lookup, runnable from the repository root.

这一行是模块的文档字符串（docstring），用来说明这个模块的用途。
作用：
- 把 token ID 映射成“可学习的向量”（embedding lookup）
- 可以从仓库根目录直接运行
"""

import torch
from torch import nn

# 导入 PyTorch 核心库
# torch：张量、自动求导、随机数、设备管理等
# nn：神经网络模块，这里用来使用 nn.Embedding 和 nn.Module


class TokenEmbedding(nn.Module):
    """Map integer token IDs shaped [B, S] to vectors shaped [B, S, D]."""

    # 定义一个名为 TokenEmbedding 的类
    # 继承自 nn.Module，表示这是一个 PyTorch 神经网络模块
    # 输入：整数 token ID，形状是 [B, S]
    #   B = batch size（一批有多少句话）
    #   S = sequence length（每句话有多少个 token）
    # 输出：向量，形状是 [B, S, D]
    #   D = d_model（每个 token 用多长的向量表示）

    def __init__(self, vocab_size: int, d_model: int) -> None:
        # 类的构造函数（初始化方法）
        # vocab_size：词表大小（一共有多少个不同的 token）
        # d_model：每个 token 对应的向量维度
        super().__init__()
        # 调用父类 nn.Module 的初始化方法
        # 这是写 PyTorch 模型的固定写法

        self.table = nn.Embedding(vocab_size, d_model)
        # 创建一个 embedding 表（查找表）
        # 本质是一个可训练的参数矩阵，形状是 [vocab_size, d_model]
        # 每个 token ID 对应这个表里的一行向量
        # PyTorch 会自动把它注册为模型参数（Parameter）

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 前向传播函数：定义“输入数据后如何计算输出”
        # token_ids：输入的整数张量，形状 [B, S]
        # 返回值：向量张量，形状 [B, S, D]

        # nn.Embedding performs indexed row lookup; IDs are indices, not floats.
        # 注释：nn.Embedding 做的是“按行查表”
        # token ID 是“行号（索引）”，不是浮点数，不会被当成数值计算

        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")
            # 安全检查：Embedding 的索引必须是整数类型
            # 如果不是 torch.long（int64），直接报错
            # 这样可以尽早发现错误，避免后面出现难以排查的 bug

        return self.table(token_ids)
        # 用 token_ids 去 embedding 表中查找对应的向量
        # self.table(...) 等价于 self.table.forward(...)
        # 输出形状：[B, S, D]


def main() -> None:
    # 主函数：用来测试这个模块是否工作正常
    # 当直接运行本文件时会被调用

    layer = TokenEmbedding(vocab_size=7, d_model=3).double()
    # 创建一个 TokenEmbedding 层
    # vocab_size=7：词表里有 7 个 token（ID 从 0 到 6）
    # d_model=3：每个 token 用长度为 3 的向量表示
    # .double()：把所有参数转换成 float64（双精度）
    # 这样做是为了在测试中避免浮点误差

    with torch.no_grad():
        # 关闭梯度计算（测试用，不训练模型）
        # 在这个代码块里不会记录梯度，也不会占用额外显存
        layer.table.weight.copy_(
            torch.arange(21, dtype=torch.float64).reshape(7, 3)
        )
        # 手动给 embedding 表赋值，方便验证是否正确
        # torch.arange(21)：生成 0~20 的连续数字
        # reshape(7, 3)：变成 7 行 3 列，正好对应 7 个 token、每个 3 维
        # copy_()：把这些值拷贝进 embedding 表的权重中
        # 此时：
        # ID 0 -> [0, 1, 2]
        # ID 1 -> [3, 4, 5]
        # ID 2 -> [6, 7, 8]
        # ...以此类推

    ids = torch.tensor([[1, 4, 1], [0, 6, 2]], dtype=torch.long)
    # 构造输入 token ID
    # 形状是 [2, 3]：
    #   两句话（batch size = 2）
    #   每句话 3 个 token
    # dtype=torch.long：必须是整数类型

    actual = layer(ids)
    # 调用模型，进行 embedding 查找
    # layer(ids) 会自动调用 forward()
    # actual 是真实计算得到的 embedding 结果

    # Independent reference: each ID selects the corresponding table row.
    # 下面用“独立参考实现”来验证结果是否正确
    # 直接用 PyTorch 的索引操作，从权重表中取行
    expected = layer.table.weight[ids]
    # layer.table.weight[ids] 表示：
    # 对 ids 中的每一个 ID，取出权重矩阵中对应的一行
    # 结果形状也是 [2, 3, 3]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # 断言：actual 和 expected 必须完全一致
    # rtol=0, atol=0：要求绝对相等，不考虑任何误差
    # 如果结果不一样，这里会直接报错

    print(f"ids.shape={tuple(ids.shape)} output.shape={tuple(actual.shape)}")
    # 打印输入和输出的形状
    # ids.shape=(2, 3)
    # output.shape=(2, 3, 3)

    print(f"parameters={sum(p.numel() for p in layer.parameters())}; max_abs_error=0")
    # 打印模型参数总量
    # sum(p.numel() for p in layer.parameters())：所有参数的元素个数
    # 这里只有一张 embedding 表：7 * 3 = 21 个参数
    # max_abs_error=0：说明测试完全通过


if __name__ == "__main__":
    # 当直接运行这个 Python 文件时，才执行下面的代码
    # 如果是被别的模块 import，则不会自动运行
    main()