"""T09 的自动验收：布局尺寸、因果隔离、布尔 mask 与推理 dropout 语义。"""

from __future__ import annotations  # 允许使用尚未定义的类型注解

import unittest  # Python 内置单元测试框架
from unittest.mock import patch  # 用于 mock（替换）指定函数，便于间谍检查

import torch  # PyTorch 核心
from torch.nn import functional as F  # 函数式接口，含 scaled_dot_product_attention

from multihead_attention import CausalMultiHeadSelfAttention  # 被测模块


class CausalMultiHeadAttentionTests(unittest.TestCase):
    """测试套件：覆盖形状、因果性、mask 语义、dropout 行为等。"""

    def test_output_shapes_across_batch_sequence_and_heads(self) -> None:
        """测试不同 B/S/H 组合下输出形状、dtype、device 及有限性。"""
        # 三组输入独立覆盖不同 B、S、H；S=1 也检查最短因果序列。
        for batch, sequence, heads, d_model in [(1, 1, 1, 4), (2, 5, 2, 8), (3, 7, 4, 16)]:
            with self.subTest(batch=batch, sequence=sequence, heads=heads):
                # 固定随机种子，确保结果可复现
                torch.manual_seed(100 + batch + sequence + heads)
                # 实例化被测层，并设为评估模式（关闭 dropout、batchnorm 等训练专属行为）
                layer = CausalMultiHeadSelfAttention(d_model, heads).eval()
                # 随机生成输入：[B, S, D]
                x = torch.randn(batch, sequence, d_model)
                # 前向传播
                output = layer(x)
                # 断言输出形状与输入完全一致
                self.assertEqual(output.shape, (batch, sequence, d_model))
                # 断言 dtype 不变
                self.assertEqual(output.dtype, x.dtype)
                # 断言 device 不变（CPU/GPU）
                self.assertEqual(output.device, x.device)
                # 断言所有元素都是有限数（非 NaN、非 inf）
                self.assertTrue(torch.isfinite(output).all())

    def test_future_token_changes_do_not_affect_past_outputs(self) -> None:
        """测试因果性：修改未来 token 不应影响过去 token 的输出。"""
        torch.manual_seed(109)  # 固定种子
        # 创建被测层，d_model=12, num_heads=3
        layer = CausalMultiHeadSelfAttention(d_model=12, num_heads=3).eval()
        # 原始输入 [2, 6, 12]
        original = torch.randn(2, 6, 12)
        # 复制并大幅修改后半部分（位置 4 和 5，即“未来”token）
        changed = original.clone()
        changed[:, 4:, :] += 100.0

        # 分别前向传播
        original_output = layer(original)
        changed_output = layer(changed)

        # 位置 0..3 的 Q/K/V 计算不得读取位置 4..5；
        # 因此两次运行中这些位置输出应逐元素完全相同。
        torch.testing.assert_close(
            original_output[:, :4], changed_output[:, :4], rtol=0.0, atol=0.0
        )
        # 同时，位置 4 和 5 的输出应该不同（因为输入被改了）
        self.assertGreater(
            (original_output[:, 4:] - changed_output[:, 4:]).abs().max().item(), 0.0
        )

    def test_sdpa_boolean_true_means_allowed(self) -> None:
        """测试 SDPA 布尔 mask 语义：True 表示允许参与注意力。"""
        # q=0 令两个 key 的分数相等（点积均为 0），这样注意力权重会平均分配。
        q = torch.zeros(1, 1, 1, 1)   # [1, 1, 1, 1]
        k = torch.zeros(1, 1, 2, 1)   # [1, 1, 2, 1]
        v = torch.tensor([[[[2.0], [10.0]]]])  # 两个 value，形状 [1,1,2,1]
        # mask 只允许第 0 个 value 时，输出应等于 2.0
        allow_first_only = torch.tensor([[True, False]])
        # mask 只允许第 1 个 value 时，输出应等于 10.0
        allow_second_only = torch.tensor([[False, True]])

        # 调用 SDPA，传入不同 mask
        first = F.scaled_dot_product_attention(q, k, v, attn_mask=allow_first_only)
        second = F.scaled_dot_product_attention(q, k, v, attn_mask=allow_second_only)
        # 验证输出符合预期
        torch.testing.assert_close(first, torch.tensor([[[[2.0]]]]))
        torch.testing.assert_close(second, torch.tensor([[[[10.0]]]]))

    def test_boolean_lower_triangle_matches_is_causal(self) -> None:
        """测试显式下三角布尔 mask 与 is_causal=True 的等价性。"""
        torch.manual_seed(110)
        # 随机生成 q, k, v，形状 [2, 3, 5, 4] 即 [B, H, S, d]
        q = torch.randn(2, 3, 5, 4)
        k = torch.randn(2, 3, 5, 4)
        v = torch.randn(2, 3, 5, 4)
        # 创建下三角布尔矩阵：allowed[i, j] = True 当 j <= i
        allowed = torch.ones(5, 5, dtype=torch.bool).tril()
        # 显式传入 mask
        explicit = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        # 使用 is_causal 隐式生成相同 mask
        implicit = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # 两者应几乎完全相同（浮点误差范围内）
        torch.testing.assert_close(explicit, implicit, rtol=1e-6, atol=1e-6)

    def test_inference_always_passes_zero_dropout_to_functional_sdpa(self) -> None:
        """测试推理时 dropout 被强制设为 0。"""
        # 创建层，设置 dropout_p=0.75（训练时很高），但设为 eval 模式
        layer = CausalMultiHeadSelfAttention(d_model=8, num_heads=2, dropout_p=0.75).eval()
        x = torch.randn(2, 4, 8)
        original_sdpa = F.scaled_dot_product_attention  # 保存原函数
        # 使用 patch 替换 F.scaled_dot_product_attention，用 wraps 包装以便间谍检查
        with patch(
            "multihead_attention.F.scaled_dot_product_attention", wraps=original_sdpa
        ) as sdpa_spy:
            first = layer(x)   # 第一次前向
            second = layer(x)  # 第二次前向
        # 断言 SDPA 被调用了两次（每个 layer(x) 一次）
        self.assertEqual(sdpa_spy.call_count, 2)
        # 断言每次调用时 dropout_p 参数都是 0.0（推理时不 dropout）
        self.assertEqual(sdpa_spy.call_args.kwargs["dropout_p"], 0.0)
        # 断言两次输出完全相同（因为无 dropout，结果确定）
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_invalid_head_partition_is_rejected(self) -> None:
        """测试非法参数（d_model 不能被 num_heads 整除）时抛出 ValueError。"""
        with self.assertRaisesRegex(ValueError, "divisible"):
            CausalMultiHeadSelfAttention(d_model=10, num_heads=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)  # 以详细模式运行所有测试