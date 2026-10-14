"""B04/T12：CPU float64 数学与位置语义验收。

通过二维标量公式、逐头循环和 PyTorch SDPA 核对；覆盖错误头映射、
重置位置及矩形因果 mask（掩码）反例。unittest 调度，每项独立持有输入。
运行方式见本章 README；不下载模型权重，不测试真实生成/梯度/性能。
"""

import math
import unittest

import torch
from torch.nn import functional as F

from config_example import describe_config
from gqa import causal_mask, grouped_attention, repeat_kv
from position_demo import compare_positions
from rope import apply_rope


class T12ReferenceTests(unittest.TestCase):
    """每项测试只用 CPU；float64 小张量按 1e-12 数学误差阈值检查。"""

    def test_two_dimensional_hand_rotation(self) -> None:
        x = torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float64)
        actual = apply_rope(x, torch.tensor([[1]]))
        # 用 Python math 的标量表达式独立核对 PyTorch 广播公式。
        expected = torch.tensor([[[[3 * math.cos(1) - 4 * math.sin(1),
                                    3 * math.sin(1) + 4 * math.cos(1)]]]], dtype=x.dtype)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(x, torch.tensor([[[[3.0, 4.0]]]], dtype=x.dtype), rtol=0, atol=0)

    def test_half_split_layout_and_frequencies(self) -> None:
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float64)
        # d=4 时分别配 (1,3)、(2,4)，角度 1 与 0.01。
        expected = torch.tensor([[[[
            math.cos(1) - 3 * math.sin(1),
            2 * math.cos(0.01) - 4 * math.sin(0.01),
            math.sin(1) + 3 * math.cos(1),
            2 * math.sin(0.01) + 4 * math.cos(0.01),
        ]]]], dtype=x.dtype)
        torch.testing.assert_close(apply_rope(x, torch.tensor([[1]])), expected, rtol=1e-12, atol=1e-12)

    def test_norm_zero_and_noncontiguous_input(self) -> None:
        x = torch.arange(48, dtype=torch.float64).reshape(2, 3, 2, 4).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        positions = torch.tensor([[0, 1, 2], [9, 10, 11]])
        result = apply_rope(x, positions)
        torch.testing.assert_close(result.square().sum(-1), x.square().sum(-1), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(result[0, :, 0], x[0, :, 0], rtol=0, atol=0)

    def test_relative_dot_product(self) -> None:
        q = torch.tensor([[[[1.0, 2.0, -1.0, 3.0]]]], dtype=torch.float64)
        k = torch.tensor([[[[4.0, -2.0, 3.0, 1.0]]]], dtype=torch.float64)
        # R(2)q · R(5)k 与 q · R(3)k；同时平移两者位置不改变点积。
        left = (apply_rope(q, torch.tensor([[2]])) * apply_rope(k, torch.tensor([[5]]))).sum()
        right = (q * apply_rope(k, torch.tensor([[3]]))).sum()
        shifted = (apply_rope(q, torch.tensor([[12]])) * apply_rope(k, torch.tensor([[15]]))).sum()
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(left, shifted, rtol=1e-12, atol=1e-12)

    def test_head_mapping_and_extremes(self) -> None:
        x = torch.tensor([10.0, 20.0]).reshape(1, 2, 1, 1)
        self.assertEqual(repeat_kv(x, 4).flatten().tolist(), [10.0, 10.0, 20.0, 20.0])
        self.assertIs(repeat_kv(x, 2), x)  # Multi-Head Attention：一一对应。
        single = x[:, :1]
        self.assertEqual(repeat_kv(single, 4).flatten().tolist(), [10.0] * 4)
        self.assertEqual(x.flatten().tolist(), [10.0, 20.0])

    def test_grouped_attention_against_head_loop(self) -> None:
        generator = torch.Generator().manual_seed(1202)
        q = torch.randn(2, 4, 3, 4, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 2, 5, 4, generator=generator, dtype=torch.float64)
        v = torch.randn(2, 2, 5, 4, generator=generator, dtype=torch.float64)
        allow = causal_mask(torch.tensor([[2, 3, 4], [12, 13, 14]]),
                            torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]))
        expected = torch.empty_like(q)
        # 逐 batch / query 头核对 head//2 的键值头，避免沿用 repeat_kv 的实现。
        for batch in range(2):
            for head in range(4):
                scores = (q[batch, head] @ k[batch, head // 2].T) / 2
                weights = scores.masked_fill(~allow[batch, 0], float("-inf")).softmax(-1)
                expected[batch, head] = weights @ v[batch, head // 2]
        torch.testing.assert_close(grouped_attention(q, k, v, allow), expected, rtol=1e-12, atol=1e-12)

    def test_rectangular_mask_and_sdpa_alignment_counterexample(self) -> None:
        mask = causal_mask(torch.tensor([[6, 7]]), torch.arange(8).reshape(1, 8))
        self.assertEqual(mask[0, 0].int().tolist(), [[1] * 7 + [0], [1] * 8])
        # 当前 query 处在位置 6；全零 scores 使允许的 value 取平均。
        q = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
        k = torch.zeros(1, 1, 7, 2, dtype=q.dtype)
        v = torch.arange(7, dtype=q.dtype).reshape(1, 1, 7, 1).expand(-1, -1, -1, 2)
        allow = causal_mask(torch.tensor([[6]]), torch.arange(7).reshape(1, 7))
        correct = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, is_causal=False, dropout_p=0)
        wrong = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
        torch.testing.assert_close(correct, torch.full_like(correct, 3.0), rtol=0, atol=1e-12)
        self.assertEqual(wrong.flatten().tolist(), [0.0, 0.0])

    def test_incremental_and_chunk_positions(self) -> None:
        # 覆盖单位置、逐 token、prefill+逐 token、分块与非零起点，兼顾 MHA/MQA/GQA。
        cases = [
            (1, 1, 1, 1, 2, 0, (1,)),
            (1, 5, 4, 2, 4, 0, (1, 1, 1, 1, 1)),
            (2, 7, 4, 2, 4, 0, (3, 1, 3)),
            (2, 7, 4, 1, 8, 11, (3, 1, 1, 1, 1)),
            (1, 5, 2, 2, 4, 5, (2, 3)),
        ]
        for case in cases:
            with self.subTest(case=case):
                result = compare_positions(*case)
                for name in ("rotated_q_max_error", "rotated_k_max_error",
                             "chunk_attention_max_error", "sdpa_max_error"):
                    self.assertLessEqual(result[name], 1e-12)

    def test_reset_positions_is_detected(self) -> None:
        self.assertGreater(compare_positions()["reset_position_counterexample"], 1e-3)

    def test_rope_rejects_bad_inputs(self) -> None:
        x = torch.ones(1, 1, 2, 4, dtype=torch.float64)
        for positions in (torch.tensor([[-1, 0]]), torch.tensor([[1, 1]]),
                          torch.tensor([[2, 1]]), torch.tensor([[0, 1], [0, 1]])):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                apply_rope(x, positions)
        with self.assertRaises(TypeError):
            apply_rope(x, torch.tensor([[0.0, 1.0]]))
        with self.assertRaises(ValueError):
            apply_rope(x[..., :3], torch.tensor([[0, 1]]))
        with self.assertRaises(ValueError):
            apply_rope(x, torch.tensor([[0, 1]]), theta=float("nan"))

    def test_attention_rejects_bad_inputs(self) -> None:
        q = torch.ones(1, 4, 1, 2, dtype=torch.float64)
        k = torch.ones(1, 2, 1, 2, dtype=q.dtype)
        for heads in (0, 3):
            with self.assertRaises(ValueError):
                repeat_kv(k, heads)
        with self.assertRaises(ValueError):
            grouped_attention(q, k, k, torch.zeros(1, 1, 1, 1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            grouped_attention(q, k, k, torch.ones(1, 1, 1, 1, dtype=torch.float64))
        with self.assertRaises(ValueError):
            grouped_attention(q, k, k.expand(2, -1, -1, -1), torch.ones(1, 1, 1, 1, dtype=torch.bool))

    def test_frozen_model_config(self) -> None:
        config = describe_config()
        self.assertEqual(config["query_heads"], 32)
        self.assertEqual(config["kv_heads"], 4)
        self.assertEqual(config["head_dim"], 64)
        self.assertEqual(config["queries_per_kv_head"], 8)
        self.assertEqual(config["k_v_proj_weight_shape_each"], [256, 2048])
        self.assertEqual(len(config["revision"]), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
