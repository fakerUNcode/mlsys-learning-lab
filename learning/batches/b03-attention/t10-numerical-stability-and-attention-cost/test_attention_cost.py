"""T10 softmax 边界与注意力资源估算的 CPU unittest 验收。"""

from __future__ import annotations

import unittest

import torch

from attention_cost import (
    estimate_attention_resources,
    masked_softmax,
    naive_softmax,
    stable_softmax,
)


class SoftmaxTests(unittest.TestCase):
    def test_naive_softmax_overflows_but_stable_softmax_is_finite(self) -> None:
        logits = torch.tensor([[1000.0, 999.0, 998.0]], dtype=torch.float64)
        naive = naive_softmax(logits)
        stable = stable_softmax(logits)

        self.assertFalse(bool(torch.isfinite(naive).all()))
        self.assertTrue(bool(torch.isfinite(stable).all()))
        torch.testing.assert_close(stable.sum(dim=-1), torch.ones(1, dtype=torch.float64))
        torch.testing.assert_close(
            stable,
            stable_softmax(logits - 1000.0),
            rtol=0.0,
            atol=1e-15,
        )

    def test_masked_softmax_returns_zero_for_all_masked_row(self) -> None:
        logits = torch.tensor([[2.0, 1.0], [3.0, 4.0]], dtype=torch.float64)
        allowed = torch.tensor([[True, False], [False, False]])

        probabilities = masked_softmax(logits, allowed)

        torch.testing.assert_close(
            probabilities,
            torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        )
        self.assertTrue(bool(torch.isfinite(probabilities).all()))

    def test_masked_softmax_supports_broadcast_mask(self) -> None:
        logits = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            dtype=torch.float64,
        )
        allowed = torch.tensor([[True, False], [True, True]])

        probabilities = masked_softmax(logits, allowed)

        self.assertEqual(tuple(probabilities.shape), tuple(logits.shape))
        self.assertEqual(probabilities[0, 0, 1].item(), 0.0)
        self.assertAlmostEqual(probabilities[1, 1].sum().item(), 1.0)


class ResourceLedgerTests(unittest.TestCase):
    def test_doubling_sequence_quadruplicates_square_terms(self) -> None:
        short = estimate_attention_resources(1, 16, 4096, 64, 2)
        long = estimate_attention_resources(1, 16, 8192, 64, 2)

        self.assertEqual(long.score_bytes, 4 * short.score_bytes)
        self.assertEqual(long.probability_bytes, 4 * short.probability_bytes)
        self.assertEqual(long.explicit_bool_mask_bytes, 4 * short.explicit_bool_mask_bytes)
        self.assertEqual(long.attention_matmul_flops, 4 * short.attention_matmul_flops)
        self.assertEqual(long.qkvo_bytes, 2 * short.qkvo_bytes)
        self.assertEqual(long.kv_cache_bytes_per_layer, 2 * short.kv_cache_bytes_per_layer)

    def test_invalid_resource_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_attention_resources(1, 0, 10, 8, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
