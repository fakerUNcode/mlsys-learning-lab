"""T10 CUDA 设备测试；无 CUDA 设备时明确跳过。"""

from __future__ import annotations

import unittest

import torch

from attention_cost_gpu import masked_softmax_cuda, stable_softmax_cuda


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-enabled PyTorch runtime")
class AttentionCostCudaTests(unittest.TestCase):
    def test_stable_softmax_runs_on_cuda_and_avoids_overflow(self) -> None:
        device = torch.device("cuda")
        logits = torch.tensor([[1000.0, 999.0, 998.0]], dtype=torch.float64, device=device)

        probabilities = stable_softmax_cuda(logits)

        self.assertEqual(probabilities.device.type, "cuda")
        self.assertTrue(bool(torch.isfinite(probabilities).all().item()))
        torch.testing.assert_close(
            probabilities.sum(dim=-1), torch.ones(1, dtype=torch.float64, device=device),
            rtol=0.0, atol=1e-15,
        )

    def test_all_masked_cuda_row_returns_zero(self) -> None:
        device = torch.device("cuda")
        logits = torch.tensor([[2.0, 1.0], [3.0, 4.0]], dtype=torch.float64, device=device)
        allowed = torch.tensor(
            [[True, False], [False, False]], dtype=torch.bool, device=device
        )

        probabilities = masked_softmax_cuda(logits, allowed)

        expected = torch.tensor(
            [[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64, device=device
        )
        self.assertEqual(probabilities.device.type, "cuda")
        torch.testing.assert_close(probabilities, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
