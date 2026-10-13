"""Small CPU float64 reference checks for all four T11 modules."""

import unittest

import torch
from torch.nn import functional as F

from embedding import TokenEmbedding
from residual import ResidualAdd
from rmsnorm import RMSNorm
from swiglu import SwiGLU


class T11ReferenceTests(unittest.TestCase):
    def test_embedding_lookup_shape_and_parameter_count(self) -> None:
        layer = TokenEmbedding(9, 4).double()
        with torch.no_grad():
            layer.table.weight.copy_(torch.arange(36, dtype=torch.float64).reshape(9, 4))
        ids = torch.tensor([[1, 8, 1], [0, 3, 4]], dtype=torch.long)
        actual = layer(ids)
        self.assertEqual(actual.shape, (2, 3, 4))
        torch.testing.assert_close(actual, layer.table.weight[ids], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 9 * 4)

    def test_embedding_rejects_float_ids(self) -> None:
        with self.assertRaises(TypeError):
            TokenEmbedding(4, 2)(torch.tensor([[1.0]]))

    def test_residual_values_shape_and_parameter_count(self) -> None:
        left = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        right = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
        layer = ResidualAdd()
        torch.testing.assert_close(layer(left, right), torch.tensor([[4.0, 0.0]], dtype=torch.float64))
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 0)
        with self.assertRaises(ValueError):
            layer(torch.zeros(2, 3), torch.zeros(2, 4))

    def test_rmsnorm_last_axis_and_epsilon(self) -> None:
        layer = RMSNorm(3, eps=1e-6).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        x = torch.tensor([[1.0, 2.0, 2.0], [4.0, 0.0, 3.0]], dtype=torch.float64)
        x32 = x.float()
        expected = (
            x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + 1e-6)
        ).to(x.dtype) * layer.weight
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        self.assertEqual(layer(x).shape, x.shape)
        self.assertEqual(layer.weight.shape, (3,))

    def test_swiglu_branches_activation_shape_and_parameter_count(self) -> None:
        torch.manual_seed(1104)
        layer = SwiGLU(4, 6, bias=False).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        gate = x @ layer.gate_proj.weight.T
        up = x @ layer.up_proj.weight.T
        expected = (F.silu(gate) * up) @ layer.down_proj.weight.T
        actual = layer(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(actual.shape, (2, 3, 4))
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 3 * 4 * 6)

    def test_swiglu_bias_count(self) -> None:
        layer = SwiGLU(4, 6, bias=True)
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 3 * 4 * 6 + 2 * 6 + 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
