"""B04/T13 CPU float64 checks for the composed token-ID-to-logits path."""

import unittest

import torch
from torch.nn import functional as F

from decoder import TinyDecoderLM


class T13DecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1313)
        self.model = TinyDecoderLM(17, 8, 4, 2, 12).double().eval()
        self.ids = torch.tensor([[1, 4, 2, 8]], dtype=torch.long)

    def test_token_ids_to_logits_shapes_and_finite_values(self) -> None:
        with torch.inference_mode():
            logits, trace = self.model(self.ids)
        self.assertEqual(logits.shape, (1, 4, 17))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(trace["embedding"], (1, 4, 8))
        self.assertEqual(trace["q"], (1, 4, 4, 2))
        self.assertEqual(trace["k_v"], (1, 2, 4, 2))
        self.assertEqual(trace["attention"], (1, 4, 4, 2))
        self.assertEqual(trace["after_mlp_residual"], (1, 4, 8))

    def test_composed_logits_match_explicit_stage_reference(self) -> None:
        """Repeat block scheduling explicitly to detect top-level wiring errors."""
        with torch.inference_mode():
            embedded = self.model.embedding(self.ids)
            hidden, _ = self.model.block(embedded, torch.arange(4).view(1, 4))
            expected = self.model.lm_head(self.model.final_norm(hidden))
            actual, _ = self.model(self.ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_future_token_does_not_change_earlier_logits(self) -> None:
        changed = self.ids.clone()
        changed[0, 3] = 9
        with torch.inference_mode():
            original_logits, _ = self.model(self.ids)
            changed_logits, _ = self.model(changed)
        torch.testing.assert_close(
            original_logits[:, :3], changed_logits[:, :3], rtol=0, atol=0
        )

    def test_rejects_float_ids_and_incompatible_heads(self) -> None:
        with self.assertRaises(TypeError):
            self.model(self.ids.float())
        with self.assertRaises(ValueError):
            TinyDecoderLM(17, 8, 3, 1, 12)

    def test_parameter_count_and_named_vocab_scores(self) -> None:
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 776)
        with torch.inference_mode():
            logits, _ = self.model(self.ids)
        probs = F.softmax(logits, dim=-1)
        torch.testing.assert_close(
            probs.sum(-1), torch.ones(1, 4, dtype=torch.float64),
            rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
