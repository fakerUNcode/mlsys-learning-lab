"""B05/T15 correctness checks: cached prefill/decode versus full-prefix replay."""

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
T13 = HERE.parents[1] / "b04-decoder-block" / "t13-small-decoder-forward"
sys.path.insert(0, str(T13))
sys.path.insert(0, str(HERE))

from decoder import TinyDecoderLM  # noqa: E402
from kv_cache import forward_with_cache  # noqa: E402


class T15KVCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1515)
        self.model = TinyDecoderLM(17, 8, 4, 2, 12).double().eval()
        # Deterministic IDs include prompt and all 16 subsequent decode tokens.
        self.ids = torch.tensor(
            [[1, 4, 2, 8, 3, 6, 5, 9, 10, 0, 11, 12, 7, 13, 14, 15, 16,
              2, 4, 8, 1, 6]], dtype=torch.long
        )

    def _compare_prompt_and_decode(self, prompt_length: int) -> float:
        prompt = self.ids[:, :prompt_length]
        prompt_positions = torch.arange(prompt_length).view(1, -1)
        with torch.inference_mode():
            cached_prompt_logits, cache = forward_with_cache(
                self.model, prompt, prompt_positions
            )
            full_prompt_logits, _ = self.model(prompt, prompt_positions)
            torch.testing.assert_close(
                cached_prompt_logits, full_prompt_logits, rtol=0, atol=1e-12
            )
            self.assertEqual(cache.keys.shape[2], prompt_length)
            self.assertEqual(cache.values.shape[2], prompt_length)

            max_error = float(
                (cached_prompt_logits - full_prompt_logits).abs().max().item()
            )
            for step in range(16):
                position = prompt_length + step
                next_id = self.ids[:, position : position + 1]
                next_position = torch.tensor([[position]], dtype=torch.long)
                cached_logits, cache = forward_with_cache(
                    self.model, next_id, next_position, cache
                )
                full_prefix = self.ids[:, : position + 1]
                full_positions = torch.arange(position + 1).view(1, -1)
                replay_logits, _ = self.model(full_prefix, full_positions)
                reference = replay_logits[:, -1:, :]
                torch.testing.assert_close(
                    cached_logits, reference, rtol=0, atol=1e-12
                )
                expected_length = prompt_length + step + 1
                self.assertEqual(cache.keys.shape[2], expected_length)
                self.assertEqual(cache.values.shape[2], expected_length)
                self.assertEqual(cache.positions.shape[1], expected_length)
                error = float((cached_logits - reference).abs().max().item())
                max_error = max(max_error, error)
            return max_error

    def test_three_prompt_lengths_and_sixteen_incremental_steps(self) -> None:
        errors = [self._compare_prompt_and_decode(length) for length in (1, 3, 5)]
        self.assertTrue(all(error <= 1e-12 for error in errors), errors)

    def test_existing_cache_is_not_mutated_or_double_appended(self) -> None:
        prompt = self.ids[:, :3]
        with torch.inference_mode():
            _, cache0 = forward_with_cache(
                self.model, prompt, torch.arange(3).view(1, 3)
            )
            old_k = cache0.keys.clone()
            old_v = cache0.values.clone()
            _, cache1 = forward_with_cache(
                self.model, self.ids[:, 3:4], torch.tensor([[3]]), cache0
            )
        self.assertEqual(cache0.keys.shape[2], 3)
        self.assertEqual(cache1.keys.shape[2], 4)
        torch.testing.assert_close(cache0.keys, old_k, rtol=0, atol=0)
        torch.testing.assert_close(cache0.values, old_v, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
