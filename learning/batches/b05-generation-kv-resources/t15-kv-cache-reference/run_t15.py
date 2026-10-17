"""B05/T15 reproducible experiment: full-prefix replay against KV decoding."""

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
T13 = HERE.parents[1] / "b04-decoder-block" / "t13-small-decoder-forward"
sys.path.insert(0, str(T13))
sys.path.insert(0, str(HERE))

from decoder import TinyDecoderLM  # noqa: E402
from kv_cache import forward_with_cache  # noqa: E402


def main() -> None:
    """Run deterministic float64 CPU comparisons and print machine-readable JSON."""
    torch.manual_seed(1515)
    model = TinyDecoderLM(17, 8, 4, 2, 12).double().eval()
    ids = torch.tensor(
        [[1, 4, 2, 8, 3, 6, 5, 9, 10, 0, 11, 12, 7, 13, 14, 15, 16,
          2, 4, 8, 1, 6]], dtype=torch.long
    )
    rows = []
    with torch.inference_mode():
        for prompt_length in (1, 3, 5):
            prompt = ids[:, :prompt_length]
            prompt_positions = torch.arange(prompt_length).view(1, -1)
            cached_logits, cache = forward_with_cache(
                model, prompt, prompt_positions
            )
            full_logits, _ = model(prompt, prompt_positions)
            prefill_error = float((cached_logits - full_logits).abs().max())
            rows.append({
                "prompt_length": prompt_length,
                "phase": "prefill",
                "step": 0,
                "cache_length": int(cache.keys.shape[2]),
                "max_abs_logit_error": prefill_error,
            })

            for step in range(1, 17):
                position = prompt_length + step - 1
                next_id = ids[:, position : position + 1]
                cached_logits, cache = forward_with_cache(
                    model, next_id, torch.tensor([[position]]), cache
                )
                prefix = ids[:, : position + 1]
                full_positions = torch.arange(position + 1).view(1, -1)
                full_logits, _ = model(prefix, full_positions)
                error = float(
                    (cached_logits - full_logits[:, -1:, :]).abs().max()
                )
                rows.append({
                    "prompt_length": prompt_length,
                    "phase": "decode",
                    "step": step,
                    "cache_length": int(cache.keys.shape[2]),
                    "max_abs_logit_error": error,
                })

    summary = {
        "seed": 1515,
        "device": "cpu",
        "dtype": "float64",
        "model": {"vocab_size": 17, "d_model": 8, "query_heads": 4,
                  "kv_heads": 2, "head_dim": 2, "d_ff": 12,
                  "decoder_blocks": 1},
        "tolerance": {"rtol": 0, "atol": 1e-12},
        "prompt_lengths": [1, 3, 5],
        "incremental_steps_per_prompt": 16,
        "max_abs_logit_error": max(row["max_abs_logit_error"] for row in rows),
        "checks": len(rows),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
