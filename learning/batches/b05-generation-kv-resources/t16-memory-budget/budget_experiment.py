"""B05/T16: formula-based memory budget plus a small CUDA allocation audit.

Run from the repository root. The CUDA section uses the T13 toy decoder only;
the separate large-model table is an explicitly hypothetical capacity example.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
T13 = HERE.parents[1] / "b04-decoder-block" / "t13-small-decoder-forward"
sys.path.insert(0, str(T13))

from decoder import TinyDecoderLM  # noqa: E402

GIB = 2**30


@dataclass(frozen=True)
class ModelShape:
    """Only fields needed by the weight/KV equations."""

    name: str
    parameters: int
    layers: int
    kv_heads: int
    head_dim: int
    weight_bytes_per_param: int
    kv_bytes_per_element: int

    def weight_bytes(self) -> int:
        return self.parameters * self.weight_bytes_per_param

    def kv_bytes(self, batch: int, sequence: int) -> int:
        # One key and one value per layer, batch row, KV head, token and feature.
        return (
            2 * self.layers * batch * self.kv_heads * sequence
            * self.head_dim * self.kv_bytes_per_element
        )


def gib(value: int | float) -> float:
    return float(value) / GIB


def actual_cuda_audit() -> dict:
    """Measure model, synthetic KV tensors, and forward peak on the available GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this task requires the allocated/reserved comparison")

    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_at_start, total_bytes = torch.cuda.mem_get_info(device)
    allocator_baseline = torch.cuda.memory_allocated(device)
    reserved_baseline = torch.cuda.memory_reserved(device)

    # These are deliberately tiny T13-compatible weights: their GPU footprint
    # is safe to measure even when the laptop GPU is shared with other work.
    target = TinyDecoderLM(17, 8, 4, 2, 12).to(device=device, dtype=torch.float32).eval()
    draft = TinyDecoderLM(17, 8, 4, 1, 12).to(device=device, dtype=torch.float32).eval()
    torch.cuda.synchronize(device)
    after_models = torch.cuda.memory_allocated(device)
    reserved_after_models = torch.cuda.memory_reserved(device)
    target_shape = ModelShape(
        "tiny_target", sum(p.numel() for p in target.parameters()), 1, 2, 2, 4, 4
    )
    draft_shape = ModelShape(
        "tiny_draft", sum(p.numel() for p in draft.parameters()), 1, 1, 2, 4, 4
    )
    expected_weight_bytes = target_shape.weight_bytes() + draft_shape.weight_bytes()
    target_parameter_tensor_bytes = sum(
        p.numel() * p.element_size() for p in target.parameters()
    )
    draft_parameter_tensor_bytes = sum(
        p.numel() * p.element_size() for p in draft.parameters()
    )

    # Keep explicit K and V allocations alive to model both requests' cache
    # state. The leading layer axis makes the formula general to L layers.
    batch, cached_length = 2, 128
    target_kv = (
        torch.empty((1, batch, 2, cached_length, 2), device=device, dtype=torch.float32),
        torch.empty((1, batch, 2, cached_length, 2), device=device, dtype=torch.float32),
    )
    draft_kv = (
        torch.empty((1, batch, 1, cached_length, 2), device=device, dtype=torch.float32),
        torch.empty((1, batch, 1, cached_length, 2), device=device, dtype=torch.float32),
    )
    torch.cuda.synchronize(device)
    after_kv = torch.cuda.memory_allocated(device)
    reserved_after_kv = torch.cuda.memory_reserved(device)
    expected_kv_bytes = (
        target_shape.kv_bytes(batch, cached_length)
        + draft_shape.kv_bytes(batch, cached_length)
    )
    actual_kv_tensor_bytes = sum(
        t.numel() * t.element_size() for t in (*target_kv, *draft_kv)
    )
    if actual_kv_tensor_bytes != expected_kv_bytes:
        raise AssertionError("KV formula does not match allocated K/V tensor sizes")

    def measure_forward(model: TinyDecoderLM, sequence: int) -> dict:
        # IDs/positions are part of the input working set, not temporary activations.
        ids = (torch.arange(batch * sequence, device=device) % 17).reshape(batch, sequence)
        positions = torch.arange(sequence, device=device).expand(batch, -1)
        # First call initializes the backend's CUDA handles/workspaces. Keep
        # that persistent overhead in the steady baseline, then measure a warm
        # call so it is not mislabeled as per-forward activation memory.
        with torch.inference_mode():
            warm_logits, _ = model(ids, positions)
        del warm_logits
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            logits, _ = model(ids, positions)
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result = {
            "batch": batch,
            "sequence": sequence,
            "input_ids_bytes": ids.numel() * ids.element_size(),
            "positions_bytes": positions.numel() * positions.element_size(),
            "output_logits_bytes": logits.numel() * logits.element_size(),
            "warm_baseline_allocated_bytes": baseline,
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": peak_allocated,
            "peak_extra_allocated_bytes": peak_allocated - baseline,
            "peak_reserved_bytes": peak_reserved,
        }
        del logits, ids, positions
        torch.cuda.synchronize(device)
        return result

    # Prefill uses the existing context. A full-prefix validation call at
    # context+draft_length is a conservative proxy for verifier working space;
    # it is not a cached multi-token verification implementation.
    prefill = measure_forward(target, cached_length)
    verifier = measure_forward(target, cached_length + 5)
    free_at_end, _ = torch.cuda.mem_get_info(device)

    result = {
        "device": torch.cuda.get_device_name(device),
        "total_device_bytes": total_bytes,
        "free_device_bytes_at_start": free_at_start,
        "free_device_bytes_at_end": free_at_end,
        "allocator_baseline_bytes": allocator_baseline,
        "reserved_baseline_bytes": reserved_baseline,
        "weight_measurement": {
            "target_parameters": target_shape.parameters,
            "draft_parameters": draft_shape.parameters,
            "target_plus_draft_formula_bytes": expected_weight_bytes,
            "target_parameter_tensor_bytes": target_parameter_tensor_bytes,
            "draft_parameter_tensor_bytes": draft_parameter_tensor_bytes,
            "unattributed_active_bytes_after_model_load": max(
                0, after_models - allocator_baseline - expected_weight_bytes
            ),
            "allocated_after_models_bytes": after_models,
            "allocated_delta_bytes": after_models - allocator_baseline,
            "reserved_after_models_bytes": reserved_after_models,
        },
        "kv_measurement": {
            "batch": batch,
            "cache_length": cached_length,
            "target_formula_bytes": target_shape.kv_bytes(batch, cached_length),
            "draft_formula_bytes": draft_shape.kv_bytes(batch, cached_length),
            "target_plus_draft_formula_bytes": expected_kv_bytes,
            "target_actual_tensor_bytes": sum(t.numel() * t.element_size() for t in target_kv),
            "draft_actual_tensor_bytes": sum(t.numel() * t.element_size() for t in draft_kv),
            "formula_matches_actual_tensor_bytes": actual_kv_tensor_bytes == expected_kv_bytes,
            "allocated_after_kv_bytes": after_kv,
            "allocated_delta_for_kv_bytes": after_kv - after_models,
            "reserved_after_kv_bytes": reserved_after_kv,
        },
        "forward_peak_measurement": {
            "prefill": prefill,
            "full_prefix_verifier_proxy": verifier,
            "note": "peak extra allocated includes output and live temporaries; it excludes model/KV tensors already live and is not a production verifier measurement",
        },
        "measurement_limits": [
            "CUDA context and non-PyTorch allocations are not represented by memory_allocated",
            "memory_reserved is allocator pool size and must not be added to memory_allocated",
            "external processes can change device free memory during the run",
        ],
    }
    # Keep references live through measurements above, then let process teardown free them.
    _ = target_kv, draft_kv, target, draft
    return result


def hypothetical_capacity(total_bytes: int, free_bytes: int) -> list[dict]:
    """Show why both models, both KV states, verification and headroom matter."""
    target = ModelShape("target_7B_bf16", 7_000_000_000, 32, 8, 128, 2, 2)
    draft = ModelShape("draft_1B_bf16", 1_000_000_000, 24, 8, 64, 2, 2)
    rows = []
    # Two requests demonstrate batch scaling; 5 positions cover a draft block
    # and a possible correction/bonus position in a conservative cache budget.
    for batch in (1, 2, 4):
        context, verify_tokens = 4096, 5
        max_live_length = context + verify_tokens
        target_weights = target.weight_bytes()
        draft_weights = draft.weight_bytes()
        target_kv = target.kv_bytes(batch, max_live_length)
        draft_kv = draft.kv_bytes(batch, max_live_length)
        model_and_kv = target_weights + draft_weights + target_kv + draft_kv
        activation_reserve = 2 * GIB  # explicit planning reserve, not measured
        framework_and_safety_reserve = max(int(total_bytes * 0.10), GIB)
        total_budget = model_and_kv + activation_reserve + framework_and_safety_reserve
        rows.append({
            "scenario": "hypothetical 7B target + 1B draft, bf16",
            "batch": batch,
            "context_tokens": context,
            "verification_tokens": verify_tokens,
            "cache_length_budget": max_live_length,
            "target_weight_gib": gib(target_weights),
            "draft_weight_gib": gib(draft_weights),
            "target_kv_gib": gib(target_kv),
            "draft_kv_gib": gib(draft_kv),
            "activation_reserve_gib": gib(activation_reserve),
            "framework_and_safety_reserve_gib": gib(framework_and_safety_reserve),
            "required_total_gib": gib(total_budget),
            "device_total_gib": gib(total_bytes),
            "device_free_at_audit_start_gib": gib(free_bytes),
            "fits": total_budget <= total_bytes,
            "headroom_or_shortfall_gib": gib(total_bytes - total_budget),
            "headroom_or_shortfall_vs_free_gib": gib(free_bytes - total_budget),
        })
    return rows


def main() -> None:
    torch.manual_seed(1616)
    result = {
        "date": "2026-10-17",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "weight_and_kv_dtype": "float32 for the measured toy audit; hypothetical capacity rows use bf16 (2 bytes/element)",
        "cuda_audit": actual_cuda_audit(),
    }
    result["hypothetical_capacity"] = hypothetical_capacity(
        result["cuda_audit"]["total_device_bytes"],
        result["cuda_audit"]["free_device_bytes_at_start"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
