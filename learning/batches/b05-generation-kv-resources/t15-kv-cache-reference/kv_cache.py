"""B05/T15: explicit single-layer KV-cache reference for the T13 tiny decoder.

The cache stores post-RoPE keys, values, and their absolute positions. It is
owned by the caller and each call returns a new cache value; no global state is
modified, so independent requests can use the same model concurrently provided
the model parameters are not changed during inference.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from decoder import TinyDecoderLM
from gqa import causal_mask, grouped_attention
from rope import apply_rope, validate_positions


class KVCache(NamedTuple):
    """One layer's accumulated keys, values, and absolute key positions.

    ``NamedTuple`` keeps the cache immutable as a container while retaining
    named access (``cache.keys``) and tuple unpacking. Tensor contents remain
    ordinary PyTorch tensors; this module never writes into an old cache.
    """

    keys: torch.Tensor       # [B, Hkv, N, d], already rotated by RoPE
    values: torch.Tensor     # [B, Hkv, N, d], content vectors (not rotated)
    positions: torch.Tensor  # [B, N], absolute token positions


def forward_with_cache(
    model: TinyDecoderLM,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    past: KVCache | None = None,
) -> tuple[torch.Tensor, KVCache]:
    """Run a prefill or one-token decode and return logits plus updated cache.

    Args:
        model: T13's one-block decoder. Caller keeps it in eval/inference mode.
        token_ids: New token IDs [B,T]; T may be a prompt length for prefill,
            but must be 1 when extending an existing cache.
        positions: Absolute positions [B,T] for only these new tokens.
        past: Previous layer cache or None for prefill.

    Returns:
        Logits [B,T,V] for the newly supplied tokens and a new cache of length
        old_length + T. Inputs and an existing cache are never modified.
    """
    # Embedding uses integer IDs as row addresses, so floating-point IDs are
    # rejected before any model computation starts.
    if token_ids.dtype != torch.long or token_ids.ndim != 2:
        raise TypeError("token_ids must be Long [B,T]")
    if token_ids.shape[1] == 0:
        raise ValueError("token_ids must contain at least one token")
    if positions.shape != token_ids.shape or positions.dtype != torch.long:
        raise ValueError("positions must be Long [B,T] matching token_ids")
    if positions.device != token_ids.device or model.embedding.table.weight.device != token_ids.device:
        raise ValueError("model, token_ids, and positions must share a device")
    validate_positions(positions)

    # A cache is valid only for the immediately following absolute position.
    # This catches accidental replay of a whole prefix and skipped/duplicated
    # positions before they can silently corrupt RoPE or the attention mask.
    if past is not None:
        if token_ids.shape[1] != 1:
            raise ValueError("cached decode accepts exactly one new token")
        if past.keys.shape[:2] != (token_ids.shape[0], model.block.kv_heads):
            raise ValueError("cache batch/head dimensions do not match model")
        if past.keys.shape != past.values.shape:
            raise ValueError("cached keys and values must have matching shapes")
        if past.positions.shape != (token_ids.shape[0], past.keys.shape[2]):
            raise ValueError("cache positions do not match cached sequence length")
        if past.keys.device != token_ids.device or past.values.device != token_ids.device:
            raise ValueError("cache and token_ids must share a device")
        if past.keys.dtype != model.block.k_proj.weight.dtype:
            raise ValueError("cache dtype must match model projection dtype")
        if not torch.equal(positions[:, :1], past.positions[:, -1:] + 1):
            raise ValueError("new token position must immediately follow cache")

    block = model.block
    batch, current_length = token_ids.shape
    # Embedding changes integer IDs [B,T] into learned floating vectors [B,T,D].
    # The temporary hidden states live only for this call; the caller owns the
    # returned cache, which is the state carried between decode calls.
    hidden = model.embedding(token_ids)
    normalized = block.attn_norm(hidden)

    # Project only new hidden states. Q is transient; only K/V can serve future
    # attention because future queries have not been computed yet.
    q = block.q_proj(normalized).reshape(
        batch, current_length, block.query_heads, block.head_dim
    ).transpose(1, 2)
    new_k = block.k_proj(normalized).reshape(
        batch, current_length, block.kv_heads, block.head_dim
    ).transpose(1, 2)
    new_v = block.v_proj(normalized).reshape(
        batch, current_length, block.kv_heads, block.head_dim
    ).transpose(1, 2)

    q = apply_rope(q, positions)
    # Rotate K before saving it, so prior positions are not transformed again
    # during each incremental decode call.
    new_k = apply_rope(new_k, positions)

    # Prefill starts with no history. Decode concatenates old and new sequence
    # axes once; torch.cat returns new tensors and leaves `past` untouched.
    if past is None:
        all_k, all_v = new_k, new_v
        all_positions = positions
    else:
        all_k = torch.cat((past.keys, new_k), dim=2)
        all_v = torch.cat((past.values, new_v), dim=2)
        all_positions = torch.cat((past.positions, positions), dim=1)

    # The query axis is T while the key axis is the full cached length N. Using
    # absolute positions creates the required rectangular mask [B,1,T,N].
    allow = causal_mask(positions, all_positions)
    attended = grouped_attention(q, all_k, all_v, allow)
    merged = attended.transpose(1, 2).contiguous().reshape(
        batch, current_length, block.d_model
    )
    hidden = block.attn_residual(hidden, block.o_proj(merged))
    hidden = block.mlp_residual(hidden, block.mlp(block.mlp_norm(hidden)))
    # Only logits for the newly supplied positions are returned. Older logits
    # are not cache state and are not needed to predict later tokens.
    logits = model.lm_head(model.final_norm(hidden))

    return logits, KVCache(all_k, all_v, all_positions)
