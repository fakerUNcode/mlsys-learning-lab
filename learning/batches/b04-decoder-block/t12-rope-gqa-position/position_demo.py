"""B04/T12：将同一组 Q/K/V 整段计算与分块计算进行位置语义对照。

这是注意力层的局部实验，不是完整 Decoder（解码器）或生成器。
缓存只是函数内的已旋转 K 和未旋转 V 张量，结束后释放；不读写外部状态。
"""

import json

import torch
from torch.nn import functional as F

from gqa import causal_mask, grouped_attention, repeat_kv
from rope import apply_rope


def compare_positions(batch: int = 2, length: int = 7, query_heads: int = 4,
                      kv_heads: int = 2, head_dim: int = 4, start: int = 0,
                      chunks: tuple[int, ...] = (3, 1, 3)) -> dict[str, float]:
    """返回整段/分块/SDPA 误差，以及故意重置位置产生的反例误差。

    参数描述 CPU float64 小实验，chunks 为正的分块长度且总和等于 length。
    其余维度为正，头数整除且 head_dim 为偶数；起点非负。
    返回只有数字的字典；局部 Generator（随机数生成器）固定种子 1212，
    不改变全局随机状态。并发调用各自拥有临时张量，不能从外部改写它们。
    """
    if min(batch, length, query_heads, kv_heads, head_dim) <= 0 or start < 0:
        raise ValueError("dimensions must be positive and start nonnegative")
    if head_dim % 2 or query_heads % kv_heads:
        raise ValueError("head_dim must be even and query_heads divisible by kv_heads")
    if not chunks or min(chunks) <= 0 or sum(chunks) != length:
        raise ValueError("positive chunks must sum to length")

    # 1. 固定相同投影后输入。局部 seed 保证重跑一致，无需模型权重。
    generator = torch.Generator().manual_seed(1212)
    q = torch.randn(batch, query_heads, length, head_dim, generator=generator, dtype=torch.float64)
    k = torch.randn(batch, kv_heads, length, head_dim, generator=generator, dtype=torch.float64)
    v = torch.randn(batch, kv_heads, length, head_dim, generator=generator, dtype=torch.float64)
    # 每个 batch 暂时使用相同位置；expand 是只读视图，不复制位置表。
    positions = torch.arange(start, start + length).unsqueeze(0).expand(batch, -1)

    # 2. 整段：旋转 Q/K，V 保持原值；用绝对位置建立方形因果 mask（掩码）。
    full_q, full_k = apply_rope(q, positions), apply_rope(k, positions)
    allow = causal_mask(positions, positions)
    full = grouped_attention(full_q, full_k, v, allow)
    # Scaled Dot Product Attention（缩放点积注意力）只作为另一个实现的对照。
    # 显式扩头避免依赖 enable_gqa 的后端限制；bool True=允许。
    sdpa = F.scaled_dot_product_attention(
        full_q, repeat_kv(full_k, query_heads), repeat_kv(v, query_heads),
        attn_mask=allow, is_causal=False, dropout_p=0.0,
    )

    # 3. 分块：只旋转当前新 K，拼接到旧的已旋转 K；旧 K 不再次旋转。
    # 空缓存仍保留 B/Hkv/d 轴，cat 在序列轴 dim=2 上增长。
    cached_k, cached_v = k[:, :, :0, :], v[:, :, :0, :]
    outputs, rotated_queries = [], []
    offset = 0
    wrong_k = k[:, :, :0, :]
    wrong_outputs = []
    for count in chunks:
        end = offset + count
        current_positions = positions[:, offset:end]
        current_q = apply_rope(q[:, :, offset:end], current_positions)
        current_k = apply_rope(k[:, :, offset:end], current_positions)
        cached_k = torch.cat((cached_k, current_k), dim=2)
        cached_v = torch.cat((cached_v, v[:, :, offset:end]), dim=2)
        rectangular = causal_mask(current_positions, positions[:, :end])
        outputs.append(grouped_attention(current_q, cached_k, cached_v, rectangular))
        rotated_queries.append(current_q)

        # 4. 故意错误：每块位置从 0 重新开始，mask 却仍按真实时间先后。
        # 这样可以隔离“RoPE 位置重置”的影响，不混入错误 mask 的影响。
        reset = torch.arange(count).unsqueeze(0).expand(batch, -1)
        wrong_q = apply_rope(q[:, :, offset:end], reset)
        wrong_k = torch.cat((wrong_k, apply_rope(k[:, :, offset:end], reset)), dim=2)
        wrong_outputs.append(grouped_attention(wrong_q, wrong_k, cached_v, rectangular))
        offset = end

    # 5. 拼回原序列顺序，报告最大绝对误差。这里只比较局部注意力输出。
    return {
        "rotated_q_max_error": (torch.cat(rotated_queries, dim=2) - full_q).abs().max().item(),
        "rotated_k_max_error": (cached_k - full_k).abs().max().item(),
        "chunk_attention_max_error": (torch.cat(outputs, dim=2) - full).abs().max().item(),
        "sdpa_max_error": (sdpa - full).abs().max().item(),
        "reset_position_counterexample": (torch.cat(wrong_outputs, dim=2) - full).abs().max().item(),
    }


def main() -> None:
    """打印默认多块对照以及位置 6/7 对历史位置 0..7 的矩形 mask。"""
    print(json.dumps(compare_positions(), indent=2))
    mask = causal_mask(torch.tensor([[6, 7]]), torch.arange(8).reshape(1, 8))
    print("query positions [6,7], key positions [0..7]:")
    print(mask[0, 0].int().tolist())


if __name__ == "__main__":
    main()
