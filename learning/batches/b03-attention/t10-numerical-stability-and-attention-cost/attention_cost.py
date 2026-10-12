"""T10 数值稳定性与 Attention 成本实验。

本文件位于 B03 Attention 学习批次：用可读 PyTorch 运算复现 softmax 溢出、
稳定归一化、全 mask 行处理，并按 shape/dtype 计算注意力矩阵资源账。
它不是 FlashAttention kernel，也不据此测量 GPU 带宽或运行性能。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def naive_softmax(logits: torch.Tensor) -> torch.Tensor:
    """直接按行计算 softmax，故意保留大正数溢出风险。

    参数 logits：至少一维的浮点 Tensor，最后一轴是每行候选分数。
    返回：与 logits 同形状的概率；发生溢出时可能含 NaN。
    并发要求：仅读输入并创建结果；调用期间不能并发原地改写 logits。
    """
    exponentials = torch.exp(logits)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """沿最后一轴做减最大值的稳定 softmax。

    参数/返回和 naive_softmax 相同；要求每个 logits 都是有限值；空行或非浮点输入会被明确拒绝。
    每行减去其最大值不改变 softmax 概率，但会让最大指数的输入变成 0。
    """
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty last dimension")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


def masked_softmax(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """对允许位置归一化，并将全 mask 行明确定义为全零。

    参数 logits：形状 [..., query, key] 的浮点分数。
    参数 allowed：与 logits 同设备、可广播到 logits 的 bool Tensor；True 表示该 key 可参与。
    返回：与 logits 同形状的概率；全 mask 行概率和为 0，避免 NaN。
    并发要求：只读输入、分配新结果；调用期间不可原地改写输入。

    先把屏蔽位置设为负无穷；若直接对全负无穷行减最大值，会出现
    -inf - (-inf)=NaN。因此检测有效行，并为无有效 key 的行用 0 作安全最大值。
    """
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty last dimension")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if allowed.dtype != torch.bool:
        raise TypeError("allowed must be a bool tensor")
    if allowed.device != logits.device:
        raise ValueError("allowed and logits must be on the same device")
    try:
        allowed = torch.broadcast_to(allowed, logits.shape)
    except RuntimeError as error:
        raise ValueError("allowed must be broadcastable to logits") from error

    masked_logits = logits.masked_fill(~allowed, -torch.inf)
    has_key = allowed.any(dim=-1, keepdim=True)
    row_max = masked_logits.max(dim=-1, keepdim=True).values
    safe_row_max = torch.where(has_key, row_max, torch.zeros_like(row_max))
    # 不允许的位置指数固定为 0；全 mask 行分母为 0，下面用 where 输出全零。
    exponentials = torch.where(
        allowed,
        torch.exp(masked_logits - safe_row_max),
        torch.zeros_like(logits),
    )
    denominator = exponentials.sum(dim=-1, keepdim=True)
    safe_denominator = torch.where(has_key, denominator, torch.ones_like(denominator))
    return exponentials / safe_denominator


# dataclass 自动生成初始化等方法；frozen=True 禁止创建后重新赋值，便于把结果当只读账本传递。
@dataclass(frozen=True)
class AttentionResourceLedger:
    """注意力资源估算；所有字节数均为十进制字节的精确整数。"""

    score_bytes: int
    probability_bytes: int
    explicit_bool_mask_bytes: int
    qkvo_bytes: int
    kv_cache_bytes_per_layer: int
    attention_matmul_flops: int


def estimate_attention_resources(
    batch: int,
    heads: int,
    sequence: int,
    head_dim: int,
    element_bytes: int,
    *,
    mask_bytes: int = 1,
) -> AttentionResourceLedger:
    """估算一个 self-attention 层的矩阵和主要 matmul 资源。

    输入整数分别为 batch、头数、序列长度、每头维度、数据元素字节数、
    以及显式布尔 mask 每元素字节数。返回不含投影层和 allocator 暂存区的账本。
    该纯函数无共享状态，可并发调用。
    """
    dimensions = (batch, heads, sequence, head_dim, element_bytes, mask_bytes)
    if any(value <= 0 for value in dimensions):
        raise ValueError("all dimensions and element sizes must be positive")
    score_elements = batch * heads * sequence * sequence
    qkvo_elements = 4 * batch * heads * sequence * head_dim
    # QK^T 与 P@V 两次矩阵乘法，每个乘加按 2 FLOPs 计数。
    attention_matmul_flops = 4 * batch * heads * sequence * sequence * head_dim
    return AttentionResourceLedger(
        score_bytes=score_elements * element_bytes,
        probability_bytes=score_elements * element_bytes,
        explicit_bool_mask_bytes=score_elements * mask_bytes,
        qkvo_bytes=qkvo_elements * element_bytes,
        kv_cache_bytes_per_layer=2 * batch * heads * sequence * head_dim * element_bytes,
        attention_matmul_flops=attention_matmul_flops,
    )


def format_bytes(byte_count: int) -> str:
    """把字节数格式化为二进制 MiB/GiB，保留一位小数。"""
    mib = 1024**2
    gib = 1024**3
    if byte_count >= gib:
        return f"{byte_count / gib:.1f} GiB"
    return f"{byte_count / mib:.1f} MiB"


def main() -> None:
    """运行固定 CPU float64 的反例与资源表；不启动 GPU kernel。"""
    print(f"torch={torch.__version__} device=cpu dtype=float64")

    # 直接 exp(1000) 超过 float64 可表示范围；减最大值后指数最大为 exp(0)=1。
    logits = torch.tensor([[1000.0, 999.0, 998.0]], dtype=torch.float64)
    naive = naive_softmax(logits)
    stable = stable_softmax(logits)
    print(f"overflow_input={logits.tolist()}")
    print(f"naive={naive.tolist()} finite={bool(torch.isfinite(naive).all())}")
    print(f"stable={stable.tolist()} finite={bool(torch.isfinite(stable).all())}")

    # 第二行没有任何可用 key。这里的课程约定是返回全零概率与全零加权输出。
    masked_logits = torch.tensor([[2.0, 1.0], [3.0, 4.0]], dtype=torch.float64)
    allowed = torch.tensor([[True, False], [False, False]])
    probabilities = masked_softmax(masked_logits, allowed)
    print(f"masked_probabilities={probabilities.tolist()}")
    print(f"masked_row_sums={probabilities.sum(dim=-1).tolist()}")

    # B=1、H=16、d=64、fp16；序列翻倍只改变 S，便于直接核对线性与平方项。
    print("\nB=1 H=16 head_dim=64 dtype=fp16; one attention layer")
    print("S | scores(each) | probabilities(each) | explicit bool mask | Q,K,V,O | K+V cache | 2 matmuls")
    ledgers = {}
    for sequence in (4096, 8192):
        ledger = estimate_attention_resources(1, 16, sequence, 64, 2)
        ledgers[sequence] = ledger
        print(
            f"{sequence} | {format_bytes(ledger.score_bytes)} | "
            f"{format_bytes(ledger.probability_bytes)} | "
            f"{format_bytes(ledger.explicit_bool_mask_bytes)} | "
            f"{format_bytes(ledger.qkvo_bytes)} | "
            f"{format_bytes(ledger.kv_cache_bytes_per_layer)} | "
            f"{ledger.attention_matmul_flops:,}"
        )
    short, long = ledgers[4096], ledgers[8192]
    print(
        "doubling_ratios="
        f"score:{long.score_bytes // short.score_bytes}x "
        f"qkvo:{long.qkvo_bytes // short.qkvo_bytes}x "
        f"attention_flops:{long.attention_matmul_flops // short.attention_matmul_flops}x"
    )


if __name__ == "__main__":
    main()
