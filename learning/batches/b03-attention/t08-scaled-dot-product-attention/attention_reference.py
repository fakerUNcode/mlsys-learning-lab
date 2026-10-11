"""T08 单头缩放点积 Attention 的可读 reference 与数值对照入口。

本文件位于 Stage-02 的 T08 章节，用于把 QKᵀ、缩放、稳定 softmax 和乘 V
逐步写开，再与 PyTorch 的 scaled_dot_product_attention 对照；它不是高性能 kernel。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionResult:
    """保存输出与显式中间矩阵，供 shape、数值和存储量检查。"""

    output: torch.Tensor
    scores: torch.Tensor
    weights: torch.Tensor


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """沿最后一轴稳定归一化 logits。

    参数 logits: 任意前导维度、最后一维非空的浮点 Tensor；每行代表一组候选分数。
    返回与输入同形状、每行非负且总和为 1 的概率 Tensor。
    并发要求：只读输入并分配新 Tensor；调用者不可在并发期间原地改写输入。

    softmax 对一整行减去该行最大值不变；这样最大指数为 exp(0)=1，避免大正数
    直接求指数溢出。这里使用 PyTorch 基础逐 Tensor 运算，没有调用封装 Attention。
    """
    # 若 logits 维度为 0 或最后一维为空则抛异常
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty last dimension")
    # 对 logits 的最后一维（每个 query 对所有 key 的打分）求最大值。keepdim=True 保持维度，例如 [B, S, S] → [B, S, 1]，方便后续广播相减。得到每个 query 位置对应的最大分数。
    row_max = logits.max(dim=-1, keepdim=True).values
    # 将整行分数减去该行最大值。
    shifted = logits - row_max
    # 对偏移后的分数逐元素求指数。
    exponentials = torch.exp(shifted)
    # 对指数结果按行求和，得到 softmax 的分母（归一化项）。形状同样保持为 [B, S, 1]，用于广播除法。
    denominator = exponentials.sum(dim=-1, keepdim=True)
    return exponentials / denominator


def attention_reference(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> AttentionResult:
    """手写单头 attention，输入输出按 [B,S,D] 排列。

    参数 query/key/value: 同 device、同 dtype、同形状的浮点 Tensor [B,S,D]；本
    任务是 self-attention reference，所以 query/key/value 的序列长度均为 S，特征
    维均为 D。实际数值可以来自同一输入经不同线性投影，但投影不属于本任务。
    返回 output [B,S,D]，并保留 scores/weights [B,S,S] 供教学检查。
    并发要求：函数只读输入并创建新结果；不可与输入的原地写入并发。

    B 是互不混合的样本数；S 是每个样本的 token/位置数；D 是每个位置的特征数。
    scores[b,i,j] 表示第 b 个样本中 query 位置 i 对 key 位置 j 的匹配分数。
    """
    # 检查 Q/K/V 是否都是三维张量
    # 这里约定输入格式为 [Batch, Sequence, Feature]，即 [B, S, D]
    # 如果不是三维，直接报错，避免后续 transpose/matmul 维度错乱
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must all have shape [B,S,D]")

    # 检查 Q/K/V 的形状是否完全一致
    # 因为是 self-attention reference，要求三者来自同一输入的不同投影
    # 形状不同会导致矩阵乘法维度不匹配
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("this self-attention reference requires equal [B,S,D] shapes")

    # 检查 B、S、D 三个维度是否都不为 0
    # 任何一个维度为 0 都会导致矩阵乘法无意义（空张量）
    # 提前拦截，给出明确错误信息
    if query.shape[0] == 0 or query.shape[1] == 0 or query.shape[2] == 0:
        raise ValueError("B, S, and D must be non-zero")

    # 检查 Q/K/V 是否都是浮点类型（float32/float64 等）
    # 整数类型做 softmax/除法会产生截断错误，结果完全错误
    # 这里用 PyTorch 内置方法统一判断
    if not (query.is_floating_point() and key.is_floating_point() and value.is_floating_point()):
        raise TypeError("query, key, and value must be floating-point tensors")

    # 检查 Q/K/V 是否在同一个设备上（CPU/GPU 一致）
    # 不同设备之间不能直接做矩阵乘法，会直接抛 RuntimeError
    # 提前检查给出更清晰的错误提示
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")

    # 检查 Q/K/V 的数据类型是否完全一致（float32 vs float64 等）
    # 混合精度输入会导致隐式类型提升或报错，影响数值对照结果
    # 统一 dtype 保证手写 reference 和 SDPA 的黑盒对照公平
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")

    # 最后一个维度是要取出来运算的特征维度
    feature_size = query.shape[-1]
    # 这行代码就是 Attention 公式里的 QKᵀ，Q[B,S,D] @ Kᵀ[B,D,S] -> [B,S,S]：每个 query 位置与所有 key 位置逐一做点积。transpose(-2, -1) 是把最后两个维度交换，对于该单样本实验相当于转置
    raw_scores = torch.matmul(query, key.transpose(-2, -1))
    # 按 key/query 特征数 D 的算术平方根缩放，防止 D 增大时点积方差同步增大。
    scores = raw_scores / math.sqrt(feature_size)
    # 对最后一轴（key 位置 j）归一化：每个 query 位置获得一行注意力权重。
    weights = stable_softmax(scores)
    # [B,S,S] @ [B,S,D] -> [B,S,D]：用权重对所有 value 向量加权求和。
    output = torch.matmul(weights, value)
    return AttentionResult(output=output, scores=scores, weights=weights)


def sdpa_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """调用 PyTorch SDPA 作黑盒对照；本函数不用于手写算法本身。"""
    # SDPA 约定 [..., heads, sequence, feature]；在 [B,S,D] 中插入 H=1 的单头轴，
    # 使形状成为 [B,1,S,D]。末尾移除该轴后恢复 [B,S,D]。
    return F.scaled_dot_product_attention(
    query.unsqueeze(1),   # [B, 1, S, D]
    key.unsqueeze(1),     # [B, 1, S, D]
    value.unsqueeze(1),   # [B, 1, S, D]
    attn_mask=None,       # 不加掩码
    dropout_p=0.0,        # 训练时不 dropout
    is_causal=False,      # 非自回归/非因果，不屏蔽未来位置
    scale=None,           # 自动用 1/√D 缩放
).squeeze(1)


def estimated_cost(batch: int, sequence: int, features: int, element_size: int) -> tuple[int, int]:
    """返回两次矩阵乘法 FLOPs 估算，以及一个 [B,S,S] 矩阵字节数。"""
    # 一个 [m,k] @ [k,n] 的稠密矩阵乘法约为 2*m*k*n FLOPs（乘和加各计一次）。
    matmul_flops = 4 * batch * sequence * sequence * features
    one_attention_matrix_bytes = batch * sequence * sequence * element_size
    return matmul_flops, one_attention_matrix_bytes


def main() -> None:
    """在 CPU float64 上跑三组尺寸，核对手写 reference、SDPA 和不变量。"""
    torch.manual_seed(20261011)
    cases = [(1, 3, 2), (2, 5, 8), (3, 17, 32)]
    print(f"torch={torch.__version__} device=cpu dtype=torch.float64 seed=20261011")
    for batch, sequence, features in cases:
        # generator 固定输入来源；float64 CPU 便于先检查公式实现，不能代表低精度 GPU 误差。
        generator = torch.Generator(device="cpu").manual_seed(20261011 + batch + sequence + features)
        query = torch.randn(batch, sequence, features, generator=generator, dtype=torch.float64)
        key = torch.randn(batch, sequence, features, generator=generator, dtype=torch.float64)
        value = torch.randn(batch, sequence, features, generator=generator, dtype=torch.float64)

        manual = attention_reference(query, key, value)
        sdpa = sdpa_reference(query, key, value)
        torch.testing.assert_close(manual.output, sdpa, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(
            manual.weights.sum(dim=-1), torch.ones(batch, sequence, dtype=torch.float64),
            rtol=0.0, atol=1e-12,
        )
        assert manual.output.shape == (batch, sequence, features)
        assert manual.scores.shape == (batch, sequence, sequence)
        assert manual.weights.shape == (batch, sequence, sequence)
        assert torch.isfinite(manual.output).all()

        max_abs_error = (manual.output - sdpa).abs().max().item()
        flops, matrix_bytes = estimated_cost(batch, sequence, features, query.element_size())
        print(
            f"shape=[{batch},{sequence},{features}] output={list(manual.output.shape)} "
            f"max_abs_error={max_abs_error:.3e} "
            f"matmul_flops~{flops:,} one_[B,S,S]_matrix={matrix_bytes:,}B"
        )

    # 极大 logits 用于验证“减最大值”确实避免 exp(1000) 溢出；概率应有限且归一化。
    extreme_logits = torch.tensor([[1000.0, 999.0, -1000.0]], dtype=torch.float64)
    extreme_probabilities = stable_softmax(extreme_logits)
    assert torch.isfinite(extreme_probabilities).all()
    torch.testing.assert_close(
        extreme_probabilities.sum(dim=-1), torch.ones(1, dtype=torch.float64),
        rtol=0.0, atol=1e-15,
    )
    print(f"extreme_softmax_finite={bool(torch.isfinite(extreme_probabilities).all())}")
    print("T08 checks passed")


if __name__ == "__main__":
    main()
