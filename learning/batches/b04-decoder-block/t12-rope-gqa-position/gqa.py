"""
B04/T12：Grouped Query Attention（分组查询注意力）头映射与显式 reference 实现

背景说明：
- 在标准的 Multi-Head Attention（MHA）中：
    Q、K、V 都使用相同数量的注意力头（Hq = Hkv）
- 在 Multi-Query Attention（MQA）中：
    K、V 只有一个头，被所有 Query 头共享
- 在 Grouped Query Attention（GQA）中：
    K/V 头数 Hkv 少于 Query 头数 Hq，但多于 1
    => 每个 KV 头被一组（group）Query 头共享

本模块作用：
1. 将较少的 KV 头（Hkv）扩展为与 Query 头数（Hq）匹配
2. 显式构造因果掩码（causal mask）
3. 使用缩放点积注意力（scaled dot-product attention）计算输出
4. 不保存缓存、不持有可学习参数，适合作为参考实现

输入约定：
- Q:  [B, Hq, T, d]
- K/V:[B, Hkv, N, d]
其中：
- B: batch size
- Hq: query 头数
- Hkv: key/value 头数（Hkv 整除 Hq）
- T: query 序列长度
- N: key/value 序列长度
- d: 每个头的特征维度

重要性质：
- 本实现是“函数式”的：
    * 不修改输入张量
    * 不使用缓存
    * 不持有模型参数
- 因此适合并发调用，只要调用方不修改共享输入
"""

import torch

from rope import validate_positions


def repeat_kv(x: torch.Tensor, query_heads: int) -> torch.Tensor:
    """
    将 K/V 张量从 [B, Hkv, N, d] 扩展为 [B, Hq, N, d]。

    核心思想：
    - 每个 KV 头被连续分配给 g 个 Query 头
    - g = Hq / Hkv（group size，必须为正整数）

    示例（Hkv=2, Hq=4）：
    KV 头编号: [0, 1]
    扩展后 Query 头对应的 KV 头顺序:
        [0, 0, 1, 1]

    参数
    ----
    x : torch.Tensor
        原始 KV 张量，形状为 [B, Hkv, N, d]
    query_heads : int
        目标 Query 头数 Hq，必须是 Hkv 的正整数倍

    返回
    ----
    torch.Tensor
        形状为 [B, Hq, N, d] 的扩展 KV 张量
    """

    # ---- 输入校验 ----
    if x.ndim != 4 or min(x.shape) == 0:
        # 确保 x 是 4 维且每个维度大小均大于 0
        raise ValueError("x must be nonempty [B,Hkv,N,d]")

    if (
        not isinstance(query_heads, int)
        or query_heads <= 0
        or query_heads % x.shape[1]
    ):
        # Hq 必须是正整数，并且能被 Hkv 整除
        raise ValueError(
            "query_heads must be a positive multiple of KV heads"
        )

    # 解包 KV 张量的各个维度
    batch, kv_heads, length, d = x.shape

    # 每个 KV 头对应的 Query 头数量
    groups = query_heads // kv_heads

    # 特殊情况：如果 group size 为 1
    # 说明 Hq == Hkv，不需要复制或扩展
    if groups == 1:
        return x

    # ---- KV 头扩展 ----
    # 在 KV 头维度后插入一个新维度，表示组内重复
    # [B, Hkv, N, d] -> [B, Hkv, 1, N, d]
    expanded = x[:, :, None, :, :]

    # expand 只改变“逻辑形状”，不分配新内存
    # 新插入的 group 维度 stride 为 0，表示多个头共享同一份数据
    # [B, Hkv, 1, N, d] -> [B, Hkv, groups, N, d]
    expanded = expanded.expand(batch, kv_heads, groups, length, d)

    # reshape 将两个头维度合并：
    # [B, Hkv, groups, N, d] -> [B, Hq, N, d]
    # 合并顺序是：
    #   (kv0, kv0, ..., kv1, kv1, ...)
    # 这正是 GQA 所需的“连续分组”语义
    #
    # 注意：
    # - 这里可能触发实际的数据复制
    # - 因此不能声称该操作是“完全零拷贝”
    return expanded.reshape(batch, query_heads, length, d)


def causal_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor
) -> torch.Tensor:
    """
    构造因果注意力掩码（causal mask）。

    掩码规则：
        key 的绝对位置 <= query 的绝对位置 时允许 attention
        否则禁止 attention（未来信息不可见）

    返回张量：
        bool 类型，形状为 [B, 1, T, N]
        其中：
        True  -> 允许 attention
        False -> 屏蔽 attention

    参数
    ----
    query_positions : torch.Tensor
        每个 query token 的绝对位置，形状 [B, T]
    key_positions : torch.Tensor
        每个 key token 的绝对位置，形状 [B, N]

    返回
    ----
    torch.Tensor
        bool 掩码，形状 [B, 1, T, N]
    """

    # 校验位置张量的合法性（非负、严格递增等）
    validate_positions(query_positions)
    validate_positions(key_positions)

    # batch 维度和设备必须一致
    if (
        query_positions.shape[0] != key_positions.shape[0]
        or query_positions.device != key_positions.device
    ):
        raise ValueError(
            "query/key positions must match batch and device"
        )

    # 利用广播机制构造比较矩阵：
    #
    # query_positions[:, :, None] -> [B, T, 1]
    # key_positions[:, None, :]   -> [B, 1, N]
    #
    # 比较结果 shape: [B, T, N]
    # 表示：每个 query 是否可以看到对应的 key
    comparison = key_positions[:, None, :] <= query_positions[:, :, None]

    # 插入一个“头”维度，使掩码可被所有 Hq 个头共用
    # [B, T, N] -> [B, 1, T, N]
    return comparison.unsqueeze(1)


def grouped_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow: torch.Tensor
) -> torch.Tensor:
    """
    显式实现 Grouped Query Attention。

    计算流程：
        1. 将 K/V 从 Hkv 个头扩展到 Hq 个头
        2. 计算 QK^T 并缩放
        3. 使用因果掩码屏蔽非法位置
        4. 对 value 做加权求和

    参数
    ----
    q : torch.Tensor
        Query 张量，形状 [B, Hq, T, d]
    k : torch.Tensor
        Key 张量，形状 [B, Hkv, N, d]
    v : torch.Tensor
        Value 张量，形状 [B, Hkv, N, d]
    allow : torch.Tensor
        注意力掩码，形状 [B, 1, T, N]，bool 类型
        True 表示允许 attention

    返回
    ----
    torch.Tensor
        注意力输出，形状 [B, Hq, T, d]
    """

    # ---- 输入合法性检查 ----
    if any(x.ndim != 4 or min(x.shape) == 0 for x in (q, k, v)):
        raise ValueError("q/k/v must be nonempty 4D tensors")

    if q.dtype not in (torch.float32, torch.float64) or any(
        x.dtype != q.dtype for x in (k, v)
    ):
        raise TypeError("q/k/v must share float32/float64 dtype")

    if any(x.device != q.device for x in (k, v, allow)):
        raise ValueError("q/k/v/mask must share device")

    if (
        k.shape != v.shape
        or q.shape[0] != k.shape[0]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError(
            "q/k/v must match batch, head dimension and KV shape"
        )

    # 掩码必须与 query/key 长度匹配
    required = (q.shape[0], 1, q.shape[2], k.shape[2])
    if allow.dtype != torch.bool or allow.shape != required:
        raise ValueError(f"allow must be bool with shape {required}")

    # 不允许出现“完全无法看到任何 key”的 query 行
    if not allow.any(dim=-1).all():
        raise ValueError(
            "fully masked rows are outside this exercise"
        )

    # ---- 1. 扩展 K/V 头 ----
    # 原始 K/V 只有 Hkv 个头
    # 扩展为与 Query 相同的 Hq 个头（逻辑复制）
    keys = repeat_kv(k, q.shape[1])
    values = repeat_kv(v, q.shape[1])

    # ---- 2. 计算注意力分数 ----
    # Q : [B, Hq, T, d]
    # K^T: [B, Hq, d, N]
    # scores: [B, Hq, T, N]
    scores = (q @ keys.transpose(-2, -1)) / q.shape[-1] ** 0.5

    # ---- 3. 应用注意力掩码 ----
    # allow 中 False 的位置填充负无穷
    # 经过 softmax 后权重变为 0
    scores = scores.masked_fill(~allow, float("-inf"))

    # 对 key 维度做 softmax，得到注意力权重
    probabilities = scores.softmax(dim=-1)

    # ---- 4. 加权求和 Value ----
    # probabilities: [B, Hq, T, N]
    # values:        [B, Hq, N, d]
    # output:        [B, Hq, T, d]
    return probabilities @ values


def main() -> None:
    """
    演示函数：
    用 KV 头的编号作为数值，直观观察 KV 头如何映射到 Query 头。

    示例配置：
        Hkv = 2
        Hq  = 4
    期望结果：
        KV 头 [10, 20]
        映射后 Query 头看到的值：
        [10, 10, 20, 20]
    """

    # 构造 KV 张量：
    # 2 个 KV 头，数值分别为 10 和 20
    kv = torch.tensor([10.0, 20.0]).reshape(1, 2, 1, 1)

    print("KV heads:", kv.flatten().tolist())

    # 将 KV 头从 2 扩展到 4
    expanded = repeat_kv(kv, 4)

    print(
        "query head -> KV value:",
        expanded.flatten().tolist()
    )


if __name__ == "__main__":
    main()