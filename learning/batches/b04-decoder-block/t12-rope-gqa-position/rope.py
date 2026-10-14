"""B04/T12：默认 Rotary Position Embedding（旋转位置嵌入）的独立练习。

接收已拆头的 [B,H,S,d]，按 Llama 的前后半区配对旋转，返回同形张量。
这是 float32/float64 公式 reference；不实现频率缩放或模型权重加载。
所有函数无缓存、无输入原地修改；调用期间调用方不能改写输入。
"""

import math

import torch


def validate_positions(positions: torch.Tensor) -> None:
    """检查非空 Long [B,S] 非负、严格递增的位置；只读，无返回值。"""
    if positions.dtype != torch.long:
        raise TypeError("positions must use torch.long")
    if positions.ndim != 2 or min(positions.shape) == 0:
        raise ValueError("positions must have nonempty shape [B,S]")
    if (positions < 0).any() or (positions[:, 1:] <= positions[:, :-1]).any():
        raise ValueError("positions must be nonnegative and strictly increasing")


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """将 x 的每对特征旋转 positions 指定的角度。

    参数：x 为 float32/float64 [B,H,S,d]，d 必须为偶数；positions 为
    同设备 Long [B,S]；theta 是正且有限的频率底数。
    返回：同 shape/device/dtype 的新张量，x 的所有权仍属于调用方。
    并发：函数无共享状态；调用期间不允许另一线程修改输入。
    """
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("reference supports only float32/float64")
    if x.ndim != 4 or min(x.shape) == 0 or x.shape[-1] % 2:
        raise ValueError("x must be nonempty [B,H,S,d] with even d")
    validate_positions(positions)
    if positions.shape != (x.shape[0], x.shape[2]) or positions.device != x.device:
        raise ValueError("positions must match x batch, sequence and device")
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be positive and finite")

    # 1. arange(0,d,2) 产生 0,2,...,d-2；除以 d 后得到频率指数。
    # 使用输入精度便于 CPU float64 数学核对；官方实现用 float32 计算角度，
    # 因此这里验证公式语义，不承诺与官方实现逐位一致。
    d = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d)
    # positions[...,None] 增加最后一轴：[B,S] -> [B,S,1]，广播乘 [d/2]。
    angles = positions.to(x.dtype)[..., None] * inv_freq
    # 2. 两半使用同一组角度；unsqueeze(1) 为 H 轴留下大小为 1 的广播位置。
    phase = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    # 3. Llama 配对方式是 (x_i,x_{i+d/2})，不是相邻 (x_0,x_1)。
    # 切片是只读视图；cat 创建新张量 [-后半,前半]，没有改写 x。
    rotated_half = torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)
    return x * phase.cos() + rotated_half * phase.sin()


def main() -> None:
    """固定二维例子：d=2 的唯一频率为 1，位置 1 对应旋转 1 弧度。"""
    x = torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float64)
    result = apply_rope(x, torch.tensor([[1]]))
    print("input [3,4], position=1 rad:", result.flatten().tolist())
    print("squared norms:", x.square().sum().item(), result.square().sum().item())


if __name__ == "__main__":
    main()
