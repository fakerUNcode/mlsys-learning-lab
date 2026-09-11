#!/usr/bin/env python3
"""测量 PyTorch 向量加法的延迟与有效读写带宽。

这个脚本是项目的 benchmark 入门样例，重点不是实现自定义 CUDA kernel，
而是建立可信测量习惯：

1. 先验证结果正确；
2. 正式采样前 warmup；
3. CUDA 使用 Event，避免只测到异步提交时间；
4. 重复采样并报告分布；
5. 记录 device、dtype、输入规模和迭代次数。

示例：
    python benchmarks/benchmark_vector_add.py \
        --device cpu --n 1048576 --warmup 5 --iters 20

    python benchmarks/benchmark_vector_add.py \
        --device cuda --n 16777216 --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse  # 解析命令行参数并自动生成 --help
import statistics  # 计算算术平均数和样本标准差
import time  # perf_counter 提供适合 CPU 短耗时测量的高精度时钟

import torch  # Tensor 创建、向量加法、CUDA Event 和正确性检查


def parse_args() -> argparse.Namespace:
    """定义并解析 benchmark 参数。

    返回的 Namespace 包含 device、n、warmup、iters 和 dtype。
    参数验证中与“取值集合”有关的部分交给 argparse；
    正数/非负数约束在 main() 中检查，以提供统一错误消息。
    """

    parser = argparse.ArgumentParser(description=__doc__)

    # auto 会在 CUDA 可用时选择 CUDA，否则退回 CPU。
    # 显式传 cuda 时若不可用，main() 会退出，不静默切换设备。
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="execution device; auto prefers CUDA when available",
    )

    # 1 << 24 等于 16,777,216。使用位移是常见的二次幂规模写法。
    parser.add_argument(
        "--n",
        type=int,
        default=1 << 24,
        help="number of vector elements",
    )

    # warmup 不进入统计，用于触发初始化、缓存建立等一次性工作。
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="number of unmeasured warmup iterations",
    )

    # iters 是正式记录的样本数量，必须大于 0。
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="number of measured iterations",
    )

    # 当前只开放两种 dtype，避免结果表出现未经验证的类型。
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="tensor element type",
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    """使用线性插值计算分位数。

    values:
        毫秒样本列表。调用者保证列表非空。
    q:
        [0, 1] 范围内的分位位置，例如 0.95 表示 p95。

    算法先排序，再把分位位置映射到相邻两个样本之间。
    当位置不是整数时，按 fraction 做线性插值。
    """

    ordered = sorted(values)

    # 单个样本没有“相邻点”可插值，它自己就是所有分位数。
    if len(ordered) == 1:
        return ordered[0]

    # position 位于 [0, len-1] 的样本索引轴上。
    position = (len(ordered) - 1) * q
    lower = int(position)  # int 对非负值向下截断，得到左侧索引
    upper = min(lower + 1, len(ordered) - 1)  # 防止超过末尾
    fraction = position - lower  # 距离左侧样本的比例

    # 左值 + 相邻差值 × 比例。
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def summarize(samples_ms: list[float]) -> dict[str, float]:
    """把原始毫秒样本汇总成常见延迟统计量。"""

    return {
        "min_ms": min(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "p99_ms": percentile(samples_ms, 0.99),
        # statistics.stdev 要求至少两个样本；单样本波动定义为 0。
        "std_ms": (
            statistics.stdev(samples_ms)
            if len(samples_ms) > 1
            else 0.0
        ),
        "max_ms": max(samples_ms),
    }


def main() -> None:
    """创建输入、验证正确性、预热、计时并输出报告。"""

    args = parse_args()

    # n 和 iters 为 0 时无法形成有效实验；
    # warmup 可以为 0，便于快速调试，但不能为负数。
    if args.n <= 0 or args.warmup < 0 or args.iters <= 0:
        raise SystemExit(
            "--n and --iters must be positive; "
            "--warmup cannot be negative"
        )

    # auto 只在这里解析一次，后续所有 Tensor 都使用同一个 device。
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    # 用户明确请求 CUDA 时，不允许悄悄退回 CPU；
    # 否则报告中的 device 与用户意图不一致。
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but "
            "torch.cuda.is_available() is false"
        )

    # 把命令行字符串映射为真正的 torch.dtype。
    dtype = (
        torch.float32
        if args.dtype == "float32"
        else torch.float16
    )

    # a 全为 1，b 全为 2，所以参考结果应全为 3。
    # 三个 Tensor 从一开始就在目标 device，计时不包含 H2D/D2H。
    a = torch.ones(args.n, device=device, dtype=dtype)
    b = torch.full_like(a, 2)
    out = torch.empty_like(a)

    def run_once() -> None:
        """执行一次被测操作，不重新分配输出 Tensor。"""

        # out=out 避免每轮为结果重新申请内存。
        # CUDA 路径通常异步提交到当前 stream。
        torch.add(a, b, out=out)

    # 计时前先执行并检查正确性。
    # 这一轮不计入样本，同时也会触发部分惰性初始化。
    run_once()
    expected = torch.full_like(out, 3)
    torch.testing.assert_close(out, expected)

    # warmup 的结果不记录。它让后续采样更接近稳态。
    for _ in range(args.warmup):
        run_once()

    # CUDA 是异步的：warmup 循环结束只代表 Host 已提交任务。
    # 同步后才能保证正式计时不会夹带尚未完成的 warmup。
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples_ms: list[float] = []

    if device.type == "cuda":
        # CUDA Event 在 Device 时间线上记录位置，适合测量 kernel 区间。
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()  # 在当前 CUDA stream 记录起点
            run_once()  # 异步提交向量加法
            end.record()  # 在同一 stream 记录终点

            # 等待终点完成；否则 Host 可能在 GPU 完成前读取时间。
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))
    else:
        # CPU 操作同步完成，可直接用单调高精度时钟包围调用。
        for _ in range(args.iters):
            start = time.perf_counter()
            run_once()
            elapsed_seconds = time.perf_counter() - start
            samples_ms.append(elapsed_seconds * 1000)

    result = summarize(samples_ms)

    # 每个元素理论上读取 a、读取 b、写出 out，共搬运 3 个元素。
    # element_size() 返回当前 dtype 的单元素字节数。
    # 这是算法层面的有效带宽估算，不等于实际 DRAM transaction，
    # 也不能直接当作硬件理论峰值。
    bytes_per_iteration = (
        args.n
        * 3
        * torch.tensor([], dtype=dtype).element_size()
    )

    # p50_ms 先除以 1000 转为秒，再除以 1e9 转为 GB/s。
    bandwidth_gb_s = (
        bytes_per_iteration
        / (result["p50_ms"] / 1000)
        / 1e9
    )

    # 输出同时包含实验参数、正确性和统计量，方便保存为文本证据。
    print("=== vector add benchmark ===")
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"elements: {args.n}")
    print(f"warmup: {args.warmup}")
    print(f"iterations: {args.iters}")
    print("correctness: PASS")
    for name, value in result.items():
        print(f"{name}: {value:.4f}")
    print(
        "effective_read_write_bandwidth_GB_s: "
        f"{bandwidth_gb_s:.2f}"
    )


# 只有直接运行本文件时才执行 main()。
# 若其他测试 import 本模块，可复用函数而不会自动启动 benchmark。
if __name__ == "__main__":
    main()
