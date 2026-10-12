"""T10 的 CUDA 设备数值实验入口。

本文件位于 B03/T10，使用 PyTorch 张量运算在当前 CUDA GPU 上复现 softmax 溢出、
减最大值稳定化和全 mask 行约定，并复核相同的整数资源账。这里是手写的
PyTorch GPU reference：张量运算由 PyTorch CUDA 后端调度，不是自定义 CUDA/Triton kernel，
也不做性能计时或分配资源表中的大尺寸矩阵。
"""

from __future__ import annotations

import torch

from attention_cost import estimate_attention_resources, format_bytes


def stable_softmax_cuda(logits: torch.Tensor) -> torch.Tensor:
    """在 CUDA Tensor 上执行减最大值 softmax。

    参数 logits：CUDA 上、形状 [..., key]、有限值的浮点 Tensor。
    返回：同设备同形状的概率；有效行总和为 1。
    并发要求：仅读输入并分配新 Tensor；调用期间不可原地改写 logits。
    每个 torch 运算都在 logits.device 执行，最大指数输入被限制为 0，避免 exp 溢出。
    """
    if logits.device.type != "cuda":
        raise ValueError("logits must be on a CUDA device")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty last dimension")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")

    # PyTorch CUDA max reduction 返回 [...,1]，keepdim 保留 key 轴以支持逐行广播。
    row_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - row_max
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


def masked_softmax_cuda(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """在 CUDA 上归一化允许位置，并把全 mask 行定义为全零。

    参数 logits：CUDA 上形状 [..., query, key] 的有限浮点分数。
    参数 allowed：与 logits 同设备、可广播到其形状的 bool Tensor；True 表示允许。
    返回：同设备同形状的概率；无有效 key 的行概率为全零。
    并发要求：仅读输入并分配新 Tensor；调用期间不可原地改写输入。
    对全 mask 行用安全最大值和分母，避免 -inf-(-inf) 和除以零。
    """
    if logits.device.type != "cuda" or allowed.device.type != "cuda":
        raise ValueError("logits and allowed must be on CUDA devices")
    if logits.device != allowed.device:
        raise ValueError("logits and allowed must use the same CUDA device")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty last dimension")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if allowed.dtype != torch.bool:
        raise TypeError("allowed must be a bool tensor")
    try:
        allowed = torch.broadcast_to(allowed, logits.shape)
    except RuntimeError as error:
        raise ValueError("allowed must be broadcastable to logits") from error

    # masked_fill 把无效 key 的分数改为 -inf；全 mask 行因此需要单独选安全最大值。
    masked_logits = logits.masked_fill(~allowed, -torch.inf)
    has_key = allowed.any(dim=-1, keepdim=True)
    row_max = masked_logits.max(dim=-1, keepdim=True).values
    safe_row_max = torch.where(has_key, row_max, torch.zeros_like(row_max))
    exponentials = torch.where(
        allowed,
        torch.exp(masked_logits - safe_row_max),
        torch.zeros_like(logits),
    )
    denominator = exponentials.sum(dim=-1, keepdim=True)
    safe_denominator = torch.where(has_key, denominator, torch.ones_like(denominator))
    return exponentials / safe_denominator


def run_gpu_experiment() -> None:
    """在当前 GPU 运行小输入验收，并打印静态资源账。

    只在 CUDA 可用时调用。固定显式 logits，无随机数；实际 GPU 张量为 [1,3] 与 [2,2]。
    资源表由 CPU 整数公式算出，不会创建 [B,H,S,S] 的大张量。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this GPU experiment requires a CUDA device")

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    print(f"torch={torch.__version__} device={properties.name} cuda={torch.version.cuda}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(
        f"memory_before_free={format_bytes(free_bytes)} "
        f"total={format_bytes(total_bytes)} "
        f"allocated={format_bytes(torch.cuda.memory_allocated(device))} "
        f"reserved={format_bytes(torch.cuda.memory_reserved(device))}"
    )

    # device=device 令输入直接驻留 GPU；CPU 代码中相同的逐 Tensor 公式在 CUDA 后端执行。
    logits = torch.tensor([[1000.0, 999.0, 998.0]], dtype=torch.float64, device=device)
    naive_exp = torch.exp(logits)
    naive = naive_exp / naive_exp.sum(dim=-1, keepdim=True)
    stable = stable_softmax_cuda(logits)
    assert not bool(torch.isfinite(naive).all().item())
    assert bool(torch.isfinite(stable).all().item())
    torch.testing.assert_close(
        stable.sum(dim=-1), torch.ones(1, dtype=torch.float64, device=device),
        rtol=0.0, atol=1e-15,
    )
    print(f"naive={naive.cpu().tolist()} finite=False")
    print(f"stable={stable.cpu().tolist()} finite=True")

    # 第二行全 mask；显式输出全零概率，避免把空集合错误归一化为 NaN。
    masked_logits = torch.tensor(
        [[2.0, 1.0], [3.0, 4.0]], dtype=torch.float64, device=device
    )
    allowed = torch.tensor(
        [[True, False], [False, False]], dtype=torch.bool, device=device
    )
    probabilities = masked_softmax_cuda(masked_logits, allowed)
    expected = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64, device=device
    )
    torch.testing.assert_close(probabilities, expected, rtol=0.0, atol=0.0)
    print(f"masked_probabilities={probabilities.cpu().tolist()}")
    print(f"masked_row_sums={probabilities.sum(dim=-1).cpu().tolist()}")

    # 显式同步让上面的 GPU 检查完成；此程序不把同步区间当作性能样本。
    torch.cuda.synchronize(device)
    print(
        f"memory_after_allocated={format_bytes(torch.cuda.memory_allocated(device))} "
        f"reserved={format_bytes(torch.cuda.memory_reserved(device))}"
    )

    # GPU 演示只分配小 Tensor；S=4096/8192 仍由 CPU 整数函数核算。
    print("\nB=1 H=16 head_dim=64 dtype=fp16; estimated only, no large allocation")
    print("S | score/probability(each) | explicit bool mask | Q,K,V,O | K+V per layer | 2 matmuls FLOPs")
    for sequence in (4096, 8192):
        ledger = estimate_attention_resources(1, 16, sequence, 64, 2)
        print(
            f"{sequence} | {format_bytes(ledger.score_bytes)} | "
            f"{format_bytes(ledger.explicit_bool_mask_bytes)} | "
            f"{format_bytes(ledger.qkvo_bytes)} | "
            f"{format_bytes(ledger.kv_cache_bytes_per_layer)} | "
            f"{ledger.attention_matmul_flops:,}"
        )
    print("GPU T10 checks passed (correctness only; no performance benchmark)")


if __name__ == "__main__":
    run_gpu_experiment()
