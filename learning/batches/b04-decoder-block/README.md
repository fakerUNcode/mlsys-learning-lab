# B04 Decoder Block

B04 属于任务清单阶段 1，第 4 周，依赖 B03 Attention（注意力）。本批次逐步建立 Decoder Block（解码器块）的组成模块。

| 任务 | 主题 | 章节与证据 |
| --- | --- | --- |
| T11 | Embedding、残差、RMSNorm、SwiGLU | [四个独立模块与详细笔记](t11-embedding-residual-rmsnorm-swiglu/README.md)、[完成记录](../../../plan/completed/T11.md)、[验证报告](../../../reports/stage-02/2026-10-13-t11-validation.md) |
| T12 | RoPE、GQA 与位置语义 | 计划中 |
| T13 | 简化 Decoder 前向与第一月复盘 | 计划中 |

T11 的四个模块均可单独运行，CPU float64 reference 检查 6 项通过。未覆盖低精度、GPU、反向传播、性能或完整 Decoder Block；任务边界见[完成记录](../../../plan/completed/T11.md)。
