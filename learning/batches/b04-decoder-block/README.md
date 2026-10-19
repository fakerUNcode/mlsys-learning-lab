# B04 Decoder Block

B04 属于任务清单阶段 1，第 4 周，依赖 B03 Attention（注意力）。本批次逐步建立 Decoder Block（解码器块）的组成模块。

| 任务 | 主题 | 章节与证据 |
| --- | --- | --- |
| T11 | Embedding、残差、RMSNorm、SwiGLU | [四个独立模块与详细笔记](t11-embedding-residual-rmsnorm-swiglu/README.md)、[完成记录](../../../plan/completed/T11.md)、[验证报告](../../../reports/stage-02/2026-10-13-t11-validation.md) |
| T12 | RoPE、GQA 与位置语义 | [从问题出发的完整导读](t12-rope-gqa-position/README.md)、[完成记录](../../../plan/completed/T12.md)、[验证报告](../../../reports/stage-02/2026-10-14-t12-validation.md)、[笔记重写核对](../../../reports/stage-02/2026-10-14-t12-notes.md) |
| T13 | 简化 Decoder 前向与第一月复盘 | [小型前向、shape 图与复盘](t13-small-decoder-forward/README.md)、[完成记录](../../../plan/completed/T13.md)、[验证报告](../../../reports/stage-02/2026-10-14-t13-validation.md) |

T11 的四个模块均可单独运行，CPU float64 reference 检查 6 项通过。未覆盖低精度、GPU、反向传播、性能或完整 Decoder Block；任务边界见[完成记录](../../../plan/completed/T11.md)。

T12 的四个例子与 12 项 CPU 检查通过；五组输入整段 / 增量位置及注意力输出一致，核对固定 TinyLlama 配置。未加载真实模型或测试正式缓存、GPU、低精度与性能。

T13 的单层 CPU float64 小模型覆盖 `[B,S]` token IDs 至 `[B,S,V]` logits，5 项检查通过，涵盖关键 shape 与因果隔离。推测解码仅解释 target/draft/验证/纠正职责，没有算法或性能实现。第一月复盘和 B05 会话预算见 [T13 章节](t13-small-decoder-forward/README.md)。T14 已在 [B05](../b05-generation-kv-resources/README.md) 完成训练与逐 token 生成；下一任务为 T15 KV Cache reference。
