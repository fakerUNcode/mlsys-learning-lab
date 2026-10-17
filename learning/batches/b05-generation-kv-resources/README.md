# B05 生成与资源

B05 属于任务清单阶段 1，第 5～6 周，依赖 B04 Decoder Block（解码器块）。按顺序覆盖小模型训练与逐 token 生成、KV Cache reference、权重/KV/激活内存预算。

| 任务 | 主题 | 章节与证据 |
| --- | --- | --- |
| T14 | 小语言模型训练与逐 token 生成 | [训练与生成详解](t14-tiny-lm-training-generation/README.md)、[完成记录](../../../plan/completed/T14.md)、[验证报告](../../../reports/stage-02/2026-10-15-t14-validation.md) |
| T15 | KV Cache reference | [prefill/decode 对照与详细笔记](t15-kv-cache-reference/README.md)、[完成记录](../../../plan/completed/T15.md)、[验证报告与逐步数据](../../../reports/stage-02/2026-10-17-t15-validation.md) |
| T16 | 权重/KV/激活预算 | [估算表、CUDA 实测与双模型容量例](t16-memory-budget/README.md)、[完成记录](../../../plan/completed/T16.md)、[验证报告与原始数据](../../../reports/stage-02/2026-10-17-t16-validation.md) |

T14 复用 T13 小模型，在 CPU float32 上拟合两条手工 token 序列并记录 prefix/logits/token 轨迹；没有使用真实语料或 tokenizer。T15 复用同一小模型，在 CPU float64 比较完整前缀重算与缓存 prefill/单步 decode；不覆盖真实模型和性能测量。T16 用 toy 双模型 CUDA 审计核对权重/KV/前向峰值计量，并另列假设大模型预算；没有加载大模型权重。

2026-10-16：T14 笔记按数据、模型、概率损失、梯度更新、训练与生成重写，增加基础知识与真实语句的对应表；训练与检查脚本逐行补注释。复核与范围见[修订记录](../../../reports/stage-02/2026-10-16-t14-doc-rewrite.md)。

2026-10-17：T15 完成单层 KV Cache reference；3 种 prompt 长度各连续 decode 16 步，保存 51 条 logits 对照与 cache 长度记录。细节见[T15 验证报告](../../../reports/stage-02/2026-10-17-t15-validation.md)。

2026-10-17：T16 完成权重、双模型 KV、activation、allocator 与余量账本；toy target/draft 的 KV 公式与实际 GPU Tensor 字节完全一致，并记录小模型实测差额及假设双模型容量缺口。见[T16 验证报告](../../../reports/stage-02/2026-10-17-t16-validation.md)。
