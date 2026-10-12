# 历史路径兼容说明

learning/stage-02 是此前使用的历史路径名称，不能对应 TASKS 当前的 Stage 2。自 2026-10-11 起，阶段 1 中已有的 T01～T09 材料按任务清单的 B 批次归档。

新的主入口是[按 B 批次学习](../batches/README.md)：

- [B01：环境、Tensor 与布局](../batches/b01-environment-tensor-layout/README.md)
- [B02：梯度与训练闭环](../batches/b02-gradients-training-loop/README.md)
- [B03：Attention](../batches/b03-attention/README.md)

旧的 tensors/ 目录只保留早期兼容脚本；新材料、命令和链接以 B 批次目录为准。后续 TASKS 阶段 1 的 B04～B06 会在内容形成后进入同一 learning/batches/ 结构。
