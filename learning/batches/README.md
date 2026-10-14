# 按 B 批次学习

本目录按 [任务清单](../../plan/TASKS.md) 的 B 批次组织章节。任务清单的 Stage 是路线大阶段，B 是周级学习批次，T 是可独立阅读和验收的任务。仓库原有的 `learning/stage-02` 是历史路径，不能据此判断当前路线处于 Stage 2；旧入口保留在[兼容说明](../stage-02/README.md)。

## 阶段 1

TASKS 的阶段 1 是模型基础，共 B01～B06。当前进度在 B04；T08～T10 有 CPU 程序证据，T10 另有 CUDA GPU 数值检查；T11 四个基础模块与 T12 旋转、头映射及位置实验的 CPU 检查已完成。

| 批次 | 主题 | 任务 | 章节入口 | 状态 |
| --- | --- | --- | --- | --- |
| B01 | 环境、Tensor 与布局 | T01～T04 | [B01 目录](b01-environment-tensor-layout/README.md) | 现有材料已迁入 |
| B02 | 梯度与训练闭环 | T05～T07 | [B02 目录](b02-gradients-training-loop/README.md) | 现有材料已迁入 |
| B03 | Attention | T08～T10 | [B03 目录](b03-attention/README.md) | T08～T10 均有 CPU 证据；T10 另有 GPU float64 检查，未测低精度/性能 |
| B04 | Decoder Block | T11～T13 | [B04 目录](b04-decoder-block/README.md) | T11～T12 完成；T13 待创建 |
| B05 | 生成、KV 与资源估算 | T14～T16 | TASKS 中定义 | 待创建材料 |
| B06 | 真实 target-only baseline | T17～T20 | TASKS 中定义 | 待创建材料 |

后续阶段按同样方式逐批增加 B 目录；没有实际章节时不预先创建空任务目录。每个 B 目录保留任务清单对应的 T 子目录，便于独立运行、复用与验收。

## 后续阶段

- TASKS 阶段 2：B07～B10，GPU 与概率基础。
- TASKS 阶段 3：B11～B14，固定 K 真实推测解码。
- TASKS 阶段 4 及以后：继续按 TASKS 的 B 批次扩展。

阶段计划与目录入口分开维护：任务顺序、依赖和验收要求以 TASKS 为准；已落地材料以对应 B 目录为准。
