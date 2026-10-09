# PyTorch 与模型基础

状态：B01/T01～T04 的材料和对应验证已完成；B02/T05 的手算与数值对照已完成，讲解题仍可用于读者自测；训练、Attention 与模型前向尚待补充。此目录延续历史编号，阶段顺序见[任务清单](../../plan/TASKS.md)。

## 前四周

| 周次 | 验证任务 | 需要提交的证据 |
| --- | --- | --- |
| 1 | Tensor、广播、索引、reshape/transpose、stride、device/dtype | 自己预测维度；连续与非连续输入对照；环境报告 |
| 2 | 链式法则、有限差分、autograd、梯度累积、MLP 训练 | 手算计算图；数值对照；可独立解释的训练脚本 |
| 3 | Q/K/V、因果 mask、多头 Attention | 标注维度的 reference；与参考实现对照；未来 token 不影响过去位置的测试 |
| 4 | Embedding、残差、RMSNorm、SwiGLU、Decoder Block | 简化前向、结构图、边界与数值测试 |

每个实验建立独立目录，包含代码、验证和 README。已有 Tensor 实验；后续任务清单不计为完成成果。

已完成的 T01～T03 材料见 [Tensor 学习目录](tensors/README.md)、[T02 作答与批阅](tensors/t02/README.md)和[T03 实验记录](tensors/t03/README.md)。T04 见 [Buffer 所有权练习与验收](cpp-ownership/t04/README.md)。T05 材料与脚本见[链式法则与有限差分](t05/README.md)，数值证据见[验证报告](../../reports/stage-02/2026-10-08-t05-validation.md)。

## 第五至八周

实现小型语言模型训练与生成，验证有/无 KV Cache 的下一 token logits，估算权重/激活/KV 容量。本目录先保留可单独运行的简化验证代码；公开模型基线尚未覆盖。

## 通过标准

能独立预测 Tensor shape/stride、写训练循环、解释梯度与前向数据流；Attention 与缓存有数值证据。训练出高质量模型不是要求。验收后进入 CUDA，不无限扩展训练理论。

每项实验记录失败案例，并脱离现成答案复写一个小实现或定位一次错误。更新 [学习进度](../../daily-record/README.md) 时区分确认掌握、代码验证与待验证事项。
