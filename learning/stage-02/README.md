# PyTorch 与模型基础

状态：任务清单已建立，学习和代码验收尚未完成。此目录延续历史编号，对应新版年度计划的第 1～8 周。

## 前四周

| 周次 | 验证任务 | 需要提交的证据 |
| --- | --- | --- |
| 1 | Tensor、广播、索引、reshape/transpose、stride、device/dtype | 自己预测维度；连续与非连续输入对照；环境报告 |
| 2 | 链式法则、有限差分、autograd、梯度累积、MLP 训练 | 手算计算图；数值对照；可独立解释的训练脚本 |
| 3 | Q/K/V、因果 mask、多头 Attention | 标注维度的 reference；与参考实现对照；未来 token 不影响过去位置的测试 |
| 4 | Embedding、残差、RMSNorm、SwiGLU、Decoder Block | 简化前向、结构图、边界与数值测试 |

每个实验建立独立目录，包含代码、测试和 README。当前没有这些实现，不把任务清单计为成果。

本周已完成的 T01/T02 与正在进行的 T03 材料见 [Tensor 学习目录](tensors/README.md)、[T02 作答与批阅](tensors/t02/README.md)和[T03 实验记录](tensors/t03/README.md)。

## 第五至八周

实现小型语言模型训练与生成，验证有/无 KV Cache 的下一 token logits，估算权重/激活/KV 容量。同步在 [年度主项目](../../../overlap-decode/README.md) 跑通公开小模型基线；本目录保留简化验证代码。

## 通过标准

能独立预测 Tensor shape/stride、写训练循环、解释梯度与前向数据流；Attention 与缓存有数值证据。训练出高质量模型不是要求。验收后进入 CUDA，不无限扩展训练理论。

每项实验记录失败案例，并脱离现成答案复写一个小实现或定位一次错误。更新 [学习进度](../../daily-record/README.md) 时区分确认掌握、代码验证与待验证事项。
