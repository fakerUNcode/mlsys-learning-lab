# PyTorch 与模型基础

状态：B01/T01～T04 的材料和对应验证已完成；B02/T05 的手算与数值对照已完成，讲解题仍可用于读者自测；T06 梯度累积与推理模式已完成故障对照；T07 的脚本、详细笔记、程序验收及学习者手写练习已完成；Attention 与模型前向尚待补充。此目录延续历史编号，阶段顺序见[任务清单](../../plan/TASKS.md)。

## 前四周

| 周次 | 验证任务 | 需要提交的证据 |
| --- | --- | --- |
| 1 | Tensor、广播、索引、reshape/transpose、stride、device/dtype | 自己预测维度；连续与非连续输入对照；环境报告 |
| 2 | 链式法则、有限差分、autograd、梯度累积、MLP 训练 | 手算计算图；数值对照；可独立解释的训练脚本 |
| 3 | Q/K/V、因果 mask、多头 Attention | 标注维度的 reference；与参考实现对照；未来 token 不影响过去位置的测试 |
| 4 | Embedding、残差、RMSNorm、SwiGLU、Decoder Block | 简化前向、结构图、边界与数值测试 |

每个实验建立独立目录，包含代码、验证和 README。已有 Tensor 实验；后续任务清单不计为完成成果。

## 章节目录

每个已学习项目都按 `tNN-章节名` 放在本目录下。章节内保留对应 README、代码和分章材料；验证报告与独立验收记录集中放在仓库的 `reports/` 和 `plan/completed/`。

| 目录 | 主题 | 入口与证据 |
| --- | --- | --- |
| [t01-environment-verification](t01-environment-verification/README.md) | 当前设备环境验收 | [环境报告](../../reports/stage-02/2026-09-27-t01-environment.md) |
| [t02-tensor-basics](t02-tensor-basics/README.md) | Tensor 创建、索引、广播与批量矩阵乘法 | [完成记录](../../plan/completed/T02.md) |
| [t03-tensor-layout-and-storage](t03-tensor-layout-and-storage/README.md) | storage、stride、视图、dtype 与 device | [完成记录](../../plan/completed/T03.md) |
| [t04-cpp-buffer-ownership](t04-cpp-buffer-ownership/README.md) | C++ Buffer 所有权与移动 | [完成记录](../../plan/completed/T04.md)、[验证报告](../../reports/stage-02/2026-10-08-t04-validation.md) |
| [t05-chain-rule-and-finite-differences](t05-chain-rule-and-finite-differences/README.md) | 链式法则、autograd 与有限差分 | [验证报告](../../reports/stage-02/2026-10-08-t05-validation.md) |
| [t06-gradient-accumulation-and-inference-modes](t06-gradient-accumulation-and-inference-modes/README.md) | 梯度累积、detach 与推理模式 | [完成记录](../../plan/completed/T06.md)、[验证报告](../../reports/stage-02/2026-10-09-t06-validation.md) |
| [t07-mlp-training-loop](t07-mlp-training-loop/README.md) | MLP、DataLoader 与独立训练闭环 | [验证报告](../../reports/stage-02/2026-10-10-t07-validation.md) |

T02 与 T03 共用的 Tensor 术语表在[本目录](terms.md)，配图源文件索引在[figure-index.md](figure-index.md)。旧的 `learning/stage-02/tensors/` 仅保留兼容入口，新的学习入口以本表为准。

## 第五至八周

实现小型语言模型训练与生成，验证有/无 KV Cache 的下一 token logits，估算权重/激活/KV 容量。本目录先保留可单独运行的简化验证代码；公开模型基线尚未覆盖。

## 通过标准

能独立预测 Tensor shape/stride、写训练循环、解释梯度与前向数据流；Attention 与缓存有数值证据。训练出高质量模型不是要求。验收后进入 CUDA，不无限扩展训练理论。

每项实验记录失败案例，并脱离现成答案复写一个小实现或定位一次错误。更新 [学习进度](../../daily-record/README.md) 时区分确认掌握、代码验证与待验证事项。
