# T14 验证报告

## 问题

验证从 T13 decoder logits 到训练目标和生成循环的连接是否正确：模型在每个位置读入当前及之前 token，预测右移后的标签；训练后从 prefix 开始逐步选 token，并保存每一步前缀和词表 logits。正确性由结构化证据判断，不通过文本观感判断。

## 环境

- 日期：2026-10-15；开始时仓库 `HEAD` 为 `af757a5`（该提交之前已有用户工作区修改，本任务未覆盖或提交它们）。
- Python 3.12.3；PyTorch 2.11.0+cu128；CUDA 可用，但实验固定运行在 CPU float32。
- 设备：CPU；随机种子：1415；PyTorch CPU threads：1。
- 模型：复用 T13 `TinyDecoderLM`；`V=6, D=8, Hq=4, Hkv=2, F=12`，总参数 600。
- 数据：两条固定 5-token 序列 `[0,3,4,5,2]` 与 `[1,5,4,3,2]`；完整序列 `[2,5]`，输入与标签 `[2,4]`。

## 方法

### 标签对齐

程序以 `inputs = sequences[:, :-1]` 和 `targets = sequences[:, 1:]` 右移一个位置。模型输出 logits `[B,S-1,V]=[2,4,6]`，再将 batch/token 轴摊成 `[8,6]`，与 8 个类别索引标签计算 `CrossEntropyLoss`。它接收原始 logits，不预先 softmax。

### 训练闭环

用 AdamW（learning rate 0.02、weight decay 0）做全批次训练；每步依次清梯度、前向、交叉熵、反向和参数更新。达到 loss `<0.01` 后停止，之后再用 eval/inference mode 对更新后的参数独立计算最终 loss，并要求所有位置 argmax 命中标签。

### 生成

从两个训练起始 token 分别 greedy 生成。每步将当前完整 prefix 重新送入模型，只读取最后位置 logits；选择一个 token 后追加到序列。EOS ID 为 2，追加后即停。另用独立 seed 为温度 0.8 的 multinomial sampling 生成一条可重放轨迹。每个 trace entry 保存 prefix、长度为 6 的 logits、选出的 token ID 和策略。

## 结果

脚本原始输出保存在 [train_generate.txt](2026-10-15-t14/train_generate.txt)。

| 检查项 | 观测 |
| --- | --- |
| shift 切片 | 输入 `[[0,3,4,5],[1,5,4,3]]`；目标 `[[3,4,5,2],[5,4,3,2]]` |
| logits shape | `[2,4,6]`；内部 Q `[2,4,4,2]`、K/V `[2,2,4,2]`、Attention `[2,4,4,2]`，残差/norm `[2,4,8]` |
| 参数与训练 | 600 参数；33 次更新；初始 loss `1.86223102`；最终重测 loss `0.00848716`；token accuracy `1.0` |
| 固定标签检查 | 所有 8 个位置的 argmax 与 next-token target 相同 |
| Prefix `[0]` greedy | `[0]→3`，`[0,3]→4`，`[0,3,4]→5`，`[0,3,4,5]→2(EOS)` |
| Prefix `[1]` greedy | `[1]→5`，`[1,5]→4`，`[1,5,4]→3`，`[1,5,4,3]→2(EOS)` |
| Prefix `[0]` sampling | seed 1415、temperature 0.8 下本次抽样也是 `3,4,5,2`；每步 logits 同 greedy |

逐步正确性可独立检查原始 trace：greedy 所选 ID 是其 logits 的最大值索引；下一步 prefix 等于前一步 prefix 加上所选 ID；EOS 被追加后轨迹停止。sampling 偶然生成与 greedy 相同，只说明这一次随机数落在当前主概率 token 上，不代表抽样恒等于贪心。

## 测试

运行命令：

```bash
.venv/bin/python learning/batches/b05-generation-kv-resources/t14-tiny-lm-training-generation/train_generate.py
.venv/bin/python learning/batches/b05-generation-kv-resources/t14-tiny-lm-training-generation/test_t14.py
```

实际结果：脚本退出码 0；5 项 unittest 通过。测试覆盖标签切片、固定样本过拟合与全位置 argmax、贪心轨迹的 prefix/logits 关系、EOS 停止、sampling 固定 seed 的完整重放，以及训练返回后 CPU 线程配置恢复。

## 失败与修正

初始数据草案让两个序列都以 token 0 开始，却分别要求下一个 token 是 3 和 4。相同上下文的确定性 next-token 模型不能同时满足相互冲突的标签，首次运行在 500 步后以 loss `0.17342538` 失败。修正为不同起始 ID 0 和 1 后，标签不再冲突，实际在 33 步满足目标。该失败说明过拟合实验的数据契约本身也必须可满足；它不是通过增加训练时长掩盖掉的问题。

## 结论与边界

观测事实：本地 T13 单层模型能学习这两条小序列上的右移标签，并从对应 prefix 逐 token 生成到 EOS；逐步 logits 支持核对 greedy 决策，seed 支持重放 sampling 轨迹。

本结果不支持对自然语言能力、未见数据泛化、真实 tokenizer、GPU/低精度、长上下文或生成性能作结论。该生成器每步完整重算 prefix，没有 KV Cache，下一项 T15 才对缓存一致性进行单独验证。

## 复现

从仓库根目录使用上方两条命令。源脚本、测试、教程与报告均在本任务工作区新增；本次没有提交或推送。训练输出的 JSON trace 已保存为仓库内文本，数值四舍五入到 6 位小数。训练标签的 argmax 检查使用原 Tensor；生成轨迹测试用记录中的舍入数值比较最大值。本例最高分间隔较大，但舍入轨迹不能支持对任意近并列分数的精确决策重建。

## 文档修订

2026-10-16 的系统基础重写与逐行注释复核见[修订记录](2026-10-16-t14-doc-rewrite.md)。本报告保留原实验日期与结果；本次复测输出独立存放。现有测试未独立重算每步记录的 logits，也未做抽样频率统计检验，不将这些内容算作已经通过的检查。
