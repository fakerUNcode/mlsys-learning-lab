# T07 验证

日期：2026-10-10。范围：CPU 合成分类数据、手写 batch training loop、损失目标、state_dict 保存/加载和 eval logits 对照。

## 环境与命令

- Python 虚拟环境：仓库 `.venv`。
- PyTorch：`2.11.0+cu128`；本实验 device 为 CPU。
- 种子：`20261010`；DataLoader shuffle generator 使用 `20261011`。
- 命令：`.venv/bin/python learning/stage-02/t07-mlp-training-loop/train_mlp.py`。
- 参考：R01 [DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)、[Build Model](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)、[Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[Save & Load](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)，官方在线文档于 2026-10-10 查阅。

## 预先确定的目标

在本次第一次运行之前，代码中固定验收条件：最终训练集交叉熵小于 `0.12`，并且最终/初始损失比例不大于 `0.10`。这是这个小合成任务的优化目标，不代表模型泛化质量。

## 实际输出

```text
seed=20261010 torch=2.11.0+cu128 device=cpu
samples=384 batch_size=48 epochs=80
initial_loss=1.084943
epoch=01 train_loss=0.922251
epoch=10 train_loss=0.081640
epoch=40 train_loss=0.014221
epoch=80 train_loss=0.007215
final_loss=0.007154 ratio=0.0066
save_load_max_logit_error=0.0
T07 checks passed
```

进程退出码为 0。最终损失 $0.007154<0.12$，相对比例 $0.0066<0.10$，两个运行时断言均通过。新建相同结构的 MLP 并加载保存权重后，在 eval 模式、相同全量输入上的 logits 最大绝对差为 $0$；严格 `rtol=0, atol=0` 的断言通过。checkpoint 使用临时目录保存，退出后清理。

## 手写验收

2026-10-10，学习者提交了不查源码的四步 batch 更新、logit 梯度方向、epoch/DataLoader 训练外壳、样本加权 epoch loss 以及阈值和 checkpoint 验证说明，并与 `train_mlp.py` 记录对照。核心流程、张量形状、标签类型、随机流隔离、损失统计和严格 logits 对比说明正确，满足本任务的独立手写解释验收。

批阅时补充两个适用范围：`p - onehot(y)` 的符号直接说明对 logits 本身做梯度下降时各分量的方向；模型实际更新共享参数，不能据此保证每个样本的 logits 分别都按同方向变化。正确类概率接近 1 时该样本的 logit 梯度趋近零，但这不构成一般情形下参数或 logits 有限步停止增长的保证。`zero_grad(set_to_none=True)` 后反向传播会为有梯度的参数建立 `.grad`；默认 `backward()` 直接调用要求标量输出，非标量可提供同形状 `gradient` 做向量-雅可比积。

## 证据范围与未验证项

- 训练脚本实际执行了 `zero_grad → model(batch) → CrossEntropyLoss → loss.backward → optimizer.step`，并通过固定种子重放一轮完整训练。
- 运行时断言验证损失阈值和保存/加载后的 logits 一致；这不是独立测试集泛化证据。
- 笔记给出手写分步、数据形状、梯度公式、执行顺序、保存恢复流程及限制。
- 手写验收基于学习者提交的练习答案及其与脚本记录的对照；该记录验证了本任务要求的核心解释能力，不代表对所有 PyTorch autograd 细节的全面考核。
- 未覆盖 GPU、随机性跨版本/设备逐位保证、持久化断点续训、优化器状态恢复、数值有限差分对照或大模型质量。
