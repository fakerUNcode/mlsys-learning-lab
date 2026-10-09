# T06 梯度累积与推理模式验证报告

日期：2026-10-09。结论：CPU `float64` 故障对照脚本通过。实验观察到重复 `backward()` 会累加 `.grad`；省略清零会改变后续 SGD 更新；`detach` 截断上游图；`model.eval()` 改变 Dropout 行为但仍允许反向传播；`no_grad` 和 `inference_mode` 创建的 Tensor 在后续 autograd 使用上存在区别。

## 来源与环境

- 学习材料：[T06 README](../../learning/stage-02/t06-gradient-accumulation-and-inference-modes/README.md)；源码：[compare_grad_modes.py](../../learning/stage-02/t06-gradient-accumulation-and-inference-modes/compare_grad_modes.py)。
- 参考资料：R01 [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)；R03 [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)、[Zeroing out gradients](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)。官方文档于 2026-10-09 查阅。
- 术语核对：[Dropout API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)、[Parameters, buffers, and modules](https://docs.pytorch.org/docs/stable/notes/modules.html)、[BatchNorm1d API](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)。用于核对模块模式、参数登记、buffer 和 BatchNorm 运行统计的笔记表述。
- 环境：仓库 `.venv`，PyTorch `2.11.0+cu128`；运算在 CPU `torch.float64` 上进行；Dropout 使用固定随机种子 17。没有数据集、GPU 运算或性能计时。

## 复现命令

从仓库根目录运行：

```bash
.venv/bin/python learning/stage-02/t06-gradient-accumulation-and-inference-modes/compare_grad_modes.py
```

程序用断言验证数值与状态。全部通过时最后一行是 `T06 checks passed`。

## 观测结果

```text
accumulation: first_grad= 1.0
accumulation: after_second_backward= 5.0
accumulation: after_zero_grad= None
accumulation: fresh_second_sample_grad= 4.0
optimizer omission: stale_grad= 1.0
optimizer omission: accumulated_before_second_step= 4.6
optimizer omission: final_weight= 0.44000000000000006
correct loop: final_weight= 0.54
detach: detached_requires_grad= False detached_grad_fn= None
detach: new_leaf_grad= 12.0
detach: original_source_grad= None
model mode: train output requires_grad= True
model mode: train dropout zero_count= 10
model mode: eval output requires_grad= True
model mode: eval output equals input= True
model mode: eval backward reaches weight/input= True True
no_grad: module.training= True
no_grad: output requires_grad= False
no_grad: dropout zero_count= 10
no_grad output: later weight_grad= [2.0, 4.0]
no_grad output: original source_grad= None
inference output: later grad-mode use= rejected ( Inference tensors cannot be saved for backward. ... )
inference output: is_inference= True
inference output: clone is_inference= False
inference output: cloned feature weight_grad= [2.0, 4.0]
T06 checks passed
```

### 梯度累积与更新

损失为 `L=0.5*(w*x)^2`，梯度为 `dL/dw=w*x^2`。固定 `w=1`、样本 `x=1` 和 `x=2` 分别产生梯度 1 和 4。两次 backward 之间不更新参数时，缓冲区从 1 累到 5；清零后只处理第二个样本得到 4。

单独比较 SGD 路径时，第一步将 `w` 从 1 更新到 0.9。第二个样本在新参数处的梯度是 `0.9*2^2=3.6`。故障路径未清零，将旧值 1 一并相加成 4.6，第二次更新到 0.44；正确路径清零后使用 3.6，更新到 0.54。两组参数从相同初值开始，数据顺序与学习率相同。

### 计算图与模式

- `detach()` 后的标量值 6 被重新设为可求导叶子，下游平方产生叶子梯度 12；原始输入梯度为 None。
- 训练态 Dropout 对 16 个单位值置零 10 个；固定种子确保复现。评估态输出恢复为输入，但输出仍 `requires_grad=True`，输入和权重的 backward 梯度均存在。
- 在 `model.train()` 状态中使用 `no_grad()`，Dropout 仍置零 10 个值，而输出不需要梯度。这直接展示模块状态与 autograd 记录是两个独立开关。
- no-grad 中创建的 `[2,4]` 特征可在上下文外作为常量参与乘法，产生可训练权重梯度 `[2,4]`；inference mode 中创建的 Tensor 带 inference 标记，在本例乘法的 backward 需要保存它时触发 RuntimeError。上下文外 clone 得到普通 Tensor，后续权重梯度计算成功。

## 覆盖和限制

| 验收点 | 证据 | 限制 |
| --- | --- | --- |
| 演示不清梯度的累积 | `.grad` 从 1 累加到 5；省略清零时参数到 0.44 | 单标量、SGD，无动量、权重衰减或梯度缩放 |
| 正确 `zero_grad` | `set_to_none=True` 后为 None；正确第二步参数到 0.54 | 未比较 `set_to_none=False` 的性能或内存影响 |
| `detach` | 新叶子 grad=12，上游 grad=None | 未执行共享存储的原地改写测试 |
| eval 与禁用梯度 | eval 仍可 backward；train+no-grad 仍执行 Dropout | 使用 Dropout 作为模块行为例子，没有测试 BatchNorm |
| no-grad 与 inference mode | no-grad 结果进入后续图成功；一个 inference Tensor 保存场景失败；clone 后成功 | 推理 Tensor 是否触发限制取决于后续操作是否需要由 autograd 保存它；未测性能差异 |

该脚本通过只能证明列出的固定路径在当前 PyTorch 环境中符合预期，不能推出所有模型子模块、优化器配置或张量操作都具有相同表现。实验也未验证 CUDA、AMP、DDP 或显存收益。

## 笔记与注释补充复核

按最新仓库约定补充了 T06 README 和脚本注释：在术语首次出现处解释 Dropout、`nn.Parameter`、BatchNorm、autograd、参数与 buffer；说明 `train()/eval()` 与梯度模式分别控制什么，并把实际使用的 Linear/Sequential、形状、公式、函数职责和验证边界连到代码。BatchNorm 仅作为概念对照，未把本脚本输出当作其运行证据。

本次补充后再次运行复现命令，全部断言通过，输出与上方记录一致。剥离 Python 注释和文档字符串后，脚本 AST 与本轮注释更新前相同；因此实验数值代码未改变。README 的本地链接、可移植公式格式和尾随空格检查通过。上述检查只覆盖本报告所列脚本与 Markdown 文件。
