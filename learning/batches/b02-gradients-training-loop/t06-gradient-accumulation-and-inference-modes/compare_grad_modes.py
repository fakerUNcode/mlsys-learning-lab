"""T06 梯度状态对照实验的可运行入口。

PyTorch 是用于张量计算和机器学习的 Python 库；Tensor 是它保存标量、向量或矩阵
数据的对象。autograd 是 automatic differentiation（自动微分）系统：前向计算时记录
依赖关系，backward（反向传播）时沿这些依赖用链式法则计算梯度。本文件与同目录
README.md（概念、语法、实验路径）及验证报告（实测环境和输出）配合，实际创建 Tensor、
执行自动微分/优化器操作并用断言检查结果。forward（前向计算）从输入算出输出与损失；
optimizer（优化器）根据梯度更新交给它管理的参数。

直接运行本文件时，最下方的入口依次调用四组实验；数据是手工构造的小型 CPU
(Central Processing Unit，中央处理器) Tensor，统一使用 torch.float64（64 位双精度
浮点数），不读取数据集，也不测真实模型的训练速度。实验先比较 backward 如何把梯度
加进 .grad，再对照漏清零和正确清零的 SGD（Stochastic Gradient Descent，随机梯度下降）
更新；接着展示 detach 如何切断旧计算图；最后比较模块 train/eval 行为和 no-grad/
inference-mode（推理模式）下的梯度记录差异。断言条件为假时，Python 会抛出
AssertionError，后续实验不会继续执行。
"""

# future import 是 Python 的特殊导入；它让本文件中的类型注解延后求值，因此
# `torch.Tensor`、`nn.Sequential` 等注解主要用于阅读和静态工具，不会在定义函数时求值。
from __future__ import annotations

# `import torch` 导入 PyTorch 的公开 Python 接口；`from torch import nn` 把 torch.nn
# （neural network，神经网络）命名空间导入为短名称 nn，后文用这个别名访问层和参数类。
import torch
from torch import nn


# 此命名约定表示本实验统一使用 64 位浮点 Tensor；大写名称只是 Python 约定，
# 并不会自动阻止后续代码重新给 DTYPE 赋值。
DTYPE = torch.float64


def scalar_loss(weight: torch.Tensor, x: float, target: float = 0.0) -> torch.Tensor:
    """计算一个标量样本的平方损失并返回零维 Tensor。

    `weight: torch.Tensor` 和 `x: float` 是 Python 类型注解，说明参数预期分别为
    Tensor 与 Python 浮点数；`-> torch.Tensor` 注明返回值类型，解释器不会因此自动
    检查类型。target 是可省略的 Python 数值，省略时使用默认值 0.0。本函数由梯度
    累积实验调用，输入参数不被原地修改，输出 loss 保留对 weight 的 autograd 关系。

    先算 prediction = weight*x，再算 error = prediction-target，最后返回
    0.5*error^2。平方求导的系数 2 与前面的 0.5 抵消，因此 d(loss)/d(weight)
    = (weight*x-target)*x；例如 weight=1、x=2、target=0 时，loss=2，梯度为 4。
    """
    prediction = weight * x
    error = prediction - target
    return 0.5 * error.square()


def show_gradient_accumulation() -> None:
    """运行梯度缓冲区和 SGD 更新的故障/正确路径对照。

    本函数无参数、无返回值，由 main 调用；它创建标量 Parameter 和 SGD 优化器，
    打印中间梯度/最终参数，并用断言核对计算结果。`-> None` 是返回类型注解，表示
    调用方依赖打印和断言观察结果，而不是接收返回对象。

    `nn.Parameter` 是 Tensor 的参数子类，默认 `requires_grad=True`，表示默认梯度模式
    下 autograd 会为它计算梯度；赋给 `nn.Module` 属性时会被登记并出现在
    `model.parameters()` 中。本函数没有构造持有 weight 属性的
    Module，因而直接把 `[weight]` 参数列表交给 `torch.optim.SGD`。SGD 全称 Stochastic
    Gradient Descent（随机梯度下降），通常用随机样本/小批次估计梯度；本例样本顺序固定，
    只演示更新规则 `weight <- weight - lr*weight.grad`，其中 lr 是 learning rate（学习率）。
    loss 对 weight 的导数是 `(weight*x)*x`：初值 1 时，x=1 贡献 1，x=2 贡献 4。先只
    调用两次 backward、不 step，便能单独观察 .grad 从 1 累加到 5。

    随后两条训练路径都从 weight=1、按 x=1 再 x=2、使用相同学习率出发。故障路径
    第一次 step 后 weight=0.9，却漏掉第二批前的清零；第二批新梯度 3.6 被加到旧梯度
    1 上，优化器使用 4.6 更新。正确路径每批先清零，再 backward、step，因此第二批
    的梯度只来自当前参数和样本，最终得到 0.54。这里的批次各含一个样本，未演示
    loss 按 batch 大小取均值等其他训练约定。
    """
    weight = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
    optimizer = torch.optim.SGD([weight], lr=0.1)

    # zero_grad 是优化器方法，会清理它管理的参数梯度；set_to_none=True 把 .grad
    # 设为 None，而不是写入全零 Tensor。此处参数是叶子节点，所以 backward 将梯度保存在它的 .grad 属性。
    optimizer.zero_grad(set_to_none=True)
    scalar_loss(weight, x=1.0).backward()
    first_grad = weight.grad.item()

    # backward 默认把新梯度累加到已有 .grad，而不会替你清零；w 仍为 1，故 x=2
    # 贡献 4，缓冲区由 1 变为 1+4=5。这个例子把两个样本的梯度求和，没有做平均。
    scalar_loss(weight, x=2.0).backward()
    accumulated_grad = weight.grad.item()

    # 再清一次后，assert 检查 .grad 确实为空；后续 backward 会为这次计算写入新梯度。
    # 参数没有经过 step，仍为 1，因此这次只计算 x=2 的梯度 4。
    optimizer.zero_grad(set_to_none=True)
    assert weight.grad is None
    scalar_loss(weight, x=2.0).backward()
    fresh_grad = weight.grad.item()

    print("accumulation: first_grad=", first_grad)
    print("accumulation: after_second_backward=", accumulated_grad)
    print("accumulation: after_zero_grad= None")
    print("accumulation: fresh_second_sample_grad=", fresh_grad)
    assert first_grad == 1.0
    assert accumulated_grad == 5.0
    assert fresh_grad == 4.0

    # 故障路径按“清零 -> 前向/损失 -> backward -> step”处理第一批，得到 weight=0.9。
    # 这里故意在第二批前不清零，所以第二批梯度 3.6 与残留的 1 相加成 4.6；随后
    # 第二次 step 执行 0.9 - 0.1*4.6 = 0.44。行尾注释说明这处遗漏是实验故障。
    broken_weight = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
    broken_optimizer = torch.optim.SGD([broken_weight], lr=0.1)
    broken_optimizer.zero_grad(set_to_none=True)
    scalar_loss(broken_weight, x=1.0).backward()
    broken_optimizer.step()
    stale_grad = broken_weight.grad.item()
    scalar_loss(broken_weight, x=2.0).backward()  # 故意缺少 zero_grad。
    broken_second_grad = broken_weight.grad.item()
    broken_optimizer.step()

    # 这个 for 循环依次从二元素 tuple `(1.0, 2.0)` 取样本；每轮都清零，再计算当前
    # 样本损失、反传并更新参数。它与故障路径保持初值、样本次序和学习率相同，只修正
    # 梯度清理时机，因此对照结果能归因于是否清零。
    correct_weight = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
    correct_optimizer = torch.optim.SGD([correct_weight], lr=0.1)
    for sample_x in (1.0, 2.0):
        correct_optimizer.zero_grad(set_to_none=True)
        scalar_loss(correct_weight, x=sample_x).backward()
        correct_optimizer.step()

    print("optimizer omission: stale_grad=", stale_grad)
    print("optimizer omission: accumulated_before_second_step=", broken_second_grad)
    print("optimizer omission: final_weight=", broken_weight.item())
    print("correct loop: final_weight=", correct_weight.item())
    assert broken_second_grad == 4.6
    assert abs(broken_weight.item() - 0.44) < 1e-12
    assert abs(correct_weight.item() - 0.54) < 1e-12


def show_detach() -> None:
    """验证 detach 后的梯度停在新叶子，不会回传到原始输入。

    无参数、无返回值，由 main 调用；打印原 Tensor、分离后 Tensor 的状态与梯度，
    并断言原始 source.grad 仍为空。初始 source=2 且 requires_grad=True，乘 3 得到
    intermediate=6；detach 创建一个与其共享数值存储、但不连接旧 autograd 图的 Tensor。
    随后的 `requires_grad_()` 是带下划线的原地属性修改方法：它在这个分离 Tensor 上
    开启梯度跟踪，使其成为新叶子。对新叶子平方并 backward，梯度为 2*6=12；图已在
    detach 处截断，因此 source 不会收到梯度。实验没有通过任一侧的原地数值写入来展示
    共享存储行为；实际代码只读取这些值。
    """
    source = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    intermediate = source * 3.0

    # detach() 得到共享数值存储但不连接 intermediate 历史的 Tensor；它起初不要求梯度，
    # 且没有 grad_fn。requires_grad_() 修改返回 Tensor 自身的跟踪属性，让它成为新叶子。
    detached_leaf = intermediate.detach().requires_grad_()
    downstream_loss = detached_leaf.square()
    downstream_loss.backward()

    print("detach: source=", source.item(), "intermediate=", intermediate.item())
    print(
        "detach: detached_requires_grad=",
        intermediate.detach().requires_grad,
        "detached_grad_fn=",
        intermediate.detach().grad_fn,
    )
    print("detach: new_leaf_grad=", detached_leaf.grad.item())
    print("detach: original_source_grad=", source.grad)
    assert detached_leaf.grad.item() == 12.0  # d(z^2)/dz，z=6。
    assert source.grad is None  # 梯度不会穿过 detach 回到原图。


def make_probe_model() -> nn.Sequential:
    """创建固定权重的线性层与随机失活层，并返回组合模型。

    无参数，由 show_model_mode_and_grad_modes 调用；返回值类型注解说明结果是
    nn.Sequential 模型。Sequential 是按给定次序串联子模块的 PyTorch 容器。Linear
    （线性层）将输入向量映射到输出向量；PyTorch 保存权重形状为 `(输出数, 输入数)`，
    计算 `input @ weight.T + bias`。符号 `@` 表示矩阵乘法，`.T` 把权重矩阵的行列交换，
    使 `(输入数, 输出数)` 与输入向量维度相乘后得到输出数个值。这里 `bias=False` 关闭加到每个输出上的可学习偏置，
    并把权重写成 16x16 单位矩阵（对角线为 1、其他位置为 0），因此 16 维输入输出相同。
    激活值是层之间传递的中间数值；Dropout（随机失活正则化层）对这些值做随机屏蔽。
    Dropout(p=0.5) 训练态以 0.5 概率将元素置零，并把保留值缩放为两倍；eval 态是恒等
    映射。Dropout 没有可学习参数；training 标志决定随机屏蔽行为，autograd 梯度模式
    另行决定是否记录该层运算。
    """
    model = nn.Sequential(
        nn.Linear(16, 16, bias=False, dtype=DTYPE),
        nn.Dropout(p=0.5),
    )
    # `with` 是 Python 上下文管理器语法：进入 no_grad 区域时暂停 autograd 记录，离开
    # 区域时恢复原状态。copy_ 是 Tensor 的原地复制方法；把 16x16 单位矩阵写入权重，
    # 可令 Linear 输出等于输入，同时避免把初始化写入操作记进训练计算图。Parameter
    # 仍保持 requires_grad=True，之后正常前向仍能为它计算梯度。
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(16, dtype=DTYPE))
    return model


def show_model_mode_and_grad_modes() -> None:
    """分别改变模块行为状态与 autograd 记录状态，并检查各自效果。

    无参数、无返回值，由 main 调用。它用一个 16 元素全 1 输入运行相同模型：train()
    使 Dropout 随机屏蔽元素并把保留值放大到 2；eval() 令 Dropout 变成恒等操作，但
    不关闭梯度跟踪；no_grad() 只在上下文内关闭计算图记录，不会自动调用 eval()。
    因而本实验分别读取输出的 requires_grad、零元素数量、参数梯度和输入梯度，说明
    模块行为开关与 autograd 开关控制的是两件不同的事。BatchNorm 是 Batch Normalization
    （批归一化）层的简称：默认跟踪运行统计时，训练态按当前批的通道均值/方差归一化并
    更新 running buffers，评估态使用保存的运行统计；no_grad 不会阻止训练态更新这些值。
    本脚本没有创建 BatchNorm，因此这里的运行结果不验证 BatchNorm 行为。
    """
    model = make_probe_model()
    values = torch.ones(16, dtype=DTYPE, requires_grad=True)

    # Module.train() 会把该模块及子模块的 training 状态设为 True；Dropout 因此启用。
    # torch.manual_seed(17) 设置 PyTorch 伪随机数生成器的种子，使同一环境中重置种子后
    # 的 Dropout 掩码可复现；不保证不同硬件或 PyTorch 版本给出完全相同序列。训练态
    # 保留值按 `1/(1-p)=2` 缩放，所以全 1 输入的输出是 0 或 2，随机平均值为 1。
    model.train()
    torch.manual_seed(17)
    train_output = model(values)
    train_requires_grad = train_output.requires_grad
    train_masked_count = int((train_output == 0).sum().item())

    # Module.eval() 等价于切换为 eval 模式（training=False）；Dropout 不再随机屏蔽，
    # 本模型的恒等 Linear 使输出等于输入。eval() 不会改变 requires_grad 或关闭 autograd，
    # 所以这里仍可 backward，并分别为 Linear 权重和需要梯度的输入 values 求导。
    model.eval()
    eval_output = model(values)
    eval_requires_grad = eval_output.requires_grad
    # Module.zero_grad 清理该模型参数的 .grad；values 是模型外部的输入 Tensor，不在
    # 模型参数表里，所以还要单独把 values.grad 设为 None，避免把前次结果混入本次观察。
    model.zero_grad(set_to_none=True)
    values.grad = None
    eval_output.sum().backward()
    eval_weight_grad = model[0].weight.grad is not None
    eval_input_grad = values.grad is not None

    # train() 把模块行为切回训练态；no_grad 上下文则只暂停这一小段前向的图记录。
    # 因此 Dropout 仍按训练规则工作，输出不要求梯度；离开 with 后 autograd 设置恢复。
    model.train()
    torch.manual_seed(17)
    with torch.no_grad():
        train_without_grad = model(values)
    no_grad_output_requires_grad = train_without_grad.requires_grad
    no_grad_training_mode = model.training
    no_grad_masked_count = int((train_without_grad == 0).sum().item())

    print("model mode: train output requires_grad=", train_requires_grad)
    print("model mode: train dropout zero_count=", train_masked_count)
    print("model mode: eval output requires_grad=", eval_requires_grad)
    print("model mode: eval output equals input=", torch.equal(eval_output, values))
    print("model mode: eval backward reaches weight/input=", eval_weight_grad, eval_input_grad)
    print("no_grad: module.training=", no_grad_training_mode)
    print("no_grad: output requires_grad=", no_grad_output_requires_grad)
    print("no_grad: dropout zero_count=", no_grad_masked_count)

    assert train_requires_grad
    assert train_masked_count > 0
    assert eval_requires_grad and torch.equal(eval_output, values)
    assert eval_weight_grad and eval_input_grad
    assert no_grad_training_mode and not no_grad_output_requires_grad
    assert no_grad_masked_count == train_masked_count


def show_no_grad_and_inference_mode() -> None:
    """对照 no_grad 与 inference_mode 产物参与后续求导时的差异。

    无参数、无返回值，由 main 调用。两个长度为 2 的输入分别是需梯度的 source=[1,2]
    和 weight=[3,4]。本实验先在 no_grad 中生成 feature，再让 feature 与可训练 weight
    相乘并求和；这时乘法反向只需保存 feature 来计算 weight 的梯度，因此可成功得到
    [2,4]，source.grad 仍为空。随后在 inference_mode 中生成 inference Tensor，尝试同样
    的后续用法；当反向需要保存 inference Tensor 时，当前 PyTorch 会抛出 RuntimeError。
    最后在上下文外 clone 一份普通 Tensor，再重复后续求导，验证复制后的数据可被保存。
    这是针对本实验中“反向需要保存该值”的具体限制，不表示 inference Tensor 在所有
    后续运算中一概不可读取或不可使用。
    """
    source = torch.tensor([1.0, 2.0], dtype=DTYPE, requires_grad=True)
    trainable_weight = torch.tensor([3.0, 4.0], dtype=DTYPE, requires_grad=True)

    # no_grad 上下文语法暂停图记录，但不会把普通 Tensor 变成 inference Tensor；离开
    # 上下文后，feature 可参加新图。本例乘法对 weight 求导需要保存 feature 的数值，
    # 而 feature 是普通 Tensor，所以反向可成功；source 的旧图未连接到 feature。
    with torch.no_grad():
        no_grad_features = source * 2.0
    later_loss = (no_grad_features * trainable_weight).sum()
    later_loss.backward()
    print("no_grad output: requires_grad=", no_grad_features.requires_grad)
    print("no_grad output: later weight_grad=", trainable_weight.grad.tolist())
    print("no_grad output: original source_grad=", source.grad)
    assert not no_grad_features.requires_grad
    assert torch.equal(trainable_weight.grad, torch.tensor([2.0, 4.0], dtype=DTYPE))
    assert source.grad is None

    # inference_mode() 也是上下文管理器；它不记录 autograd 图，并把其中创建的 Tensor
    # 标记为 inference Tensor。后面的乘法把 weight 作为可求导因子，因此反向公式需要
    # feature 的值；PyTorch 在这种需要保存 inference Tensor 的场景抛 RuntimeError。
    with torch.inference_mode():
        inference_features = source * 2.0
    inference_flag = inference_features.is_inference()
    inference_rejected = False
    # try/except 是 Python 异常处理语法：只把预期的 RuntimeError 当成本实验观测结果，
    # 其余类型的错误仍会向外传播。字符串检查用于确认错误消息指向 inference Tensor 限制。
    try:
        (inference_features * trainable_weight.detach().requires_grad_()).sum().backward()
    except RuntimeError as error:
        inference_rejected = "Inference tensors" in str(error)
        print("inference output: later grad-mode use= rejected (", str(error).splitlines()[0], ")")

    # clone() 在 inference_mode 已结束后复制数据，创建普通 Tensor；这个新对象不再带
    # inference 标记，因而可以在后续反向需要时被保存。这里显式复制也意味着它与原
    # inference Tensor 的存储分开；实验只核对类型标记和 weight 梯度。
    regular_features = inference_features.clone()
    later_weight = torch.tensor([3.0, 4.0], dtype=DTYPE, requires_grad=True)
    (regular_features * later_weight).sum().backward()

    print("inference output: is_inference=", inference_flag)
    print("inference output: clone is_inference=", regular_features.is_inference())
    print("inference output: cloned feature weight_grad=", later_weight.grad.tolist())
    assert inference_flag
    assert inference_rejected
    assert not regular_features.is_inference()
    assert torch.equal(later_weight.grad, torch.tensor([2.0, 4.0], dtype=DTYPE))


def main() -> None:
    """打印运行环境摘要并按顺序启动全部四组实验。

    无参数、无返回值，是本文件入口调用的协调函数；它先报告 PyTorch 版本和本实验
    使用的 dtype，再依次调用梯度累积、detach、模块模式/梯度模式、no_grad/inference
    mode 实验。各实验内部的 assert 若失败会抛 AssertionError，异常不会在这里吞掉，
    所以进程会以失败状态终止；全部断言通过后才打印最终通过消息。
    """
    print("torch version:", torch.__version__)
    print("device: cpu; dtype:", DTYPE)
    show_gradient_accumulation()
    show_detach()
    show_model_mode_and_grad_modes()
    show_no_grad_and_inference_mode()
    print("T06 checks passed")


# Python 在直接运行脚本时把特殊变量 __name__ 设为 "__main__"；作为模块导入时则
# 不执行此分支。这个惯用入口保护让测试或其他代码导入函数时不会自动启动整组实验。
if __name__ == "__main__":
    main()
