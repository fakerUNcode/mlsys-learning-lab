"""T07 可运行的独立 MLP 训练闭环。

文件职责与执行位置：从仓库根目录运行本文件，main() 依次造合成数据、组装
DataLoader、建立 MLP、训练、检查预先写下的损失目标，再保存并重新加载模型。
相关文件：同目录 README.md 逐步讲语法、形状、梯度公式与设计选择；
reports/stage-02/2026-10-10-t07-validation.md 保存实际运行证据。

这里的“独立”指训练循环显式写出 forward、loss、backward 和 optimizer.step，
不调用封装好的 Trainer。Linear、ReLU、CrossEntropyLoss 和 SGD 仍使用 PyTorch
已实现的基础算子。数据和模型都只用于 CPU 小实验，不代表真实数据或大模型效果。
"""

# Path 提供与操作系统路径分隔符无关的路径拼接；本文件用它定位临时 checkpoint。
from pathlib import Path
# tempfile 创建自动清理的临时目录；保存/重载验收结束后 checkpoint 会随目录删除。
import tempfile

# torch 提供 Tensor、随机数、autograd、优化器和模型状态保存/加载接口。
import torch
# torch.nn 在这里简称 nn，用于定义 Module、Linear、ReLU 与损失函数。
from torch import nn
# DataLoader 按批迭代数据；TensorDataset 将同一首轴上的特征和标签配对。
from torch.utils.data import DataLoader, TensorDataset


# 固定种子使数据生成、模型初始化和批次打乱可复现；它不保证跨设备或跨 PyTorch
# 版本逐位一致。CPU + 当前固定算子下，本脚本检查的是实际数值目标。
# SEED 是数据和模型初始化使用的整数随机种子；记录它便于本环境重放实验。
SEED = 20261010
# SAMPLE_COUNT 是合成数据总样本数；384 可被下方类别数 3 整除，保证每类等量。
SAMPLE_COUNT = 384
# BATCH_SIZE 是一次清梯度、反向和参数更新所用的样本数；本例每轮 8 个批次。
BATCH_SIZE = 48
# EPOCHS 是完整遍历训练集的次数；每个 epoch 都让模型再看过全体 384 个点一次。
EPOCHS = 80
# LEARNING_RATE 是 SGD 更新式中的正步长 η；0.08 是本合成任务经运行验证的固定值。
LEARNING_RATE = 0.08
# HIDDEN_SIZE 是隐藏层神经元/特征数，决定第一层输出轴长度以及第二层输入宽度。
HIDDEN_SIZE = 16
# CLASS_COUNT 是分类标签的可能取值数，类别索引为 0、1、2，logits 的末轴长度也为 3。
CLASS_COUNT = 3

# 验收目标在看训练输出前固定：训练集交叉熵至少下降 90%，并低于 0.12。
# 它用于检验闭环确实学到这个容易的合成分类任务，不表示泛化能力。
# MAX_FINAL_LOSS 是全训练集最终平均交叉熵的严格上限；低于它代表本次任务拟合充分。
MAX_FINAL_LOSS = 0.12
# MAX_FINAL_TO_INITIAL_RATIO 是最终/初始损失的最大比例；0.10 要求至少下降 90%。
MAX_FINAL_TO_INITIAL_RATIO = 0.10


def make_dataset(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """构造三团二维点，并返回 features [N,2] 与类别 labels [N]。

    参数 seed 是控制本函数局部随机生成器的整数种子；相同 PyTorch 环境下相同
    seed 会重现相同数据。每类有 SAMPLE_COUNT/3 个点，围绕三个固定中心加独立高斯噪声。
    局部 torch.Generator 只控制本函数随机数，不改全局随机数状态；类别标签为
    int64，因为 CrossEntropyLoss 将它解释成类别索引。返回特征 [N,2] 和标签 [N]。
    并发要求：本函数不共享可变生成器状态；并发调用各自创建生成器，但每次传入相同
    seed 会生成相同数据。调用者负责避免把多份重复数据误当成不同随机样本。
    """
    # generator 是本函数专用 CPU 随机流；manual_seed 让后续噪声和排列可重复。
    generator = torch.Generator().manual_seed(seed)
    # centers 形状 [3,2]，每行是一个类别在二维平面上的中心坐标，dtype 默认 float32。
    centers = torch.tensor([[-1.4, -0.8], [1.3, -0.9], [0.0, 1.4]])
    # labels 形状 [384]、int64；repeat_interleave 让每个类别索引连续出现 128 次。
    labels = torch.arange(CLASS_COUNT).repeat_interleave(SAMPLE_COUNT // CLASS_COUNT)
    # noise 形状 [384,2]；标准正态随机数乘 0.42，把每个坐标的标准差缩放为 0.42。
    noise = torch.randn(SAMPLE_COUNT, 2, generator=generator) * 0.42
    # features 形状 [384,2]；centers[labels] 按每个样本标签查出中心，再逐坐标加噪声。
    features = centers[labels] + noise

    # 先用同一个随机排列打乱样本和标签，再让 DataLoader 每轮重新打乱训练顺序。
    # order 是 [384] 的随机索引排列；同一索引同时重排特征和标签以保持样本配对。
    order = torch.randperm(SAMPLE_COUNT, generator=generator)
    return features[order], labels[order]


class SmallMLP(nn.Module):
    """两层全连接分类器；输入 [B,2]，输出未归一化 logits [B,3]。

    nn.Module 基类会登记子模块及其 Parameter，使 parameters()、state_dict()
    和 .to(device) 能找到它们。Linear.weight 形状是 [out,in]。
    创建 SmallMLP() 会生成模型对象；__init__ 不接收业务参数，也不返回有用值，
    它把两层 Linear 和 ReLU 保存到这个对象上。Linear 初值由 PyTorch 全局随机状态决定。
    并发要求：构造过程不共享模型对象，但全局随机数种子应由调用方在创建模型前设置，
    避免并发构造时竞争同一随机流。
    """

    def __init__(self) -> None:
        # super() 指向 nn.Module 父类；先初始化父类，才可登记本对象的子模块和参数。
        super().__init__()
        # self.hidden 是 2→16 仿射层；weight [16,2]、bias [16] 均自动登记为可训练参数。
        self.hidden = nn.Linear(2, HIDDEN_SIZE)
        # self.activation 激活层是逐元素 ReLU 子模块；形状不变，负值置零，提供非线性。
        self.activation = nn.ReLU()
        # self.output 是 16→3 仿射层；输出三个类别的 logits，不在此处做 softmax。
        self.output = nn.Linear(HIDDEN_SIZE, CLASS_COUNT)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """依次执行仿射映射、ReLU、仿射映射；不做 softmax。

        Python 会在 model(features) 时把当前模型对象自动传给 self，并调用此 forward。
        features: torch.Tensor 与 -> torch.Tensor 是类型提示，用于说明预期类型，不负责强制转换。
        参数 features 是形状 [B,2] 的浮点批次，每行保存一个二维样本；返回 [B,3]
        浮点 logits，每行保存该样本对三个类别的分数。第一层把每个二维点变成 16 维特征；ReLU 将负值置零；末层产生每类
        一个分数。CrossEntropyLoss 内部会稳定地结合 log-softmax 与 NLL，
        所以传入 logits 比先手动 softmax 更合适。本方法只读取参数并计算新 Tensor，不原地
        改动输入或模型状态。并发要求：不能与同一模型上的 optimizer.step 并发执行，
        因为参数可能在计算中途变化；本脚本单线程按批次顺序调用。
        """
        # hidden_values 形状 [B,16]，是输入经过第一层仿射映射后的隐藏特征。
        hidden_values = self.hidden(features)
        # activated 形状仍为 [B,16]，是 ReLU 对隐藏特征逐元素截断后的结果。
        activated = self.activation(hidden_values)
        return self.output(activated)


def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    """在整个 loader 上求按样本数加权的平均损失，不建立反向图。

    参数 model 是待评估的 nn.Module；loader 每次产出 (features, labels) 批次，
    features 每批为 [B,2] 浮点 Tensor，labels 为 [B] int64；loss_fn 接收 [B,3]
    logits 与 [B] 标签，返回批平均零维 Tensor。返回普通 Python float。
    状态与并发：本函数调用 model.eval()，会切换模型及子模块共享的 training 标志；
    不应与同一模型的训练或其他模式切换并发执行。torch.no_grad() 只关闭梯度记录，
    不会锁定模型，也不会防止其他线程改动参数。
    """
    model.eval()
    # loss_sum 累加各批“平均 loss × 样本数”，从而可正确处理大小不同的尾批。
    loss_sum = 0.0
    # sample_sum 记录已经计入 loss_sum 的样本总数，最终作为加权平均的分母。
    sample_sum = 0
    with torch.no_grad():
        # features 每批形状 [B,2]；labels 每批形状 [B] 且元素是 int64 类别索引。
        for features, labels in loader:
            # forward：模型把一批输入映射为每个类别的 logits。
            # logits 形状 [B,3]，每行保存一个样本对三个类别的未归一化分数。
            logits = model(features)
            # loss：CrossEntropyLoss 的默认 reduction='mean' 给出本批平均损失。
            # batch_loss 是当前 B 个样本交叉熵的平均值，形状为零维标量 Tensor。
            batch_loss = loss_fn(logits, labels)
            # count 是当前批样本数 B；用它把批平均还原为该批损失总和。
            count = labels.shape[0]
            # item() 将单元素 Tensor 变成 Python 数值，便于累加与打印，不再保留计算图。
            loss_sum += batch_loss.item() * count
            # 更新已纳入统计的样本数，供函数末尾计算整个数据集的加权均值。
            sample_sum += count
    return loss_sum / sample_sum


def main() -> None:
    """无参数地顺序运行造数、训练、断言和保存/加载验证，不返回数据。

    本函数在当前线程依次驱动 DataLoader；没有启动 Python worker 进程或用户线程。
    若脚本所在进程另有代码并发修改全局随机数状态或模型参数，固定种子与数值复现条件
    不再成立。本文件作为脚本运行时由底部入口调用。
    """
    # 设置模型初始化所用的全局随机流；数据函数另用自己的 generator。
    torch.manual_seed(SEED)
    # features [384,2] 保存浮点输入点；labels [384] 保存对应 int64 类别索引。
    features, labels = make_dataset(SEED)
    # dataset 按首轴配对 features[i] 与 labels[i]，供 DataLoader 按索引取样。
    dataset = TensorDataset(features, labels)

    # DataLoader 把 Dataset 包装成可迭代批次；专用 generator 固定 shuffle 次序。
    # shuffle_generator 控制训练批次的随机排列；使用不同种子流避免与造数耦合。
    shuffle_generator = torch.Generator().manual_seed(SEED + 1)
    # train_loader 每批 48 条并在每个 epoch 打乱顺序，数据本身不复制成新数据集。
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_generator
    )
    # eval_loader 顺序遍历同一训练集；评估结果不受批次打乱影响。
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model 持有两层 Linear 与它们的可训练参数，作为 forward 和 optimizer 的共同状态。
    model = SmallMLP()
    # loss_fn 把 [B,3] logits 与 [B] 整数标签转为批平均交叉熵。
    loss_fn = nn.CrossEntropyLoss()
    # optimizer 持有模型参数引用；每次 step 按 SGD 规则用参数 .grad 改写参数值。
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # initial_loss 是训练前模型在全体样本上的平均交叉熵，作为相对下降基线。
    initial_loss = evaluate_loss(model, eval_loader, loss_fn)
    print(f"seed={SEED} torch={torch.__version__} device=cpu")
    print(f"samples={len(dataset)} batch_size={BATCH_SIZE} epochs={EPOCHS}")
    print(f"initial_loss={initial_loss:.6f}")

    # epoch 从 0 到 EPOCHS-1；循环每轮完整消费一次 train_loader。
    for epoch in range(EPOCHS):
        model.train()
        # epoch_loss_sum 累加本轮各批的损失总和（批均值乘批大小）。
        epoch_loss_sum = 0.0
        # epoch_sample_sum 统计本轮处理过的样本数，用于得到样本加权平均。
        epoch_sample_sum = 0

        # batch_features 是当前输入 [B,2]，batch_labels 是匹配的 [B] int64 标签。
        for batch_features, batch_labels in train_loader:
            # 清除上一批留在叶子参数 .grad 中的梯度；backward 默认做加法累积。
            optimizer.zero_grad(set_to_none=True)

            # forward：得到形状 [本批样本数, 类别数] 的原始类别分数。
            # logits [B,3] 是当前参数对本批样本的前向输出，后续 loss 依赖此计算图。
            logits = model(batch_features)

            # loss：把 logits 与整数类别标签合成一个标量平均交叉熵。
            # loss 是供 backward 起算的零维标量；默认对本批 B 条样本取平均。
            loss = loss_fn(logits, batch_labels)

            # backward：沿本批计算图应用链式法则，把 dL/d参数写入各参数的 .grad。
            loss.backward()

            # step：SGD 按 parameter -= learning_rate * parameter.grad 更新权重和偏置。
            optimizer.step()

            # 统计本批更新前算出的 loss；乘批大小以便 epoch 末按样本数求平均。
            # count 是本批样本数；将更新前算出的批平均损失换算成损失总和。
            count = batch_labels.shape[0]
            epoch_loss_sum += loss.item() * count
            epoch_sample_sum += count

        if epoch in (0, 9, 39, EPOCHS - 1):
            print(f"epoch={epoch + 1:02d} train_loss={epoch_loss_sum / epoch_sample_sum:.6f}")

    # final_loss 是训练后同一全体样本的加权平均损失，用来检查预先设定的绝对目标。
    final_loss = evaluate_loss(model, eval_loader, loss_fn)
    # ratio 是无单位的最终/初始损失比例；不超过 0.10 表示相对初始值至少下降 90%。
    ratio = final_loss / initial_loss
    print(f"final_loss={final_loss:.6f} ratio={ratio:.4f}")
    assert final_loss < MAX_FINAL_LOSS, (
        f"final loss {final_loss:.6f} did not reach {MAX_FINAL_LOSS}"
    )
    assert ratio <= MAX_FINAL_TO_INITIAL_RATIO, (
        f"loss ratio {ratio:.4f} did not reach {MAX_FINAL_TO_INITIAL_RATIO}"
    )

    # state_dict 包含参数张量；只保存权重状态时，需要先用同结构构造模型再加载。
    model.eval()
    with torch.no_grad():
        # logits_before [384,3] 是保存前 eval 模型在完整固定输入上的基准输出。
        logits_before = model(features)

    # temp_dir 是临时目录路径；上下文结束时自动删除目录和其中的 checkpoint 文件。
    with tempfile.TemporaryDirectory(prefix="t07-mlp-") as temp_dir:
        # checkpoint_path 指向临时权重文件，不会把实验产物留在仓库中。
        checkpoint_path = Path(temp_dir) / "model_state.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # reloaded_model 是新建的同结构网络；它的随机初值随后由 checkpoint 中权重覆盖。
        reloaded_model = SmallMLP()
        # weights_only=True 将反序列化限制在权重等安全常见类型；map_location 保证载入 CPU。
        # state 是参数名到 Tensor 的字典；加载到 CPU，且只允许权重类常见对象。
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded_model.load_state_dict(state)
        reloaded_model.eval()

        with torch.no_grad():
            # logits_after [384,3] 是重载模型在相同输入、eval 模式下的输出。
            logits_after = reloaded_model(features)

    # 同一输入、同一 eval 模式与同一权重应给出相同 logits；断言直接检查最大差为零。
    torch.testing.assert_close(logits_before, logits_after, rtol=0.0, atol=0.0)
    # max_logit_error 是两个 [384,3] 输出逐元素绝对差的最大值，转成 Python float 打印。
    max_logit_error = (logits_before - logits_after).abs().max().item()
    print(f"save_load_max_logit_error={max_logit_error:.1f}")
    print("T07 checks passed")


if __name__ == "__main__":
    main()
