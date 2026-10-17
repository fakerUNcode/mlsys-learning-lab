"""B05/T14：把 T13 的“编号→分数”模型接成“训练→逐词元生成”的完整实验。

读法：先读同目录 README 的“研究问题”和“基础应用”，再按本文件的阶段顺序读。
每个可执行语句旁都解释语法、数据变化或该步的理由；括号闭合不单独产生运算。

入口 main → train_toy_model → _train_toy_model_single_thread：
两条完整整数序列 [2,5] → 右移输入/标签 [2,4] → T13 logits [2,4,6]
→ 8 个分类问题的平均交叉熵 → 梯度 → 参数更新 → 更新后复测。
随后 main → generate → choose_next_token：
前缀 [1,P] → 分数 [1,P,6] → 最后位置 [6] → 一个新编号 → 追加并重复。

本文件持有数据与训练/生成流程；Decoder（解码器）内部运算由 T13 实现。
本地默认运行在 CPU（Central Processing Unit，中央处理器）、float32（32 位浮点数）；
没有真实分词器、变长填充、模型保存或 KV Cache（Key-Value Cache，键值缓存）。
没有填充屏蔽不等于没有因果屏蔽：T13 内部仍禁止当前位置读取未来内容。

并发约束：训练重置进程随机状态并临时更改 CPU 线程数，不能与依赖这些状态的任务并发。
生成临时改变传入模型的 training 属性，也不能与同一模型的训练并发。
建议从仓库根目录按 README 命令运行；模块导入实际通过 __file__ 定位，不依赖当前目录。
"""

# Python 语法：把类型注解延迟处理；注解用于说明接口，不会自动检查张量形状。
from __future__ import annotations

# 标准库 json 把列表/字典转成文本，便于保存每一步的数值证据。
import json
# 标准库 sys 提供模块搜索路径 sys.path；这里用来找到已经写好的 T13 模型。
import sys
# Path 是 pathlib 库的路径类；用 / 拼路径，避免手写路径分隔符。
from pathlib import Path
# Any 是类型提示：“此位置可以放不同类型的值”；轨迹字典同时装整数、列表和字符串。
from typing import Any

# PyTorch 主库提供张量（Tensor，多维数组）、自动微分和优化器。
import torch
# as F 是导入别名；F.cross_entropy/F.softmax 是无需先构造层对象的函数。
from torch.nn import functional as F

# 阶段一：导入模型。下划线开头只是“内部使用”的命名约定，不是访问权限控制。
# __file__ 是当前文件名；resolve() 转绝对路径；parent 取文件所在目录。
_HERE = Path(__file__).resolve().parent
# 向上一层到 B05 目录。
_B05 = _HERE.parent
# 再向上一层到 batches，用 Path 的 / 运算符拼出兄弟 B04 目录。
_B04 = _B05.parent / "b04-decoder-block"
# T13 文件夹内的 decoder.py 定义 TinyDecoderLM，不在此复制模型实现。
_T13 = _B04 / "t13-small-decoder-forward"
# not in 判断路径尚未注册；str 将 Path 对象变成搜索列表需要的字符串。
if str(_T13) not in sys.path:
    # 插到索引 0，使本地 decoder.py 优先被找到；这会改变进程的模块搜索路径。
    sys.path.insert(0, str(_T13))
# 导入类尚不创建模型；类实例是在训练函数中创建的。
from decoder import TinyDecoderLM  # noqa: E402：路径准备后才导入，主动豁免导入位置检查。

# 阶段二：实验配置。大写名字约定为常量，Python 本身不禁止重新赋值。
# 随机种子用于同一环境下重放初始化与抽样，不承诺跨硬件/版本逐位一致。
SEED = 1415
# Vocabulary（词表）有 6 个候选，所以每个位置必须输出 6 个分数。
VOCAB_SIZE = 6
# Beginning of Sequence（序列开始标记）A：编号 0 指示接下来走 A→B→C。
BOS_A_ID = 0
# Beginning of Sequence（序列开始标记）C：编号 1 指示接下来走 C→B→A。
BOS_C_ID = 1
# End of Sequence（序列结束标记）编号 2；既作为训练目标，也作为生成停止条件。
EOS_ID = 2
# 每个位置的特征向量宽度 D=8；整数编号本身没有 8 个特征，嵌入查表后才有。
MODEL_WIDTH = 8
# Query（查询）头数为 4，故每头宽度 8//4=2。
QUERY_HEADS = 4
# Key/Value（键/值）头数为 2；4 个查询头分组共享这 2 组键和值。
KV_HEADS = 2
# Feed-Forward Network（前馈网络）的内部宽度为 12，再投影回宽度 8。
FFN_WIDTH = 12
# 学习率控制优化器更新尺度；不是每个参数每轮固定减去 0.02。
LEARNING_RATE = 0.02
# 最多做 500 次更新，防止不满足实验契约时无限训练。
MAX_STEPS = 500
# 最终平均交叉熵必须严格小于 0.01；main 还会另外检查每个标签都命中。
LOSS_TARGET = 0.01


# def 定义函数；无参数；-> torch.Tensor 说明返回一个 PyTorch 张量。
def fixed_sequences() -> torch.Tensor:
    """创建新 CPU long 张量 [2,5]，每行是一条有顺序的完整词元编号序列。

    无外部输入；返回值由调用者持有；每次调用新分配数据，不共享可变训练数据。
    0/1 是不同起始编号，2 是结束编号，3/4/5 代表 A/B/C，无填充词元。
    本函数无全局状态写入；本实验在单线程调用它。
    """
    # torch.tensor 将嵌套 Python 列表复制为张量；外层两行构成批次轴。
    return torch.tensor(
        [  # 开始保存两条样本的外层列表；不是把它们接成一条长序列。
            [BOS_A_ID, 3, 4, 5, EOS_ID],  # 第 0 行：[0,3,4,5,2]。
            [BOS_C_ID, 5, 4, 3, EOS_ID],  # 第 1 行：[1,5,4,3,2]。
        ],
        dtype=torch.long,  # 指定 64 位整数；这些值用于嵌入查表与类别索引，不求梯度。
    )


# 多行参数表与单行 def 含义相同；末尾逗号便于今后逐项增删。
def shift_for_next_token(
    sequences: torch.Tensor,  # 完整序列，形状 [B,S]；B 是行数，S 是完整长度。
) -> tuple[torch.Tensor, torch.Tensor]:  # tuple 表示按固定顺序返回两个张量。
    """把完整 [B,S] 序列配成输入与下一个词元标签，各为 [B,S-1]。

    返回顺序为 inputs、targets；仅切片，不修改输入、不保存跨调用状态。
    切片为共享原存储的视图；调用者须让序列在使用期间有效，且不要就地改写它。
    Python/PyTorch 对象引用会保持底层存储有效；局部变量退出不会让视图悬空。
    并发只读可共享输入，但不得同时写入共享存储。
    """
    # dtype 是元素类型；ndim 是轴数；or 表示任一条件不满足就拒绝输入。
    if sequences.dtype != torch.long or sequences.ndim != 2:
        # raise 立即中止当前路径；这里没有尝试把浮点编号偷偷转换为整数。
        raise TypeError("sequences must be a rank-2 torch.long tensor")
    # shape[1] 是第二轴长度；至少两项才能有一对“当前→下一项”。
    if sequences.shape[1] < 2:
        # ValueError 表示类型可以，但长度不符合这个任务的数据契约。
        raise ValueError("each sequence must contain at least two tokens")
    # 第一个 : 取所有行；:-1 排除最后一列。第 0 行得到 [0,3,4,5]。
    inputs = sequences[:, :-1]
    # 1: 排除第一列。第 0 行得到 [3,4,5,2]；位置 t 的目标来自原位置 t+1。
    targets = sequences[:, 1:]
    # 逗号把两个对象打包为元组；调用方可用 inputs, targets = ... 解包。
    return inputs, targets


# 此内部函数由线程包装函数调用；返回模型、输入、标签和标量损失列表。
def _train_toy_model_single_thread() -> tuple[
    TinyDecoderLM, torch.Tensor, torch.Tensor, list[float]  # 依次为模型、输入、标签、损失列表。
]:
    """在约定的默认 CPU/float32 环境训练全新的模型；调用者先限制 CPU 线程为 1。

    无显式参数；配置来自上方常量。训练输入 [2,4]，目标 [2,4]，分数 [2,4,6]。
    losses 含每次更新前的损失，末项再追加更新后的独立复测损失。
    优化器只在本函数中使用；返回模型持有更新后的参数，优化器状态不随返回值保留。
    此函数重置进程级随机状态且不恢复；不能与依赖同一随机流的任务并发调用。
    """
    # 重置随机流后再构造模型，使本实验的随机初始参数在相同环境中可重放。
    torch.manual_seed(SEED)
    # 新分配完整数据 [2,5]；它在训练过程中保持不变。
    sequences = fixed_sequences()
    # 元组解包：第一个返回对象绑定 inputs，第二个绑定 targets，各为 [2,4]。
    inputs, targets = shift_for_next_token(sequences)
    # 调用类创建实例；这时才分配并随机初始化本例的 600 个可训练数值。
    model = TinyDecoderLM(
        VOCAB_SIZE,   # 6：最终词表轴宽度；与标签编号范围 0..5 一致。
        MODEL_WIDTH,  # 8：词嵌入与残差通路的特征宽度。
        QUERY_HEADS,  # 4：查询头数；必须整除特征宽度。
        KV_HEADS,     # 2：键值头数；必须整除查询头数。
        FFN_WIDTH,    # 12：前馈子层内部宽度。
    )
    # AdamW（Adaptive Moment Estimation with Decoupled Weight Decay，解耦权重衰减的
    # 自适应矩估计）保存参数引用、梯度历史的一阶/二阶矩；它不复制出另一个独立模型。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0  # 遍历注册参数；关闭权重衰减。
    )
    # [] 新建空列表；list[float] 是提示。仅存 Python 数值，避免保留每轮计算图。
    losses: list[float] = []
    # 标记训练模式；不执行训练、不自动计算梯度；本模型无随机失活或批次归一化层。
    model.train()
    # range 上界不包含在内，产生 1..500；_step 表示此轮计数未在循环体中使用。
    for _step in range(1, MAX_STEPS + 1):
        # PyTorch 默认累加参数 .grad；先设 None，避免上一轮梯度混进当前轮。
        optimizer.zero_grad(set_to_none=True)
        # 实例调用经 nn.Module 分派到 forward；返回分数 [2,4,6] 和形状字典。
        # _shape_trace 仍接收第二个返回值；下划线只表示这次不使用它。
        logits, _shape_trace = model(inputs)
        # 把 8 个位置看成 8 个六分类问题；默认 reduction='mean' 得到单个标量。
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)  # -1 自动推算为 8；两边顺序一致。
        )
        # detach 断开梯度关系但通常共享存储；float 把单元素张量变成 Python 浮点数。
        # append 只把这个数放入列表，不保存 logits 或反向图；这是“更新前”的损失。
        losses.append(float(loss.detach()))
        # 由标量损失沿本轮图反向应用链式法则，写入各参数 .grad；此步尚未更新参数。
        loss.backward()
        # 优化器读取梯度与自身状态，就地改变同一个 model 的参数；下轮前向用新值。
        optimizer.step()
        # [-1] 取列表末项；检查的是本轮更新前的损失，退出后还必须重测最终模型。
        if losses[-1] < LOSS_TARGET:
            # break 只离开 for 循环，后面的最终评估仍继续执行。
            break
    # 阶段四：最终重测。eval 切换模块模式，不会自动禁止梯度记录。
    model.eval()
    # with 在块内关闭反向图记录；评估无需为 backward 保存中间数据。
    with torch.inference_mode():
        # 用最后一次 step 后的参数重新前向；_ 表示这次不使用形状字典。
        final_logits, _ = model(inputs)
        # 多行括号只延长表达式；外层 float 取出复测损失的 Python 数值。
        final_loss = float(
            F.cross_entropy(
                final_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)  # 再对齐 [8,6] 与 [8]。
            )
        )
    # 即使循环耗尽，也以最终重测为准；不达标就拒绝交付“已过拟合”的模型。
    if final_loss >= LOSS_TARGET:
        # f 前缀允许插入变量；:.8f 将损失格式化为小数点后 8 位。
        raise RuntimeError(
            f"fixed-sample overfit criterion failed: final_loss={final_loss:.8f}"
        )
    # 最后追加的是更新后复测值，所以 len(losses)-1 才是参数更新次数。
    losses.append(final_loss)
    # 返回的是模型对象本身与数据引用，不是保存到磁盘的检查点；模型已处于评估模式。
    return model, inputs, targets, losses


# 对外训练入口；内部 helper 名称以下划线开头，这里负责进程级线程设置的恢复。
def train_toy_model() -> tuple[
    TinyDecoderLM, torch.Tensor, torch.Tensor, list[float]  # 包装函数原样转交内部四元组。
]:
    """返回过拟合损失门槛达标的模型、输入 [2,4]、标签 [2,4] 与损失轨迹。

    每次调用重新初始化，未保存模型文件；8 个标签是否全部命中由 main/测试另行检查。
    临时设 CPU（Central Processing Unit，中央处理器）运算线程数为 1 并在退出时恢复。
    线程数与随机流是进程状态；禁止与依赖这些状态的其他任务并发调用。
    只恢复线程数，不恢复内部 manual_seed 改变的随机流。
    """
    # 保存调用者原先设置，不能假设所有机器原本都用同一线程数。
    previous_threads = torch.get_num_threads()
    # 本实验张量很小，限制为一个 CPU 运算线程，避免大量并行调度掩盖流程。
    torch.set_num_threads(1)
    # try 开始受保护区域；成功 return 或内部异常都会先执行 finally。
    try:
        # 内部函数返回四元组，这里不复制张量，直接交给调用者。
        return _train_toy_model_single_thread()
    # finally 是状态清理，不是“仅报错时执行”；return 也不能跳过它。
    finally:
        # 恢复进入本函数前的线程配置；不留下持久的线程数变化。
        torch.set_num_threads(previous_threads)


# 选词函数只处理“一排候选分数”，与模型结构和完整前缀无关。
def choose_next_token(
    logits: torch.Tensor,  # [V] 浮点分数；本例 V=6，位置 v 对应词元编号 v。
    strategy: str = "greedy",  # 默认选最大分数；传 sampling 时改为概率抽取。
    temperature: float = 1.0,  # 默认温度 1，只在 sampling 分支使用。
    generator: torch.Generator | None = None,  # | 表示“或”；None 表示未提供独立随机流。
) -> int:  # 返回普通 Python 整数，不返回单元素张量。
    """由 [V] 原始 logits（Logit Scores，未归一化分数）选择一个词元编号。

    greedy 做 argmax（Argument Maximization，最大值索引）；sampling 做温度归一化后抽样。
    不改输入分数；抽样会推进传入 generator 或默认随机生成器的状态。
    同一 generator 不应被并发抽样，否则调用顺序会影响可重放性。
    此教学接口不穷尽校验：调用者须提供有限浮点分数；sampling 须提供有限正温度。
    """
    # ndim=1 才是一排候选；[B,P,V] 必须先选出最后位置，不能直接传进来。
    if logits.ndim != 1:
        # 及早报错，防止沿错误维度选出没有词表意义的索引。
        raise ValueError("logits must have shape [V]")
    # == 比较字符串；贪心分支不需要温度，也不消耗随机数。
    if strategy == "greedy":
        # argmax 返回最大分数索引的零维张量（并列取最先出现者）；item 取标量，int 转整数。
        return int(torch.argmax(logits).item())
    # 第二个独立 if：上面若 return 就不会走到这里。
    if strategy == "sampling":
        # 正常有限温度必须 >0，否则除法/分布语义不符合本实验定义。
        if temperature <= 0:
            # 不以 T=0 代替 greedy；两种策略由 strategy 明确区分。
            raise ValueError("temperature must be positive")
        # / 对所有分数做逐元素除法；dim=-1 沿最后的词表轴归一化，输出 [V] 概率。
        probabilities = F.softmax(logits / temperature, dim=-1)
        # multinomial 从类别权重中抽 1 个索引；传 generator 控制使用哪条伪随机流。
        return int(
            torch.multinomial(probabilities, 1, generator=generator).item()  # [1]→单值→Python int。
        )
    # 两个合法分支均已 return，执行到这里说明字符串写错，不静默降级为其他规则。
    raise ValueError("strategy must be 'greedy' or 'sampling'")


# 生成函数只更新局部词元列表，模型权重固定；训练已在调用前完成。
def generate(
    model: TinyDecoderLM,  # 已训练的 CPU 模型；生成时临时切换它的模式。
    prefix: list[int],  # 非空、合法词表编号组成的列表；不在此处理原始文字。
    max_new_tokens: int,  # 最多新增多少项，不包含初始 prefix 的长度。
    strategy: str = "greedy",  # 每一步均使用此选择规则。
    temperature: float = 1.0,  # sampling 专用的概率分散程度参数。
    seed: int = SEED,  # 独立抽样生成器的初始种子；默认使用文件常量。
) -> tuple[list[int], list[dict[str, Any]]]:  # 完整编号列表、逐步证据字典列表。
    """从 prefix 自回归生成；每步完整重算 [1,P] 输入，记录 [6] 末位置分数。

    tokens 是新列表，不修改调用者 prefix；trace 的每条 prefix 也是独立列表快照。
    模型参数只读；training 模式临时修改并在退出前恢复，不与同一模型并发训练。
    局部随机生成器属于本次调用；抽样推进它的状态，调用结束后不再保留。
    没有键值缓存：历史注意力被重复计算；长度 P 的密集注意力含 P×P 对位置关系。
    生成 EOS（End of Sequence，结束标记）后追加并停止，或在新增上限时退出。
    仅“新生成 EOS”触发停止；输入前缀末尾已有 EOS 时不会提前返回。
    未穷尽防御校验；调用者保证合法词表编号、CPU 设备和有效抽样配置。
    """
    # 空列表在布尔条件中为 False；没有上下文时本接口不知道从哪个位置预测。
    if not prefix:
        # 空前缀直接中止，而不是偷偷补入某种起始编号。
        raise ValueError("prefix must be nonempty")
    # 新增数允许等于 0，届时循环执行零次并返回原前缀的副本与空轨迹。
    if max_new_tokens < 0:
        # 负长度没有定义，不能直接交给循环掩盖调用错误。
        raise ValueError("max_new_tokens must be nonnegative")
    # list(prefix) 复制整数列表；后面的 append 不会改调用方的列表。
    tokens = list(prefix)
    # 每条记录是字典；Any 允许其中同时出现整数、列表、字符串。
    trace: list[dict[str, Any]] = []
    # 新建独立 CPU 随机数生成器，再设种子；sampling 不借用训练的全局随机流。
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # training 是布尔属性；保存它，使生成结束后能恢复调用者原本的模式。
    was_training = model.training
    # 切到评估模式，实际禁止求导仍由下面的 inference_mode 完成。
    model.eval()
    # 包住生成区域，使其中出错时也能走恢复模式的 finally。
    try:
        # 不更新参数、不反向传播，因此关闭计算图记录，减少不需要的反向状态。
        with torch.inference_mode():
            # range 从 0 开始，最多执行 max_new_tokens 次；EOS 分支可提早 break。
            for step in range(max_new_tokens):
                # 再复制一次作本步快照；后续 tokens 增长也不会改变已记录的前缀。
                current_prefix = list(tokens)
                # 外面再套 [] 创建样本行轴：列表 [0,3] 变为张量 [[0,3]]，形状 [1,2]。
                input_ids = torch.tensor(
                    [current_prefix], dtype=torch.long  # 整数查表编号；每轮按当前长度新建输入。
                )
                # 输入 [1,P]，输出所有位置的分数 [1,P,6]；P 是本轮前缀长度。
                all_logits, _shape_trace = model(input_ids)
                # 0 取唯一样本，-1 取最后输入位置，余下词表轴 [6]；这排分数预测“接下来”。
                # detach 明确脱离求导，cpu 确保后续 CPU 抽样；同设备时 cpu 可直接返回原对象。
                last_logits = all_logits[0, -1].detach().cpu()
                # 传入一排分数及选择配置；返回编号，不把整个分数向量接回输入。
                next_id = choose_next_token(
                    last_logits, strategy, temperature, generator  # 按函数参数顺序传递四个对象。
                )
                # 必须在 append 新编号之前记录旧前缀，才能核对“给定什么→选了什么”。
                trace.append(
                    {  # 字典把同一步的输入、输出和规则绑定在一起。
                        "step": step,  # 从 0 计数的生成轮次。
                        "prefix": current_prefix,  # 列表快照，不引用仍会增长的 tokens。
                        "logits": [  # 列表推导：对 6 个原分数逐一转 Python 数并舍入。
                            round(float(value), 6)  # 仅影响展示；next_id 已按未舍入分数决定。
                            for value in last_logits  # 按词表编号 0..5 的顺序遍历。
                        ],
                        "next_token": next_id,  # 本步真实选择，不从显示文本反推。
                        "strategy": strategy,  # 标注这一轨迹使用哪种规则。
                    }
                )
                # 自回归反馈：预测编号成为下一次模型输入的一部分；参数没有在这里改变。
                tokens.append(next_id)
                # EOS 要先写进完整结果再终止，结果与训练中的结束标签才有一致语义。
                if next_id == EOS_ID:
                    # 退出最内层 for；with 和 finally 仍按顺序退出。
                    break
    # 无论循环正常结束、break 或抛异常，都执行模式恢复。
    finally:
        # train(False) 等价于切到评估；train(True) 切回训练，不是在这里做参数更新。
        model.train(was_training)
    # tokens 含初始前缀；trace 每个元素只对应一次新词元预测，二者长度一般不相等。
    return tokens, trace


# 主函数无参数、无有效返回值（None）；它调度训练、硬性检查和文本证据输出。
def main() -> None:
    """main → 训练 → 8 个标签核对 → 两条贪心轨迹 → 一条抽样轨迹。

    不接收原始文本；不写模型文件；标准输出可重定向到报告目录。
    硬性成功条件：最终损失 <0.01 且 8 个训练位置 argmax 全等于标签。
    单进程顺序执行；不要在其他任务操作进程级随机/线程状态时并发调用。
    """
    # 解包四个返回值；其中 model 已持有训练后的参数并处于评估模式。
    model, inputs, targets, losses = train_toy_model()
    # 展示也不需要梯度图，关闭记录后再作一次独立前向。
    with torch.inference_mode():
        # 同时接收 [2,4,6] 的分数与内部各阶段形状字典。
        logits, shape_trace = model(inputs)
    # 沿最后一轴 6 个候选取索引，得到 [2,4] 的整数预测，与标签的轴一一对应。
    predictions = logits.argmax(dim=-1)
    # torch.equal 要求形状及所有元素相同；bool 明确得到 Python 布尔值。
    exact_match = bool(torch.equal(predictions, targets))
    # not 对布尔值取反；损失小仍需单独检查每个位置，因为平均值可能掩盖局部错误。
    if not exact_match:
        # 不把部分正确当成“已记住两条固定序列”。
        raise RuntimeError("fixed training labels are not all predicted exactly")
    # 以下 print 都只读结果，不参与训练；f-string 将大括号里的表达式代入字符串。
    print(f"device=cpu dtype=float32 seed={SEED}")
    # 参数迭代器逐个产生参数张量；numel 统计元素数，sum 相加，本例应为 600。
    print(
        f"parameters="  # 相邻字符串自动拼接；此段是输出字段名。
        f"{sum(parameter.numel() for parameter in model.parameters())}"  # 此段代入总数量。
    )
    # tolist 把张量内容转 Python 嵌套列表，打印完整两行，便于核对编号约定。
    print(f"sequences={fixed_sequences().tolist()}")
    # 打印送入模型的两行当前编号。
    print(f"inputs={inputs.tolist()}")
    # 打印向后错一格的标签，不把它们隐藏在文字解码里。
    print(f"targets={targets.tolist()}")
    # tuple 把 torch.Size 转成普通元组，预期显示 (2,4,6)。
    print(f"logits.shape={tuple(logits.shape)}")
    # json.dumps 把形状字典序列化；sort_keys=True 按键排序，使证据文本稳定可比。
    print(f"shape_trace={json.dumps(shape_trace, sort_keys=True)}")
    # losses 最后一项是最终复测；扣去它才是更新次数；:.8f 保留 8 位小数。
    print(
        f"steps={len(losses) - 1} "  # 实际循环更新次数，并非固定的 MAX_STEPS。
        f"initial_loss={losses[0]:.8f} "  # 第一次更新前的值，索引从 0 开始。
        f"final_loss={losses[-1]:.8f}"  # 最后一次更新后重新前向的值。
    )
    # == 逐元素比较得到布尔 [2,4]；float 转为 0/1；mean 对 8 个位置求平均。
    print(
        f"exact_training_token_accuracy="  # 此处统计训练位置准确率，不是未见样本质量。
        f"{float((predictions == targets).float().mean()):.6f}"  # Python 浮点输出六位小数。
    )
    # 与上面的平均准确率互相核对：只有全等才显示 True。
    print(f"all_training_targets_argmax_correct={exact_match}")
    # 元组内是两个独立单元素列表，分别演示两个不同起始编号的贪心生成。
    for prefix in ([BOS_A_ID], [BOS_C_ID]):
        # 最多新增 6 项；正常会在第 4 项生成 EOS 并提前结束。
        generated, trace = generate(
            model, prefix, max_new_tokens=6, strategy="greedy"  # 关键字参数按名字指定。
        )
        # generated 包含初始起始编号，长度正常为 5。
        print(f"greedy_prefix={prefix} generated={generated}")
        # ensure_ascii=False 允许中文原样输出；此处字典值主要是数字和英文策略名。
        print(f"greedy_trace={json.dumps(trace, ensure_ascii=False)}")
    # 再从 A 起始编号按概率抽样；生成函数内部使用独立随机流，默认种子为 1415。
    sampled, sample_trace = generate(
        model,  # 同一个已训练模型，权重在这些生成调用间保持不变。
        [BOS_A_ID],  # 只有起始编号，训练目标没有作为生成输入传入。
        max_new_tokens=6,  # 保护上限：最多新增 6 项。
        strategy="sampling",  # 进入概率抽样分支，而非 argmax。
        temperature=0.8,  # 正温度；只在 sampling 中影响概率。
    )
    # 打印抽样的完整结果；一次结果相同不表示 sampling 恒等于 greedy。
    print(f"sampling_prefix={[BOS_A_ID]} generated={sampled}")
    # 保存每次真实选择与原前缀，允许按种子与规则重放。
    print(f"sampling_trace={json.dumps(sample_trace, ensure_ascii=False)}")


# 直接执行本文件时 __name__ 为 __main__；被 test_t14 导入时不自动训练。
if __name__ == "__main__":
    # 实际调用调度入口；上方 def 只是定义函数，不会在定义时执行训练。
    main()
