"""B05/T14：用结构化断言检查标签、过拟合与逐步生成，不以生成文字观感判断。

位置：与 train_generate.py 同目录；直接执行本文件会运行 5 个 unittest 检查。
生命周期：setUpClass 只训练一次，所有检查复用该模型，不在各测试中更新参数。
并发约束：训练会改进程随机/线程状态；本文件按 unittest 的默认串行方式执行。
覆盖范围只限现有 5 项；README 的覆盖表明确列出未独立验证的分支。
"""

# Python 标准库 unittest 提供测试类、断言、共享准备阶段和命令行运行入口。
import unittest

# PyTorch 提供张量比较、模型前向、损失函数和无需反向图的上下文。
import torch

# 导入同目录实验的接口；导入时不会执行它的 main，因为其 __name__ 不是 __main__。
from train_generate import (
    BOS_A_ID,  # 起始编号 0；用它生成第一条序列。
    EOS_ID,  # 结束编号 2；检查生成确实追加结束符。
    fixed_sequences,  # 返回人工定义的完整正确序列。
    generate,  # 返回完整生成编号和每步证据。
    shift_for_next_token,  # 把完整序列配成当前输入/下一项标签。
    train_toy_model,  # 重新初始化并拟合整个固定批次。
)


# 继承 TestCase 后，unittest 会收集名称以 test_ 开头的方法；self 指当前测试实例。
class T14TinyLMTests(unittest.TestCase):
    # 装饰器把准备函数设为类方法；cls 指测试类，类属性可让所有测试共享模型。
    @classmethod
    # setUpClass 是框架约定名，在这个类的第一项测试前执行一次，不是每次都训练。
    def setUpClass(cls) -> None:
        # 先记住训练前的进程 CPU 线程数，后面不能只检查“等于 1”。
        cls.threads_before_training = torch.get_num_threads()
        # 四元组解包后保存在类上；只训练一次，测试方法均不更新模型权重。
        cls.model, cls.inputs, cls.targets, cls.losses = train_toy_model()
        # 训练返回后马上读取设置，验证 try/finally 的正常返回路径确实恢复状态。
        cls.threads_after_training = torch.get_num_threads()

    # 测试一：比较进入训练前后线程设置；此项没有模拟训练异常。
    def test_training_restores_process_thread_setting(self) -> None:
        # assertEqual 不相等就让该测试失败；不能只靠注释声称恢复成功。
        self.assertEqual(self.threads_after_training, self.threads_before_training)

    # 测试二：既检查轴长度，也用具体编号防止错位方向/切片轴错误。
    def test_labels_are_shifted_one_token_to_the_right(self) -> None:
        # 每次新建 [2,5] 数据，不修改训练期间共享的数据。
        sequences = fixed_sequences()
        # 返回输入与目标两个 [2,4] 切片视图。
        inputs, targets = shift_for_next_token(sequences)
        # 两行各保留前四列，输入形状应为 [2,4]。
        self.assertEqual(inputs.shape, (2, 4))
        # 标签也必须恰好有 8 项，能和 8 排词表分数对应。
        self.assertEqual(targets.shape, (2, 4))
        # 张量专用比较检查实际输入数据与预期切片一致。
        torch.testing.assert_close(inputs, sequences[:, :-1])
        # 标签与后四列一致，而不是与输入自身一致。
        torch.testing.assert_close(targets, sequences[:, 1:])
        # 索引 [0] 取首行，tolist 转列表；用人工写明的编号再核对一次。
        self.assertEqual(inputs[0].tolist(), [0, 3, 4, 5])
        # 首行最后一个目标是 EOS；模型才会学到什么时候结束。
        self.assertEqual(targets[0].tolist(), [3, 4, 5, EOS_ID])

    # 测试三：独立前向复测损失和所有位置，不只相信训练函数返回了模型。
    def test_model_overfits_every_fixed_next_token_target(self) -> None:
        # 此处不需要 backward，所以不记录反向图；模型已由训练函数切到 eval。
        with torch.inference_mode():
            # _ 接收但不使用形状字典；分数形状为 [2,4,6]。
            logits, _ = self.model(self.inputs)
            # 将输入位置合成 8 个六分类问题，再算一次默认平均交叉熵。
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), self.targets.reshape(-1)  # [8,6] 对 [8]。
            )
        # assertLess 检查严格小于；float 取零维张量中的数值。
        self.assertLess(float(loss), 0.01)
        # argmax(-1) 对每排 6 分取索引，得到 [2,4]；equal 要求每个位置全对。
        self.assertTrue(torch.equal(logits.argmax(-1), self.targets))
        # 还要求最终损失低于初始损失，确认不是把同一个初始值当训练结果。
        self.assertLess(self.losses[-1], self.losses[0])

    # 测试四：核对首条贪心路径、各步 prefix/logits/选择与 EOS 停止。
    def test_greedy_generation_records_replayable_prefix_and_logits(self) -> None:
        # 最多新增 6 项；应该 4 项就生成 EOS，不能再继续到上限。
        generated, trace = generate(self.model, [BOS_A_ID], 6, strategy="greedy")
        # 与完整固定序列逐编号比较，而不是比较文字是否“像句子”。
        self.assertEqual(generated, fixed_sequences()[0].tolist())
        # 一次预测记录一条；4 条说明 EOS 后没有额外生成。
        self.assertEqual(len(trace), 4)
        # enumerate 同时给出轮次索引和该轮字典；不是每步重新训练。
        for index, record in enumerate(trace):
            # 轨迹轮次应从 0 连续计数。
            self.assertEqual(record["step"], index)
            # 切片右端不包含：第 index 步前缀应包含起始项和之前已生成的 index 项。
            self.assertEqual(record["prefix"], generated[: index + 1])
            # 必须记录全部 6 个候选分数，只有选中分数无法核对 argmax。
            self.assertEqual(len(record["logits"]), 6)
            # max 遍历编号 0..5；key=lambda 指按对应分数比较，返回分数最大的编号。
            self.assertEqual(record["next_token"], max(
                range(6), key=lambda token_id: record["logits"][token_id]  # lambda 是匿名的小函数。
            ))
        # 最后一项应为 EOS，且保留在返回序列中。
        self.assertEqual(generated[-1], EOS_ID)

    # 测试五：独立生成器同种子重放；不要求抽样序列必然等于贪心序列。
    def test_sampling_is_reproducible_for_a_fixed_seed(self) -> None:
        # 第一次生成：同一模型、前缀、上限、温度和种子 9。
        first = generate(self.model, [BOS_A_ID], 6, "sampling", 0.8, seed=9)
        # 第二次调用会新建生成器并重新设种子，不承接第一次调用结束的随机状态。
        second = generate(self.model, [BOS_A_ID], 6, "sampling", 0.8, seed=9)
        # 返回的是 (生成列表, 轨迹列表)，assertEqual 递归比较其中全部普通 Python 值。
        self.assertEqual(first, second)
        # first[1] 是轨迹；all 检查每条记录都标明 sampling，而不是误走 greedy 分支。
        self.assertTrue(all(step["strategy"] == "sampling" for step in first[1]))


# 直接执行时运行测试；被其他文件导入时仅定义测试类。
if __name__ == "__main__":
    # verbosity=2 逐项打印测试名和结果，失败时保留可定位的堆栈。
    unittest.main(verbosity=2)
