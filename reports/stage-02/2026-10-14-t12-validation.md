# T12 验证报告

日期：2026-10-14。范围：B04/T12 默认旋转位置嵌入、分组查询注意力与位置语义。

## 环境

- 仓库：`/home/alibaba/projects/mlsys-learning-lab`；起始 HEAD 为 `8b7d8cab4081e24e72c2bcce70af80ac2dae9442`；本次新增产物在未提交工作区验证。
- CPU：Intel Core Ultra 9 275HX；Python 3.12.3；PyTorch 2.11.0+cu128；执行器 `.venv/bin/python`。
- 数值对照：CPU float64；头编号标记示例使用 float32。种子：位置实验 1212、独立逐头核对 1202；局部随机数生成器不修改全局随机状态。
- 没有运行 GPU 或模型权重；不涉及驱动、编译器、预热和吞吐统计。

## 来源

| 来源 | 版本与实际核对范围 |
| --- | --- |
| R05 [Llama 源码](https://github.com/huggingface/transformers/blob/22cfdfaad99038416b56d21e35b0d32d4f4ee2d3/src/transformers/models/llama/modeling_llama.py) | commit `22cfdfaad99038416b56d21e35b0d32d4f4ee2d3`；默认 rotary embedding（旋转位置嵌入）、前后半区旋转、KV head（键值头）扩展、显式注意力、投影 / 旋转 / 缓存顺序与默认位置 |
| R05 [配置类](https://github.com/huggingface/transformers/blob/22cfdfaad99038416b56d21e35b0d32d4f4ee2d3/src/transformers/models/llama/configuration_llama.py) | 同 commit；head_dim（每头宽度）和键值头默认初始化、`rope_parameters` 字段 |
| R06 [缓存说明](https://github.com/huggingface/transformers/blob/22cfdfaad99038416b56d21e35b0d32d4f4ee2d3/docs/source/en/cache_explanation.md) | 同 commit；历史与新 K/V 拼接、当前查询与总键长度、二维有效位置表覆盖历史与当前 |
| PyTorch [2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html) | 实际环境版本；布尔 True=允许、矩形因果约束左上对齐、显式关闭随机丢弃 |
| [TinyLlama 配置](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json) | 发布方仓库 revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`；只读配置，没有运行模型 |

全部来源于 2026-10-14 核对。TinyLlama 的 main 首先通过发布方 API 解析到 revision，再从不可变 revision 获取配置。本地[配置快照](../../learning/batches/b04-decoder-block/t12-rope-gqa-position/sources/tinyllama-config.json)与[来源及散列](../../learning/batches/b04-decoder-block/t12-rope-gqa-position/sources/model-source.json)可离线复查。浏览工具打开 TinyLlama 的 main 地址失败，实际配置通过 Python 标准库访问发布方 API 与固定 resolve URL 获取成功。

## 复现命令

从仓库根目录执行：

```bash
.venv/bin/python learning/batches/b04-decoder-block/t12-rope-gqa-position/rope.py
.venv/bin/python learning/batches/b04-decoder-block/t12-rope-gqa-position/gqa.py
.venv/bin/python learning/batches/b04-decoder-block/t12-rope-gqa-position/position_demo.py
.venv/bin/python learning/batches/b04-decoder-block/t12-rope-gqa-position/config_example.py
.venv/bin/python learning/batches/b04-decoder-block/t12-rope-gqa-position/test_t12.py
```

四个独立脚本均返回 0；修正首轮断言后，测试脚本返回 0，12 项通过。

另已检查本章及同步入口中的 113 个本地 Markdown 链接、5 个 Python 文件的语法、固定配置的 SHA-256 散列与 40 位 revision；均通过。`git diff --check` 通过；新增文件另行检查行尾空格。

## 数值结果

| 验收项 | 实际证据 | 结果 |
| --- | --- | --- |
| 区分查询头与键值头 | `gqa.py` 标记值；独立逐 batch / 查询头循环核对 | 4 个查询头对应 2 个键值头，映射 `[0,0,1,1]`；MHA/MQA 边界通过 |
| 手算二维旋转 | Python math 标量公式独立对照 | `[3,4]` 在 1 弧度后为 `[-1.7449770216271667,4.685622177896248]`；平方和保持 25 |
| 默认频率与配对布局 | d=4 独立标量公式 | 配 `(x0,x2)` / `(x1,x3)`；位置 1 的角度 1 / 0.01 |
| 长度与相对点积 | 非连续输入、零位置、位置同时平移 | 在 `rtol=1e-12,atol=1e-12` 下通过 |
| 整段与增量位置一致 | 五组输入、Q/K 旋转与注意力输出逐项对照 | 所有记录的最大绝对误差均为 0 |
| 独立接口对照 | 显式计算 vs SDPA（Scaled Dot Product Attention，缩放点积注意力） | 五组最大误差 `4.996003610813204e-16`，低于 `1e-12` |
| 错误位置反例 | 每块重置位置、保持正确可见关系 | 默认输入输出差 `1.1602167808757586`，被测试检测 |
| 错误矩形掩码 | 单查询位置 6、7 个键，全部匹配分数为 0 | 正确平均约 `[3,3]`；左上三角仅读键 0，输出 `[0,0]` |
| 实际配置对应 | 本地固定 TinyLlama 配置 | Hq=32、Hkv=4、d=64、每组 8 个查询头；theta=10000、无缩放、配置位置范围 2048 |
| 非法输入边界 | 奇数头宽、非整数位置、负 / 重复 / 逆序位置、头数不能整除、全屏蔽行、错误掩码类型与 KV 形状 | 明确报错，测试通过 |

误差阈值 `1e-12` 用于小型 float64 输入及不同计算顺序的舍入误差；不代表低精度容差。代码使用输入精度计算角度与归一化；引用的 Llama 实现使用 float32 角度与 softmax 中间值，故不声称逐位复现官方实现。

### 输入覆盖

| B | S | Hq / Hkv | d | 起点 | 块长度 | 整段 / 分块误差 | SDPA 误差 |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 1 | 1 | 1 / 1 | 2 | 0 | `(1,)` | 0 | 0 |
| 1 | 5 | 4 / 2 | 4 | 0 | `(1,1,1,1,1)` | 0 | `3.3306690738754696e-16` |
| 2 | 7 | 4 / 2 | 4 | 0 | `(3,1,3)` | 0 | `3.3306690738754696e-16` |
| 2 | 7 | 4 / 1 | 8 | 11 | `(3,1,1,1,1)` | 0 | `4.996003610813204e-16` |
| 1 | 5 | 2 / 2 | 4 | 5 | `(2,3)` | 0 | `2.220446049250313e-16` |

“误差 0”是本轮输入上的实际记录，不能推广到任意后端、数据精度或规模。

## 原始证据

- [二维旋转输出](2026-10-14-t12/rope.txt)、[头映射输出](2026-10-14-t12/gqa.txt)。
- [默认位置实验](2026-10-14-t12/position_demo.txt)、[五组输入与误差](2026-10-14-t12/cases.json)。
- [真实配置输出](2026-10-14-t12/config_example.txt)、[最终 12 项测试日志](2026-10-14-t12/test_t12.txt)。
- [首轮断言失败](2026-10-14-t12/first-test-failure.txt)：正确平均实际为 `2.9999999999999996`，原断言要求严格等于 `3.0`；将该浮点断言改为绝对容差 `1e-12`，再次运行全部 12 项通过。计算逻辑无需修改。

## 已知限制

本章验证同一组投影后 Query/Key/Value（查询/键/值）的局部注意力层，不验证模型 logits（词表未归一化分数）、完整生成、反向梯度或模型质量。没有实际加载 Transformers 模型或测试配置兼容层；checkpoint 字段版本与引用源码版本明确分开。

没有测试 GPU、float16/bfloat16、动态频率缩放、部分特征旋转、补齐、滑动窗口、峰值内存或性能。缓存是局部张量，未形成 T15 的正式缓存实现。头扩展可能产生临时复制，原始键值头较少只支持元素数推算，不能作为实测优化收益。

T12 的四项验收要求均已有代码、笔记、实验与复现证据；下一任务为 T13。
