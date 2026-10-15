# T13 验证报告

日期：2026-10-14。范围：B04/T13 一个小型 decoder-only（仅解码器）前向、token IDs 到 logits 的 shape 路径、因果隔离与推测解码职责导读。

## 环境

- 仓库：`/home/alibaba/projects/mlsys-learning-lab`；实验产物在未提交工作区运行。
- Python 3.12.3；PyTorch 2.11.0+cu128；实际张量设备 CPU，dtype float64；固定种子 1313。
- CUDA 可用，但本次没有把程序放到 GPU 执行；没有模型权重、分词器或外部服务。
- 本轮小模型配置：`V=17,D=8,Hq=4,Hkv=2,d=2,F=12`，输入 `[1,4]`；每个键值头服务两个查询头。

## 来源范围

| 资料 | 实际范围 | 边界 |
| --- | --- | --- |
| R04《Attention Is All You Need》§3 | 编码器层、解码器层及掩码自注意力的结构背景 | 小实验是 decoder-only 组合，不复现原论文完整 encoder-decoder 或交叉注意力 |
| R05 Llama 文档/实现 | pre-norm（前置归一化）、RoPE（旋转位置嵌入）、GQA（分组查询注意力）、RMSNorm（均方根层归一化）、SwiGLU（Swish 门控线性单元）组合结构；组件复用本仓库 T11/T12 | 不从 Transformers 加载代码、配置或权重；小型参数不对齐完整模型 |
| R13 speculative decoding（推测解码）论文 | Algorithm 1、§2.1–2.3：draft（草稿）提出、target（目标）并行验证、按首拒绝纠正及全接受 bonus token（额外词元） | 仅文档导读；未实现推测采样、双模型或速度测试 |
| R19《深入理解 AI Infra》 | 第 2 章模型计算/状态与第 3 章推理负载、串行依赖相关小节 | 用作结构性解释，未复现书中计算量或负载模型 |

## 复现命令

从仓库根目录运行：

```bash
.venv/bin/python learning/batches/b04-decoder-block/t13-small-decoder-forward/decoder.py
.venv/bin/python learning/batches/b04-decoder-block/t13-small-decoder-forward/test_t13.py
```

原始 stdout 保存于本目录下 `2026-10-14-t13/decoder.txt` 与 `2026-10-14-t13/test_t13.txt`。例子及 5 项检查均退出码 0。

## shape 与程序证据

```text
token_ids       [1,4]
embedding       [1,4,8]
attn_norm       [1,4,8]
Q               [1,4,4,2]
K/V 各自        [1,2,4,2]
attention       [1,4,4,2]
attn residual   [1,4,8]
SwiGLU 内部     [1,4,12] -> [1,4,8]
MLP residual    [1,4,8]
final norm      [1,4,8]
logits          [1,4,17]
```

已观察到参数数目 776；末位置前三个 logits 为 `[0.8020778374681452, 0.09022986700285214, 0.5129989611199002]`。该具体数值仅对应 CPU float64、本程序初始化与种子 1313。

| 检查 | 结果 |
| --- | --- |
| Long token IDs `[B,S]` 到有限 logits `[B,S,V]` | 通过，shape `(1,4) -> (1,4,17)` |
| 主要中间 shape | 通过，覆盖 embedding、Q、KV、attention、残差与 logits |
| 显式按模块阶段调度与模型封装输出一致 | 通过，`rtol=0, atol=0` |
| 修改位置 3 的输入 token，位置 0–2 logits 不变 | 通过，逐元素相同；验证该输入、该权重下的因果隔离 |
| 非 Long IDs 与不整除头配置 | 按预期拒绝 |
| 参数数目及 logits softmax 归一化 | 参数为 776；每位置概率和在 `1e-12` 容差内为 1 |

测试是同一个小型 CPU reference 的编排和语义检查，不是独立模型实现、真实 tokenizer 或真实权重验收。与 B05 的 T14 训练/自回归生成、T15 正式缓存、T16 显存实测分开。

## 第一月与预算

从 B01/T01 的首条阶段 1 日志 2026-09-27 到 T13 实验交付 2026-10-14，相差 17 天，首尾共 18 个日历日；索引中有 7 个阶段 1 学习日期条目（9/27、10/8、10/9、10/11–10/14），T01–T13 均已具备相应完成记录或本次 T13 产物。没有逐任务开始/结束时刻，故没有可验证的净学习小时。日期跨度不等于投入时长。

后续 B05 暂按 T14–T16 每项 2–3 次会话、合计 6–9 次会话预算，并保持 T14→T15→T16 的依赖顺序。此会话数是下一轮计划粒度，不是从未知小时数计算出的速度预测；从 T14 起记录开始/结束、暂停与净分钟，完成一项后以实际净时长调整余下预算，B05 末再校准 B06。

## 限制

没有验证 GPU、float16/bfloat16、反向、训练收敛、真实模型、模型质量、tokenizer、KV cache、自回归采样、speculative decoding 的分布保持实现或性能。没有 benchmark、profiler 或峰值显存数据。因而不能从该实验推断真实生成质量、吞吐或加速比。
