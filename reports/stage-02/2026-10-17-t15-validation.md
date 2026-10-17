# T15 KV 验证

日期：2026-10-17。范围：B05/T15 单层小型 Decoder 的 prefill 与单步 decode，逐步比较 KV Cache logits 和无缓存完整前缀 logits。

## 问题

完整前缀重算会在新 token 到来后再次计算所有历史位置。KV Cache 应保留历史 Key/Value，仅计算当前 token 的 Q/K/V，并让当前 Query 注意全部历史 K/V。本实验检查缓存是否改变 next-token logits、位置、因果可见范围或缓存长度。

## 假设

对固定模型权重和 token 序列，T13 的因果 attention 使一个位置的 K/V 只依赖该位置及其左侧 token。因此 prefill 计算出的历史 K/V 可在后续 token 到来时复用。若位置递增、RoPE 状态、矩形 mask 和缓存追加均正确，缓存路径的当前 logits 应与完整前缀 reference 一致（允许浮点舍入误差）。

## 环境

- 仓库：`/home/alibaba/projects/mlsys-learning-lab`；基线 HEAD：`af757a5`，新增实验在该工作区修改上运行。
- Python 3.12.3；PyTorch 2.11.0+cu128；CUDA 对该环境可用，但张量设备为 CPU。
- CPU、float64、`torch.inference_mode()`、`eval()`；随机种子 1515。
- 模型：`V=17,D=8,Hq=4,Hkv=2,d=2,F=12`，单 Decoder Block；无加载权重、tokenizer 或外部语料。
- 比较容差：`rtol=0, atol=1e-12`，逐元素比较完整词表 logits。

## 来源

| 资料 | 读取范围与时间 | 版本边界 |
| --- | --- | --- |
| R05 [Llama 模型文档](https://huggingface.co/docs/transformers/main/en/model_doc/llama)及 [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | 2026-10-17 核对 Q/K/V 投影、Q/K 位置旋转、past K/V 更新和 attention 调用顺序 | 官方文档与 GitHub `main` 为滚动版本；本地模型复用 T13 小型实现，不导入 Transformers |
| R06 [Cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation)及 [KV Cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache) | 2026-10-17 核对历史缓存、自回归位置与 mask 长度语义 | 官方 Cache API 与本地 tuple reference 不等同；本实验不验证 API 兼容性 |
| 本地 [T12](../../learning/batches/b04-decoder-block/t12-rope-gqa-position/README.md)、[T13](../../learning/batches/b04-decoder-block/t13-small-decoder-forward/README.md)、[T14](../../learning/batches/b05-generation-kv-resources/t14-tiny-lm-training-generation/README.md) | RoPE/GQA/位置、完整小模型前向、逐 token 前缀生成 | 复用本仓库对应 reference；不扩展为真实模型 |

## 方法

`run_t15.py` 固定 token 序列，分别取长度 1、3、5 的 prompt。每个 prompt：

1. 把完整 prompt 一次送入 `forward_with_cache`，保存 K/V/位置，比较所有 prompt 位置 logits 与 T13 完整前向。
2. 连续 16 次每次只输入一个新 token。新位置取 cache 最后位置加一；缓存路径返回单位置 logits 和增长后的 cache。
3. 无缓存路径把截至当前 token 的完整 prefix 输入原始 `TinyDecoderLM.forward`，只取末位置 logits 与缓存结果比较。
4. 逐步记录 cache 长度和最大绝对 logits 误差；测试还克隆旧 cache，验证后续调用不会改写旧对象。

复现命令（仓库根目录）：

```bash
.venv/bin/python learning/batches/b05-generation-kv-resources/t15-kv-cache-reference/run_t15.py > reports/stage-02/2026-10-17-t15/cache_comparison.json
.venv/bin/python learning/batches/b05-generation-kv-resources/t15-kv-cache-reference/test_t15.py
```

原始逐位置 JSON 保存在[实验数据](2026-10-17-t15/cache_comparison.json)，测试 stdout 保存在[test_t15.txt](2026-10-17-t15/test_t15.txt)。

## 结果

| Prompt 长度 | Prefill 后长度 | 第 16 步后长度 | Prefill/Decode 对比数 | 最大绝对误差 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 17 | 17 | `2.220446049250313e-16` |
| 3 | 3 | 19 | 17 | `2.220446049250313e-16` |
| 5 | 5 | 21 | 17 | `2.220446049250313e-16` |

共 51 条逐位置记录（3 次 prefill + 48 个增量步）。全部 `torch.testing.assert_close` 检查通过。cache 长度每次从当前长度增加 1；传入旧 cache 的长度和内容不变。2 项测试通过。

## 解释

- **为什么只缓存 K/V：** 未来每一步的 Query 来自未来 token 的隐藏状态，旧 Query 后续不再使用；历史 Key 提供匹配对象，历史 Value 提供被加权汇总的内容，后续 Query 都需要它们。K 在缓存前已经按自己的绝对位置应用 RoPE，V 不应用 RoPE。缓存没有存注意力输出或完整隐藏状态。
- **为什么带位置的 mask：** 当前 decode 查询长度为 1，但 key 轴包含全部历史位置和当前位置，mask 是 `[B,1,1,N]` 的矩形结构。用普通 `1×1` causal mask 会丢失对历史列的可见性。代码用 `key_position <= query_position` 建 mask。
- **为何 cache 增量是一次：** 缓存函数输入仅为一个新 token，旧 K/V 与新 K/V 在序列轴拼接一次并返回新对象。若每轮传完整 prefix，则旧 token 会再次投影并被重复追加；接口拒绝已有 cache 时输入长度大于 1，并校验新位置紧接缓存末尾。
- **为何 logits 可对齐：** decoder 的因果 mask 保证旧位置不读取未来 token，所以后续 token 不会改变旧位置产生的 K/V。当前 query 读取这些历史 K/V 后，与完整前缀计算等价。观测最大误差约为 float64 机器精度量级，非位级完全相同。

## 限制

- 只验证单层、CPU float64、小词表固定 token 流；没有验证 GPU、float16/bfloat16、反向传播、真实模型权重、长上下文、batch 补齐、并发请求、滑窗、beam branching、cache rollback 或生产 Cache API。
- 每步使用 `torch.cat` 创建增长后的 K/V 张量，是为显式展示状态变化的 reference；未测复制成本、延迟、吞吐或峰值内存，不能据此声称真实加速。
- R05/R06 在线页面为滚动版本，记录的是核对日期和相关语义，未将其描述成固定发行版。

## 产物

- [T15 章节](../../learning/batches/b05-generation-kv-resources/t15-kv-cache-reference/README.md)
- [KV Cache reference](../../learning/batches/b05-generation-kv-resources/t15-kv-cache-reference/kv_cache.py)、[逐步实验脚本](../../learning/batches/b05-generation-kv-resources/t15-kv-cache-reference/run_t15.py)、[测试](../../learning/batches/b05-generation-kv-resources/t15-kv-cache-reference/test_t15.py)
- [51 条逐步原始数据](2026-10-17-t15/cache_comparison.json)、[测试输出](2026-10-17-t15/test_t15.txt)
