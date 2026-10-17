# T16 显存预算验证

日期：2026-10-17。范围：模型权重、target/draft K/V、prefill 与 verifier reference 前向峰值、PyTorch allocator 的 allocated/reserved 观测，以及指定假设配置的设备容量判断。

## 问题

只按参数量估算模型常驻空间会漏掉两套模型的 K/V 状态、验证时的临时激活、运行时开销和安全余量。本实验将公式应用于 T13 toy target/draft，在当前 GPU 实测分配；再用显式假设的 7B+1B 双模型配置计算容量表。大模型配置只用于预算推演，没有下载或加载相应权重。

## 环境

- Python 3.12.3；PyTorch 2.11.0+cu128；CUDA Runtime 12.8。
- NVIDIA GeForce RTX 5080 Laptop GPU；设备 total `17,094,475,776 B`（15.920 GiB），实验开始时 free `15,685,648,384 B`（14.608 GiB）。共享 GPU 的 free 在实验运行期间会变化。
- CUDA 实测配置：T13 target `776` 参数、T13 变体 draft `744` 参数；均单层 float32、batch 2、cache 长度 128；target prefill 长度 128，full-prefix verifier reference 长度 133。
- 容量表假设：target 7B/32 层/8 KV heads/128 head dim；draft 1B/24 层/8 KV heads/64 head dim；参数与 KV 为 BF16，context 4096，额外验证位置 5。
- 脚本运行时设 seed 1616；模型为随机初始化权重，内存结论不依赖权重数值。

## 来源

| 资料 | 实际回查范围 | 版本/边界 |
| --- | --- | --- |
| R06 [Cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation)、[Cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache) | 每层 K/V、cache 随 token 增长、mask 覆盖 past+current 长度 | 2026-10-17 核对在线 `main`；本地实验不实现其生产 Cache API |
| R19 [第 1 章](https://bojieli.github.io/ai-infra-book/manuscripts/01-%E5%88%9D%E8%AF%86%20AI%20Infra.html)、[第 3 章](https://bojieli.github.io/ai-infra-book/zh-tw/chapter03.html) | 第 1 章容量/单次执行估算；第 3 章 prefill/decode 阶段负载、请求上下文及状态驻留 | 2026-10-17 查看；网页书稿持续修订 |
| R21 本地 [03-显存三分账](../../learning/LLM推理服务技术详解/03-显存三分账.md) | 参数字节、逐层 KV 公式、GQA KV head 影响、GB/GiB 换算 | 本次补充双模型、activation、allocator 指标和余量账本 |
| PyTorch [CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[`memory_allocated`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory_allocated.html)、[`memory_reserved`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory_reserved.html) | allocator active/reserved 口径、缓存池行为 | 2026-10-17 核对官方 stable 文档；版本滚动 |

## 方法

1. 先取设备 total/free、当前进程 `memory_allocated()` 和 `memory_reserved()` 基线。
2. 在 CUDA 上创建独立 target/draft toy 模型，按 `参数数 × 参数字节数` 计算逻辑权重大小，并读取每个参数 Tensor 的实际 `numel × element_size`。
3. 分别为两模型分配显式 K 和 V 张量，按 `2 × 层数 × batch × KV heads × cache length × head dim × 每元素字节` 计算公式值，对照实际 K/V Tensor 字节和 allocated delta。
4. 让 target 执行 B=2 的长度 128 prefill 和长度 133 full-prefix verifier reference。先 warmup 一次以初始化 backend，再清理输出、同步、重置 peak stats，测 steady baseline 上额外的 `max_memory_allocated` 和 `max_memory_reserved`。
5. 单独以配置表计算假设 7B target + 1B draft 的权重、两边 KV、验证 activation reserve 和 runtime/safety reserve，并对照设备 total 与实验开始时的 free。

复现命令（仓库根目录）：

```bash
.venv/bin/python learning/batches/b05-generation-kv-resources/t16-memory-budget/budget_experiment.py > reports/stage-02/2026-10-17-t16/budget_measurements.json
```

逐项结果保存在[原始 JSON](2026-10-17-t16/budget_measurements.json)。

## 结果

### 公式与实测

| 资源项 | 公式/逻辑字节 | 实测 | 差额解释 |
| --- | ---: | ---: | --- |
| Target 权重 | `776 × 4 = 3,104 B` | 参数 Tensor 共 3,104 B | 公式对逻辑参数 storage 精确 |
| Draft 权重 | `744 × 4 = 2,976 B` | 参数 Tensor 共 2,976 B | 公式对逻辑参数 storage 精确 |
| 双模型权重合计 | 6,080 B | model load 后 allocated 14,336 B | 多出 8,256 B 的 active allocated 无法按 Tensor 参数归因；归为未归属 runtime/初始化占用，不记入权重公式 |
| 双模型 K/V | Target 8,192 B + Draft 4,096 B | allocated 新增 12,288 B；Tensor `nbytes` 合计 12,288 B | 公式和实际张量相同 |
| Allocator reserved | 不由权重/KV公式直接推出 | 加载权重后 2,097,152 B；分配 KV 后仍为 2,097,152 B | allocator 以内存块管理；reserved 含 active 区和可复用的未用区，不能再加在 allocated 上 |
| Prefill activation/输出 peak | 实测项 | B=2、S=128：稳态 baseline 上增加 1,163,264 B（1.108 MiB）；peak reserved 22 MiB | 含输出与前向期间 live temporary；baseline 已含驻留模型/KV与 warmup 后 active 项 |
| Verifier reference peak | 实测项 | B=2、S=133：稳态 baseline 上增加 1,256,960 B（1.199 MiB）；peak reserved 22 MiB | full-prefix 验证参考，多 5 个位置；不是 cache-aware 多 token verifier 的直接实测 |

PyTorch 的 allocated 观察活动 Tensor 分配，reserved 观察 caching allocator 管理的池；两者是包含关系。`nvidia-smi` 设备级 used/free 还包括其他进程和 CUDA context，不等于本进程 allocated。权重加载后模型参数 Tensor 是 6,080 B，但进程 allocator active 是 14,336 B，差额 8,256 B；本实验没有对这部分底层 CUDA 分配做指针级归属，故不伪称为参数或 activation。KV Tensor 对照差额为 0，确认 KV 公式输入和 K/V 维度的实现正确。

### 双模型容量

下表把验证激活 reserve 固定为 2 GiB，把 runtime+safety reserve 设为设备 total 的 10%（1.592 GiB）。2 GiB 是明确的规划余量，**不是由 toy model 外推出来的测量**；真实模型与 attention backend 确定后应重新测并替换。容量合计为权重、双 KV、两项 reserve 的总和。

| Batch | Target 权重 | Draft 权重 | Target KV | Draft KV | 验证激活 reserve | Runtime+safety reserve | 预算合计 | 超设备 total | 超审计时 free |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13.039 GiB | 1.863 GiB | 0.501 GiB | 0.188 GiB | 2.000 GiB | 1.592 GiB | 19.182 GiB | 3.261 GiB | 4.573 GiB |
| 2 | 13.039 GiB | 1.863 GiB | 1.001 GiB | 0.375 GiB | 2.000 GiB | 1.592 GiB | 19.870 GiB | 3.949 GiB | 5.261 GiB |
| 4 | 13.039 GiB | 1.863 GiB | 2.002 GiB | 0.751 GiB | 2.000 GiB | 1.592 GiB | 21.247 GiB | 5.326 GiB | 6.638 GiB |

即使 batch=1，该双模型场景也超过设备 total；如果按实验开始时 free 显存作为可用预算，缺口更大。降低并发会线性缩小 KV，但不能消除两模型权重与安全/验证空间所造成的主缺口。此结论只对上表明确写出的参数量、层数、heads、dtype、上下文和 reserve 假设成立。

## 激活解释

prefill 处理整个已知 prompt，验证前向在本实验中处理 context 加 5 个位置。Attention 分数的显式形状和 MLP 中间宽度都会影响临时峰值；执行不同算子时，大多数中间 Tensor 不同时存活。GPU allocator 也会留下空闲块供后续复用，所以 reserved 的 peak 往往高于 active 峰值。为了做 speculative decoding 预算，先把 verifier activation 单列，不能把它当成 KV Cache 长度的另一种叫法，也不能把一次 decode 的显存值当成完整请求峰值。

## 限制

- 实测模型极小、单层 float32，CUDA 审计只验证预算公式和计量方式；不代表真实模型加载、长序列 activation 或多卡运行。
- verifier measurement 是 `TinyDecoderLM.forward` 完整前缀 reference，不是从历史 cache 开始、只验证候选 chunk 的生产路径。
- 7B+1B 配置是容量计算假设，没有读取实际模型 config/card 或加载权重；BF16 不包含量化元数据、权重 staging、padded allocation、optimizer/gradient。
- activation 预留 2 GiB 是保守预算输入，不是统一建议值。实际应按目标 attention backend、并发、最大上下文、验证长度测 peak。
- 未测 CUDA Graph 池、通信缓冲、KV 分页碎片、offload 或多个其他 GPU 进程的稳态干扰；free 值是一次快照。

## 产物

- [T16 预算章节](../../learning/batches/b05-generation-kv-resources/t16-memory-budget/README.md)
- [估算与 CUDA 审计脚本](../../learning/batches/b05-generation-kv-resources/t16-memory-budget/budget_experiment.py)
- [预算原始数据](2026-10-17-t16/budget_measurements.json)
