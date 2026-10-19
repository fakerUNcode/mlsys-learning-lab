# T17 验证报告：模型选型、许可与版本冻结

## 问题

按 T16 实测设备预算选择 B06 阶段的真实 target 模型，冻结不可变 40 位 revision、tokenizer/特殊 token、dtype、依赖与许可，产出机器可读的准备 manifest，使 T18 的前向/缓存/采样契约有唯一权威配置来源。

## 环境与依赖

| 项 | 值 |
| --- | --- |
| 设备 | NVIDIA GeForce RTX 5080 Laptop GPU；total 17,094,475,776 B（15.920 GiB）；选型时 free 15,685,648,384 B（14.608 GiB） |
| 软件 | Python 3.12.3；torch 2.11.0+cu128；transformers 5.20.1；huggingface_hub 2.2.0（本次新装入 `.venv`） |
| 网络 | HF Hub 匿名访问（未配置 HF_TOKEN）；选型与冻结全程未遇到限速失败 |
| 脚本 | `learning/batches/b06-target-only-baseline/t17-model-selection/select_and_freeze.py` |

## 方法

1. 预算口径逐项沿用 T16：BF16 权重+KV、context 4096、verify 5 token、激活预留 2 GiB、safety 10% total。判定线取实测 free（严于 total）。
2. 7 个候选（5 target + 2 同家族 draft）全部从 HF API `model_info(files_metadata=True)` 取 40 位 sha、许可与文件字节；架构字段从 pinned revision 的 `config.json` 解析。权重字节 = `model*.safetensors` 文件求和，多分片与 index `total_size` 交叉核对。
3. 对 target 候选计算 target-only（B=1, S=4096）与 pair（含同家族 draft，S=4101/4096）两套预算，取双口径均装入 free 的最大权重者为 target。
4. 冻结：按选定 sha 下载 8 个小文件（config/generation_config/tokenizer 三件套/LICENSE 等，不含权重），计算 sha256 指纹；离线加载 tokenizer 核验特殊 token 与词表；解析 generation_config。
5. 独立交叉验证：用 `curl` 直连 HF REST API 核对 sha、许可、gated 状态；二次运行脚本确认幂等（同 sha、同输出）。

## 正确性（验收逐条）

| TASKS.md 验收项 | 结果 | 证据 |
| --- | --- | --- |
| 目标模型按实测预算选择 | 通过 | 8B 权重 15.256 GiB 单独超 free 14.608 GiB 被排除；选定 Qwen3-4B target-only 11.647 GiB、pair 13.485 GiB，均 ≤ free。见[原始 JSON](2026-10-19-t17/selection_measurements.json) 的 evaluations 字段 |
| 记录不可变 40 位 revision | 通过 | `1cfa9a7208912126459214e8b04321603b3df60c`，脚本内正则断言 40 位十六进制；REST API 独立返回同一 sha |
| tokenizer/特殊 token | 通过 | vocab 151643 / len 151669；eos `<|im_end|>`=151645，pad `<|endoftext|>`=151643，bos 无，`<|im_start|>`=151644；generation_config eos 列表 `[151645, 151643]` 已单列。manifest tokenizer/generation_config 字段 |
| dtype | 通过 | config `torch_dtype: bfloat16`；权重 8,044,982,000 B（≈4.02B 参数） |
| 依赖 | 通过 | Python 3.12.3 / torch 2.11.0+cu128 / transformers 5.20.1（模型卡要求 ≥4.51.0）/ huggingface_hub 2.2.0，记入 manifest dependencies |
| 确认许可 | 通过 | cardData `apache-2.0`，REST API 同值，`gated: false`；LICENSE 文件为 Apache License 全文（11,343 B，sha256 见 manifest） |
| 不使用 "latest" | 通过 | 脚本所有下载调用都传 `revision=<sha>`；冻结产物中无分支/tag 引用；重复运行 sha 不变 |

## 分析

- **7B+1B 缺口的消化方式**：T16 指出的 3.261 GiB 缺口通过降档 target（7B→4B）解决，而不是引入量化（O02 条件任务不混入 baseline）或缩上下文假设。4B 在本机同时保住 target-only 基线和阶段 3 的 4B+0.6B 配对空间（余量 1.123 GiB）。
- **权重口径**：`tie_word_embeddings: true` 的模型没有独立 lm_head 分片，"参数数×2"与文件字节会互相误导；本报告直接采用文件字节（显存真实驻留量，含少量 safetensors 头部，偏保守）。
- **契约风险提前暴露**：generation_config 的 EOS 是双 id 列表且带非中性采样默认（do_sample/0.6/20/0.95），tokenizer eos 与其不完全一致——T18 必须显式传入停止符与采样参数，不能依赖隐式默认。thinking 开关（聊天模板 `enable_thinking`）同样留给 T18 显式冻结。
- **失败案例记录**：首次运行因单分片仓库（Qwen3-0.6B）没有 `model.safetensors.index.json` 而 404 退出；修复为文件清单求和 + 多分片才交叉核对的口径，并把该约束写成脚本内注释。该失败不影响结论，但说明了为什么权重字节来源选文件清单。

## 局限

- 未下载 8 GB 权重、未跑前向或生成；激活预留 2 GiB 仍是规划输入，待 T18/T19 实测替换后复核预算结论。
- free 是时点采样；配对余量 1.123 GiB 偏紧，T30 前必须重测，降级路径（Qwen3-1.7B）已在候选表留出。
- transformers 5.20.1 与模型卡基线 4.51.0 有大版本差；T18 遇到行为差异时以安装版源码为准并回写 manifest。
- 候选范围限定在可公开下载的 Apache 系小模型；未评估闭源许可或更大设备方案。

## 结论

Qwen/Qwen3-4B @ `1cfa9a7208912126459214e8b04321603b3df60c`（apache-2.0）冻结为 B06 阶段 target；阶段 3 draft 预留 Qwen3-0.6B（同家族同 tokenizer，待 T30 正式验收配对）。冻结配置与准备 manifest：[frozen_manifest.json](../../learning/batches/b06-target-only-baseline/t17-model-selection/frozen_manifest.json)。T17 验收项全部通过。

## 复现

```bash
# 选型 + 冻结 + manifest（幂等，重复运行应得到同一 revision 与输出）
.venv/bin/python learning/batches/b06-target-only-baseline/t17-model-selection/select_and_freeze.py

# 独立核对 revision/许可/gated
curl -s https://huggingface.co/api/models/Qwen/Qwen3-4B | jq '.sha, .cardData.license, .gated'

# T18 权重下载（必须带 --revision）
.venv/bin/hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

原始数据：[selection_measurements.json](2026-10-19-t17/selection_measurements.json)（设备快照、7 个候选的 revision/许可/架构/权重字节、两口径预算与判定）。
