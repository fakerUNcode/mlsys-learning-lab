# T17 模型选型、许可与版本冻结

## 问题

T16 的容量账本已经证明：假设的 BF16 7B+1B 双模型在 B=1 就超过本机整卡 total 3.261 GiB。T17 是 B06 的第一步，回答进入真实模型前必须冻结的三件事：

1. **选哪个 target 模型**——依据是实测设备预算，不是模型知名度；
2. **版本钉在哪里**——40 位不可变 commit revision，实验全程禁止 `main`/`latest` 浮动引用；
3. **契约记什么**——tokenizer/特殊 token、dtype、采样默认值、依赖版本与许可，全部落进 manifest，T18 的前向/缓存/采样契约直接引用它，不允许各自重新解析。

## 基础

| 名词 | 本章含义 |
| --- | --- |
| revision | HF 仓库的 git commit hash。40 位十六进制串指向不可变快照；tag 和分支名会移动，不能作为实验版本。 |
| model card 元数据 | 仓库 README.md 头部的 YAML（license、library_name 等）。HF API 的 `cardData` 字段直接返回解析结果。 |
| gated 仓库 | 需要登录并同意条款才能下载的仓库。选型优先非 gated，避免复现环境被账号状态阻塞。 |
| safetensors index | `model.safetensors.index.json` 的 `metadata.total_size`，多分片模型全部张量字节之和；单分片小模型没有该文件。 |
| 权重字节 | 本章指 `model*.safetensors` 文件字节之和。它含少量格式头部，是比"参数数×2"略保守的显存驻留估计。 |
| thinking 模式 | Qwen3 后训练模型的聊天模板开关（`enable_thinking`）。开启时输出思维链段；对 T18 是必须显式冻结的采样输入之一。 |

## 选型规则

预算口径**逐项沿用 T16 容量表**，不改口径以保证可比：BF16 权重与 KV（每元素 2 字节）、context 4096、验证空间 5 token、激活预留 2 GiB（规划输入，待 T18/T19 实测替换）、runtime+safety 预留设备 total 的 10%。

设备读数（2026-10-19 脚本采样，与 T16 审计值一致）：total 17,094,475,776 B（15.920 GiB），当时 free 15,685,648,384 B（14.608 GiB）。free 是共享 GPU 的时点值。

候选评估同时看两个口径，都按 B=1：

- **target-only**（本阶段 T18～T19 的负载）：target 权重 + KV(4096) + 激活预留 + 安全预留；
- **pair**（阶段 3 配对的立足空间检查，非验收项）：再加同家族最小 draft 的权重与 KV。

判定线取**实测 free**（比 total 更严格）。在两个口径都装进的候选里取权重最大者：研究主线是推测解码的系统行为，target 越接近真实负载，预fill/decode 成本结构越有代表性。

## 候选对比

权重字节来自各仓库 pinned revision 的文件清单求和（`files_metadata=True`），架构字段来自同 revision 的 `config.json`，没有使用记忆中的模型参数。完整原始数据见[选型原始 JSON](../../../../reports/stage-02/2026-10-19-t17/selection_measurements.json)。

| 候选（B=1） | 许可 | 权重 | target-only 预算 | 配对预算（含 draft） | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen/Qwen3-8B | apache-2.0 | 15.256 GiB | 19.411 GiB | 21.249 GiB | 排除：权重单独已超实测 free，target-only 超 total |
| **Qwen/Qwen3-4B（选定）** | apache-2.0 | 7.492 GiB | 11.647 GiB | 13.485 GiB | 装入 free 的最大 target，余量 1.123 GiB |
| Qwen/Qwen3-1.7B | apache-2.0 | 3.784 GiB | 7.814 GiB | 9.652 GiB | 可行但更小；作为降级备选 |
| Qwen/Qwen2.5-1.5B | apache-2.0 | 2.875 GiB | 6.577 GiB | 7.544 GiB | 可行；家族较旧，不作首选 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | apache-2.0 | 2.049 GiB | 5.727 GiB | 无同家族 draft | 排除偏好：T30 需跨家族配对 |

选定 **Qwen/Qwen3-4B**，配对 draft 预留 **Qwen/Qwen3-0.6B**（权重 1,503,300,328 B，28 层、8 KV heads，同 tokenizer）。选它的依据链：

1. 装进实测 free 的最大 target——8B 的排除是预算事实，不是偏好；
2. 同家族有 0.6B draft（tokenizer/词表一致），阶段 3 配对（T30）在同预算内有立足空间；
3. apache-2.0、非 gated，许可与复现无账号阻塞；
4. GQA（32 Q heads / 8 KV heads）、36 层、head_dim 128，与 T12 的 RoPE/GQA 学习内容同构，R05 的 Llama 式实现约定可直接迁移到 `modeling_qwen3.py`。

一个需要明确记录的选择：`Qwen/Qwen3-4B` 是**后训练模型**（非 `Qwen3-4B-Base`）。它带聊天模板与 thinking 开关，T18 的采样契约更完整；若后续实验需要纯 base 分布，再按同一流程另行冻结，不混用 revision。

## 冻结结果

全部字段来自 revision `1cfa9a7208912126459214e8b04321603b3df60c` 下的文件，机器可读版本见 [frozen_manifest.json](frozen_manifest.json)。

| 项 | 冻结值 |
| --- | --- |
| repo / revision | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`（40 位十六进制，已用 HF REST API 独立核对同一 sha） |
| 许可 | apache-2.0（cardData；仓库 LICENSE 文件为 Apache License 文本，本地 sha256 见 manifest） |
| 架构 | 36 层，32 Q heads / 8 KV heads，head_dim 128，hidden 2560，vocab_size 151936，`tie_word_embeddings: true` |
| dtype | bfloat16（config `torch_dtype`；权重 8,044,982,000 B ≈ 4.02B 参数） |
| tokenizer | vocab_size 151643，len 151669；eos `<|im_end|>`(151645)，pad `<|endoftext|>`(151643)，bos 无；`<|im_start|>`=151644；带聊天模板 |
| generation_config | eos_token_id `[151645, 151643]`（双停止符），do_sample true，temperature 0.6，top_k 20，top_p 0.95 |
| 依赖 | Python 3.12.3，torch 2.11.0+cu128，transformers 5.20.1（模型卡要求 ≥4.51.0），huggingface_hub 2.2.0 |

两个契约级提醒（T18 必须显式处理，不能隐式继承默认值）：

- **EOS 是双 id 列表**。tokenizer 的 eos 与 generation_config 的列表不一致时，以 generation_config 为停止语义权威，并在采样配置里显式传入；
- **do_sample/temperature/top_k/top_p 有非中性默认**（thinking 模式推荐值）。T18 冻结采样变换顺序时要显式覆盖，greedy 单独作为一种配置而不是"未传参"。

## 脚本执行路径

[select_and_freeze.py](select_and_freeze.py) 的真实运转顺序：

```text
torch.cuda.mem_get_info() 读设备 total/free（不初始化大块显存）
  → 对每个候选：model_info(files_metadata=True) 取 sha/许可/文件字节
      → sha 断言 40 位十六进制，否则拒绝冻结
      → 按 sha 下载 config.json，解析层数/头数/head_dim/dtype
      → 权重字节 = model*.safetensors 文件求和；多分片时与 index.total_size 交叉核对
  → 逐候选算 target-only 与 pair 预算（T16 公式），对照 free 判定
  → 选定：双口径 fits 的最大权重者
  → 冻结：按 sha 下载 8 个小文件（不含权重）+ sha256 指纹
  → 离线加载 snapshot 目录的 tokenizer，核验特殊 token/词表/聊天模板
  → 解析 generation_config → 写 frozen_manifest.json + reports 原始 JSON
```

设计选择与后果：

- **权重字节用文件清单而不是参数量换算**：`tie_word_embeddings` 的模型没有独立 lm_head 权重，参数数与字节数会互相误导；文件字节就是显存必须驻留的真实量。
- **冻结不下载 8 GB 权重**：T17 的验收是选型与契约冻结；权重下载命令已写进 manifest，T18 首次加载时执行。重复运行本脚本幂等（revision 钉死，二次运行 sha 不变即为证据）。
- **单分片仓库没有 index.json**：异常分支捕获 404 属预期；交叉核对只在多分片时执行，差距超过每分片 64 KiB 头部解释范围即报错，防止口径悄悄漂移。

## 复现

从仓库根目录：

```bash
.venv/bin/python learning/batches/b06-target-only-baseline/t17-model-selection/select_and_freeze.py
```

首次运行需要网络（匿名限速可用）；再次运行核对 manifest 与输出一致。独立核对 revision 可用 REST API：

```bash
curl -s https://huggingface.co/api/models/Qwen/Qwen3-4B | jq '.sha, .cardData.license, .gated'
```

T18 加载权重（按冻结 revision，禁止省略 `--revision`）：

```bash
.venv/bin/hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

## 边界

- 设备 free 是共享 GPU 的时点采样；配对口径余量 1.123 GiB 偏紧，T30 配对验收时必须重新实测，必要时降级到 Qwen3-1.7B（预算见上表）。
- 激活预留 2 GiB 仍是 T16 的规划输入；Qwen3-4B 的真实 prefill/decode 峰值由 T18/T19 实测后替换，budget 结论随之复核。
- 未加载权重、未跑前向、未验证生成质量；thinking 模式（`enable_thinking` 默认值）与聊天模板行为留给 T18 采样契约显式冻结。
- transformers 5.20.1 对 Qwen3 的实现与模型卡要求的 4.51.0 存在大版本差异；T18 若遇到行为偏差，以安装版源码为准并回写 manifest。
- 候选清单不含闭源许可模型与更大设备方案；若更换设备，本流程整体重跑，不复用本次结论。

## 资料

- R07：[Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)、[Generation config](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)。冻结版本原则与采样默认值的契约地位来自这里；在线 main 文档于 2026-10-19 查看，实际以 transformers 5.20.1 安装版为准。
- R22：[基准测试规范](../../../../benchmarks/README.md)、[报告索引](../../../../reports/README.md)。来源记录、可复现命令与结论范围要求。
- Qwen3-4B 官方仓库：[model card](https://huggingface.co/Qwen/Qwen3-4B)（许可、用法、thinking 说明）、`config.json`/`tokenizer_config.json`/`generation_config.json`，均读取自 revision `1cfa9a7208912126459214e8b04321603b3df60c`，访问日期 2026-10-19。
- huggingface_hub 2.2.0：`model_info(files_metadata=True)`、`hf_hub_download(revision=sha)`；40 位 revision 语义见 [HF 模型仓库文档](https://huggingface.co/docs/huggingface_hub/en/concepts/revisions)。
- T16 预算方法回查：[T16 显存预算](../b05-generation-kv-resources/t16-memory-budget/README.md) 的预算式、预留约定与设备读数。
