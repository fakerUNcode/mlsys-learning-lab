# B06 真实 target-only baseline

B06 属于任务清单阶段 1，第 7～8 周，依赖 B05（生成、KV 与资源预算）与 GPU 可用。目标是在冻结的真实模型上建立 target-only 基线：选型冻结、前向/缓存/采样契约、初始 benchmark 与阶段证据复核。

| 任务 | 主题 | 章节与证据 |
| --- | --- | --- |
| T17 | 选择模型、冻结许可与版本 | [选型与冻结详解](t17-model-selection/README.md)、[冻结 manifest](t17-model-selection/frozen_manifest.json)、[完成记录](../../../plan/completed/T17.md)、[验证报告](../../../reports/stage-02/2026-10-19-t17-model-selection.md) |
| T18 | 真实前向、缓存与采样契约 | 待创建 |
| T19 | target-only 初始 benchmark | 待创建 |
| T20 | 阶段 1 证据复核 | 待创建 |

T17 按 T16 实测预算（RTX 5080 Laptop total 15.920 GiB、free 14.608 GiB 时点值）在 5 个 target 候选中选定 Qwen/Qwen3-4B（apache-2.0），冻结 revision `1cfa9a7208912126459214e8b04321603b3df60c`；8B 因权重单独超 free 被排除，阶段 3 draft 预留同家族 Qwen3-0.6B（配对预算 13.485 GiB，待 T30 重测）。T18 起的模型加载、停止符与采样参数必须引用 manifest，禁止浮动 revision 与隐式默认值。
