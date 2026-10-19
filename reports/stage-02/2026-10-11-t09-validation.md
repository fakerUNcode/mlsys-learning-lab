# T09 验证

日期：2026-10-11。范围：因果多头自注意力 reference；SDPA 布尔 mask、内置 causal 参数与推理 dropout 语义。

## 环境与方法

- 环境：仓库 `.venv`；PyTorch 2.11.0+cu128；CPU；默认 float32；随机种子在测试用例中固定。
- CUDA 在该环境可用，但本次验收明确在 CPU 执行；没有据此声称 GPU 验证通过。
- 命令：`.venv/bin/python learning/batches/b03-attention/t09-causal-mask-and-multihead/test_multihead_attention.py`
- 资料：R04《Attention Is All You Need》§3.2.2 与 Figure 2（arXiv v7，2023-08-02）；PyTorch SDPA 官方 API（2026-10-11 查阅）。
- API 语义：`scaled_dot_product_attention` 的 bool `attn_mask=True` 代表允许参与；`is_causal=True` 使用因果约束，不能和 `attn_mask` 同时传入；函数不会因外层 `.eval()` 自动将 dropout 设为零。

## 实际输出

```text
test_boolean_lower_triangle_matches_is_causal ... ok
test_future_token_changes_do_not_affect_past_outputs ... ok
test_inference_always_passes_zero_dropout_to_functional_sdpa ... ok
test_invalid_head_partition_is_rejected ... ok
test_output_shapes_across_batch_sequence_and_heads ... ok
test_sdpa_boolean_true_means_allowed ... ok

Ran 6 tests in 0.048s
OK
退出码：0
```

## 验收映射

| 检查 | 输入/配置 | 结果 |
| --- | --- | --- |
| batch、序列、头数变化 | `(B,S,H)=(1,1,1),(2,5,2),(3,7,4)`，D 为 4/8/16 | 输出始终 `[B,S,D]` 且有限 |
| 未来信息隔离 | 将样本位置 4～5 的所有特征加 100 | 位置 0～3 输出最大差为 0；未来输出确实变化 |
| bool mask True 语义 | 两个分数相等的 key，分别只允许一个 key | 输出分别为对应的 value 2 和 10 |
| 因果参数等价 | 相同 Q/K/V 对比下三角 bool mask 与 `is_causal=True` | 容差 `rtol=1e-6, atol=1e-6` 内通过 |
| eval dropout | 模块配置 `dropout_p=0.75` 后进入 eval | spy 捕获 SDPA 实参 `dropout_p=0.0`；重复输出完全相同 |
| 头维约束 | D=10、H=3 | 构造时抛出可读 ValueError |

代码的数据流为 `[B,S,D] -> [B,S,3D] -> [3,B,H,S,d] -> SDPA -> [B,S,H,d] -> [B,S,D]`。投影层分别学习 Q/K/V，并由输出投影合并头信息；reshape 将 D 切成 H*d，permute/transpose 改变轴解释，transpose 后 `.contiguous()` 让合头 reshape 使用清楚的连续布局。

## 边界

- 检查证明当前 CPU float32 小尺寸实现满足这些语义断言；不能证明所有长度、设备、dtype 或后端均无误。
- 未覆盖 GPU/半精度、反向传播数值、非方形 causal bias、padding mask、GQA/MQA、KV cache、性能或真实任务精度。
