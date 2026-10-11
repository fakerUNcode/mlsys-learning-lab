# T08 验证

日期：2026-10-11。范围：CPU 双精度单头缩放点积 Attention reference、稳定 softmax、PyTorch SDPA 对照、矩阵乘法 FLOPs 与显式中间矩阵估算。

## 环境与方法

- 环境：仓库 .venv；PyTorch 2.11.0+cu128；CPU；torch.float64；种子 20261011。
- 命令：.venv/bin/python learning/batches/b03-attention/t08-scaled-dot-product-attention/attention_reference.py
- 资料：R04《Attention Is All You Need》§3.2.1（arXiv v7，2023-08-02）；PyTorch scaled_dot_product_attention 官方 API（2026-10-11 查阅）。
- 手写顺序：QKᵀ → 除以 sqrt(D) → 沿 key 轴减最大值并 softmax → 权重乘 V。SDPA 使用 attn_mask=None、dropout_p=0.0、is_causal=False、scale=None。
- FLOPs 按乘法和加法各计一次；两次矩阵乘法约 4*B*S*S*D。每个 score 或 weight 矩阵占 B*S*S*element_size 字节，不含临时量、输入输出、自动微分和分配开销。

## 实际输出

```text
torch=2.11.0+cu128 device=cpu dtype=torch.float64 seed=20261011
shape=[1,3,2] output=[1, 3, 2] max_abs_error=2.220e-16 matmul_flops~72 one_[B,S,S]_matrix=72B
shape=[2,5,8] output=[2, 5, 8] max_abs_error=6.661e-16 matmul_flops~1,600 one_[B,S,S]_matrix=400B
shape=[3,17,32] output=[3, 17, 32] max_abs_error=6.661e-16 matmul_flops~110,976 one_[B,S,S]_matrix=6,936B
extreme_softmax_finite=True
T08 checks passed
```

退出码 0。三组手写输出与 SDPA 均在 rtol=1e-10、atol=1e-12 下通过；最大绝对误差依次为 2.220e-16、6.661e-16、6.661e-16。程序也断言输出形状、权重行和为 1、输出有限。巨大 logits [1000,999,-1000] 的稳定 softmax 有限且归一化。

## 成本账

| B,S,D | 两次矩阵乘法 FLOPs | 单个 [B,S,S] float64 矩阵 | scores 与 weights 合计 |
| --- | ---: | ---: | ---: |
| 1,3,2 | 72 | 72 B | 144 B |
| 2,5,8 | 1,600 | 400 B | 800 B |
| 3,17,32 | 110,976 | 6,936 B | 13,872 B |

S 翻倍时显式 score/weight 元素数按 S 平方增长。softmax 的 O(B*S*S) 运算未计入两次矩阵乘法 FLOPs。

## 边界

- 结果覆盖三组 CPU float64 小输入，不代表 GPU、半精度、速度或真实模型。
- Reference 显式保留完整 scores/weights，未实现融合、分块、反向传播、mask、因果语义或 dropout。
- 程序断言不能证明学习者已独立复写；个人掌握需另行确认。
- README 包含逐步 shape、生活化概念解释、稳定 softmax、缩放动机和成本估算。
