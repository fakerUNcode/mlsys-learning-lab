# T11 Decoder Block 基础模块验证

日期：2026-10-13。范围：Embedding、残差加法、RMSNorm、Llama 风格 SwiGLU 的独立实现、公式/维度核对、参数计数与 reference 数值对照。

## 环境与方法

- 基准代码状态：仓库原始 `HEAD=d3ef5fe8c63e8dab2022095ab79eb954761bafc6`；本报告对应的 T11 产物尚未提交。
- 环境：Python 3.12.3、PyTorch 2.11.0+cu128；本次全部运行在 CPU。没有使用 GPU。
- 输入：Embedding 和残差使用固定 float64 值；RMSNorm 使用固定输入并模拟 R05 的 float32 统计；SwiGLU 使用固定种子 1104 和 float64 参数/输入。
- R05 源码：Hugging Face Transformers commit `22cfdfaad99038416b56d21e35b0d32d4f4ee2d3`，`LlamaRMSNorm.forward`、`LlamaMLP.__init__/forward`、`LlamaDecoderLayer.forward`、`LlamaModel.__init__/forward`。同 commit 配置默认 `hidden_act=silu`、`rms_norm_eps=1e-6`、`mlp_bias=False`。
- R01：PyTorch Learn the Basics 的 Build the Neural Network，核对 `nn.Module`、参数/子模块注册、`__init__` 和 `forward` 调用约定。

## 复现命令

```bash
.venv/bin/python learning/batches/b04-decoder-block/t11-embedding-residual-rmsnorm-swiglu/embedding.py
.venv/bin/python learning/batches/b04-decoder-block/t11-embedding-residual-rmsnorm-swiglu/residual.py
.venv/bin/python learning/batches/b04-decoder-block/t11-embedding-residual-rmsnorm-swiglu/rmsnorm.py
.venv/bin/python learning/batches/b04-decoder-block/t11-embedding-residual-rmsnorm-swiglu/swiglu.py
.venv/bin/python learning/batches/b04-decoder-block/t11-embedding-residual-rmsnorm-swiglu/test_t11.py
```

## 实际结果

| 模块 | 输入 → 输出 | Reference 结果 | 参数 |
| --- | --- | --- | ---: |
| Embedding | `[2,3]` long IDs → `[2,3,3]` | 直接按表索引，最大误差 0 | `V*D=7*3=21` |
| Residual | `[1,2,2] + [1,2,2]` → `[1,2,2]` | 输出 `[[[1.5,1.0],[5.0,7.0]]]`，逐值一致 | 0 |
| RMSNorm | `[2,4]` → `[2,4]` | 末轴归一化，epsilon `1e-6`；参考表达式逐值一致 | `D=4` |
| SwiGLU | `[2,3,4]` → gate/up `[2,3,6]` → `[2,3,4]` | gate SiLU 后与 up 逐元素相乘，再 down 投影；最大误差 `0` | `3DF=3*4*6=72` |

`test_t11.py` 执行 6 项 unittest，全部通过：Embedding 查表/shape/参数数、拒绝浮点 ID、残差值/shape/零参数、RMSNorm 轴与 epsilon、SwiGLU 两分支/激活/shape/参数数、含偏置计数。

## 参数与结构核对

- 残差两输入必须完全同形 `[B,S,D]`；本实现拒绝不匹配，不做隐式广播或投影。相加不引入权重。
- RMSNorm 的 `mean(dim=-1, keepdim=True)` 在 `[B,S,D]` 上产生 `[B,S,1]`，每个 batch/token 各自只归约 D 特征。epsilon 加在均方内、开平方前，与 R05 一致。
- SwiGLU 为 gate/up/down 三个线性映射。Llama 默认无偏置，所以参数为 `3DF`；三层启用偏置时为 `3DF+2F+D`。
- 例子 `V=9,D=4,F=6` 下，四模块参数共 `36+0+4+72=112`；若 SwiGLU 开 bias，则为 128。
- R05 decoder layer 是两次 pre-norm 残差：`residual + self_attn(input_layernorm(x))`，再 `residual + mlp(post_attention_layernorm(x))`。两个 add 均保持 `[B,S,D]`，两处 RMSNorm 各有 D 参数；Attention 与 MLP 自身也有参数，不在四模块总数中。

## 限制

- Reference 检查是小型 CPU 算子级数值对照；SwiGLU reference 使用相同存储权重的显式矩阵式，验证 forward 公式，不是独立模型实现，也不验证训练质量。
- RMSNorm 遵循 R05 的 float32 累积再转回输入类型。当前脚本没有进行 fp16/bf16 对照或 CUDA 测试。
- 未检查梯度、训练收敛、吞吐、峰值显存、真实模型权重兼容、完整 Decoder Block。
- Llama 配置允许覆盖激活函数、epsilon 与 bias；结论只覆盖记录的源码版本及默认配置。
