# 参考资料索引

入口核对日期：2026-09-26。任务行给出必读范围；同一资料只读所需部分。官方 API 应切到实际安装版本，完成记录填写文档版本或源码 commit、访问日期、准确小节/页码。第三方路线与本地讲义用于导读，API、正确性和创新判断回到官方文档与原论文。

| 编号 | 资料 | 用途与范围入口 |
| --- | --- | --- |
| R01 | [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Tensors、Datasets & DataLoaders、Build Model、Autograd、Optimization、Save & Load；按任务选小节 |
| R02 | [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)、[Broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) | stride、view/reshape/transpose、连续性、广播规则 |
| R03 | [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)、[gradcheck](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html) | 计算图、梯度累积、禁用梯度与有限差分；高级高阶导暂缓 |
| R04 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | 原论文 §3 模型结构；SDPA 参数、mask、dropout 与因果语义 |
| R05 | [Llama 模型文档](https://huggingface.co/docs/transformers/main/en/model_doc/llama)、[模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | RMSNorm、MLP、RoPE、Attention 与 decoder layer；实际选用其他模型时换为该模型官方实现 |
| R06 | [HF Cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)、[Cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation) | 自回归缓存、位置、mask、DynamicCache、实际 cache API |
| R07 | [HF Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)、[Generation config](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) | 贪心/随机采样、温度/top-k/top-p、EOS、assisted generation；使用冻结版本 |
| R08 | [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)、[PyTorch 安装入口](https://pytorch.org/get-started/locally/) | WSL 支持与限制、驱动/toolkit/runtime 区别、GPU 架构兼容；不照旧日志推断新设备 |
| R09 | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 线程层次、内存、同步、stream/event、合并访存、归约与测量；按安装版主题定位 |
| R10 | [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)、[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) | memcheck/racecheck/synccheck，错误查询、event 与 stream API |
| R11 | [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) | 时间线、launch/同步、Roofline、Memory Workload、occupancy；权限与采集限制单列 |
| R12 | [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) | 构建、注册、Tensor 契约、FakeTensor/opcheck；推理算子无需无目的实现训练 backward |
| R13 | [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) | 算法、分布保持证明、接受率与收益分析；先导读后阅读全文及相关附录 |
| R14 | [DISCO / Dynamic Speculation Lookahead](https://arxiv.org/abs/2405.04304)、[HF 动态推测说明](https://huggingface.co/blog/dynamic_speculation_lookahead) | 动态长度、置信度停止、实验与实现；核对实际随机采样支持和配置 |
| R15 | [Sequoia](https://arxiv.org/abs/2402.12374)、[EDSD](https://aclanthology.org/2026.acl-long.2145/) | 硬件成本、熵信号相关工作；只读问题/方法/实验，不要求复刻树结构 |
| R16 | [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | 校准定义、可靠性图、校准方法；本项目接受事件与分类置信度的区别要自己说明 |
| R17 | [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)、[Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) | 配对重采样、置信区间、概率误差；不能把同一请求内 token 当独立请求 |
| R18 | [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware 思路、分块/在线 softmax 与复杂度；本年度不要求完整高性能重写 |
| R19 | [《深入理解 AI Infra》](https://bojieli.github.io/ai-infra-book/) | 第 1～3 章模型/负载，第 4～5 章硬件/运行时，第 8 章推理；选学第 9、11 章 |
| R20 | [AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/) | 按 Tensor/Transformer/CUDA/性能分析/框架主题查漏，不把整站通读设为前置 |
| R21 | [本地模型基础任务](../learning/stage-02/README.md)、[推理服务讲义](../learning/LLM推理服务技术详解/README.md)、[C++ 讲义](../learning/stage-01/lessons/README.md) | 与现有学习记录衔接；服务 00～06 已学，后续读 07～09 |
| R22 | [基准测试规范](../benchmarks/README.md)、[报告索引](../reports/README.md) | 实验边界、输入、计时、原始数据与结论范围 |
| R23 | [本仓库学习路线](../Infra%20Introduction.md)、[任务清单](TASKS.md) | 阶段顺序、验收边界与实现/验证区分 |
| R24 | [环境检查脚本说明](../scripts/README.md)、[报告规范](../reports/README.md) | 环境、命令、结果和来源信息的记录方式 |
| R25 | [torch.compile 教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)、[Triton 教程](https://triton-lang.org/main/getting-started/tutorials/) | 条件选学：graph break、编译开销、基础 tiling；随实际瓶颈使用 |
| R26 | [vLLM 文档](https://docs.vllm.ai/)、[PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html) | 条件选学/后续方向：服务指标、batching、进程组与集合通信；执行前重新核对支持 |

模型与数据集不预先硬编码：选定时添加官方 model card、dataset card 的链接、许可、不可变 revision 与阅读范围。软件版本和投稿规范可能变化，使用时查对应官方来源，不在长期清单中写死过期要求。
