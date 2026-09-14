# 七、系统总览：AI Infra 做了什么

前面六章推出了六个**问题**。这一章回答：**AI Infra / ML System 领域发明了什么来逐个解决，以及为什么起作用。**

## 7.1 问题 → 手段 → 机制 对照表

| # | 问题（来自第几章） | 根因 | **AI Infra 手段** | **为什么起作用** | 代表实现 |
|---|---|---|---|---|---|
| 1 | 显存碎片浪费 60~80%（三） | 按最大长度预分配连续显存 | **PagedAttention** | 分页按需分配，浪费率 $\to$ 0.8%；相同前缀的块可共享 | vLLM、SGLang |
| 2 | GPU 空等，利用率仅 10.9%（六） | 静态 batch 等最慢者 | **Continuous Batching** | 以 decode 单步为调度单位，谁完谁走、新人补位，$U \to 1$ | vLLM、TGI |
| 3 | 长 prefill 打出 15 倍 TPOT 尖刺（六） | 大 prefill 独占算力 | **Chunked Prefill** | 切块混跑，单步算力工作量从 $2P s_p$ 降到 $2P c$ | Sarathi-Serve、vLLM v1 |
| 4 | TTFT 与 TPOT 互相拖累（五） | 两阶段瓶颈性质相反，绑在同一张卡 | **PD 分离** | prefill 池用 TP 降延迟、decode 池用大 batch 提吞吐，两类 SLO 独立调优 | DistServe、Mooncake、vLLM v1 |
| 5 | KV Cache 太大（三） | KV 头数多、精度高 | **GQA/MQA（模型侧）+ KV 量化** | 压缩比 $\rho = n_{kv}^{\text{MHA}}/n_{kv}^{\text{GQA}}$；FP8 再减半 | Llama-3/Qwen、FP8 KV |
| 6 | Decode 搬运慢，TPOT 下限 7 ms（五） | 每步必读全部权重 | **权重量化 + 算子融合 + CUDA Graph** | 字节减半 → 下限降到 3.5 ms；融合减少 kernel launch 与中间读写 | GPTQ/AWQ、TensorRT-LLM |
| 7 | 注意力显存 $O(s^2)$（四） | $s \times s$ 矩阵被实体化进 HBM | **FlashAttention / FlashInfer** | 分块 + 在线 softmax，显存与 HBM 读写降到 $O(s)$ | FlashAttention-2/3、FlashInfer |
| 8 | 相同 System Prompt 反复重算（二） | 每请求独立 prefill | **Prefix Caching / RadixAttention** | 共享前缀的 KV 块只算一次，用基数树管理复用 | SGLang、vLLM |
| 9 | 显存耗尽导致抢占抖动（六） | 并发超出 KV 容量 | **水位预留 + 准入控制 + Swap/Recompute** | 提前留缓冲避免硬抢占；真发生时按推导三选更优恢复路径 | vLLM Scheduler |
| 10 | 排队导致 TTFT 高（二） | 并发不足 | **SLO 感知调度 + 弹性扩缩容** | 按剩余 SLO 预算排序，优先处理"快超时"的请求 | 各类 Serving 网关 |
| 11 | 多租户互相饿死（六） | 纯 FCFS 下长请求堵死短请求 | **加权公平队列 + 租户配额** | 按已服务量排序、按配额限流，保证无请求被永久饿死 | 多租户网关 |
| 12 | 单卡装不下 / 要更快（五） | 模型太大或算力不足 | **TP / PP / EP 并行 + NCCL** | TP 切矩阵降单步延迟；PP 切层提吞吐；EP 稀疏激活省算力 | Megatron、vLLM |
| 13 | 小 kernel 太多，launch 开销大 | 逐算子调用 | **torch.compile / Triton / CUDA Graph** | 编译期融合算子、录制计算图，省掉大量 launch 与中间读写 | PyTorch 2、Triton |

## 7.2 两个"为什么这是关键创新"的深入解释

### PagedAttention 为什么是分水岭

它借鉴了**操作系统的虚拟内存分页**：把 KV Cache 切成固定块，**逻辑上连续、物理上离散**。三个连锁收益：

1. **碎片归零** —— 浪费率从 63% 降到 0.8%（§三 推导五）
2. **并发提升** —— 省下的显存直接变成更多 KV 块 → 更多并发 → 更短排队 → 更低 TTFT
3. **前缀共享天然成立** —— 相同前缀指向同一物理块，写时复制（Copy-on-Write）保证安全 → Prefix Cache 几乎是免费的

**所以"vLLM 为什么快"有两个并列答案**：Continuous Batching（时间维度）+ PagedAttention（空间维度）。缺一不可。

### PD 分离为什么是趋势

§五 推导三证明了：$I_{\text{pre}} \approx s$（算力争霸），$I_{\text{dec}} \approx B$（带宽争霸）。**同一个硬件配置无法同时最优**。分开之后：

- **Prefill 池**：用 **TP（Tensor Parallel，张量并行）**——把一个大矩阵乘法切开分给多张卡，降低单步延迟 → 优化 TTFT
- **Decode 池**：用大 batch + **PP（Pipeline Parallel，流水线并行）**——把不同层放到不同卡，请求像流水一样流过 → 优化吞吐与 TPOT
- 中间通过网络传 KV Cache

**代价**：KV 传输的网络开销 + 很高的工程复杂度。**收益**：TTFT 与 TPOT 从"互相绑架"变成"各自独立优化"。规模越大、SLO 越严格，这笔交易越划算。

## 7.3 因果闭环总图

```
【物理约束】GPU 的算力 F 与带宽 B 都是有限的
      │
      ├─→ Prefill: I ≈ s  >> I*  →  算力瓶颈
      └─→ Decode : I ≈ B  << I*  →  带宽瓶颈
             │
             └── 两阶段性质相反 ──→ 【调度复杂性】
                                      ├─ Chunked Prefill（防干扰）
                                      ├─ Continuous Batching（防空转）
                                      └─ PD 分离（彻底解耦）

【物理约束】显存总量固定
      │
      ├─→ 权重（固定，由 P 与精度决定）
      ├─→ 激活（随 batch 波动，含 O(s²) 注意力矩阵）
      └─→ KV Cache（随并发 × 长度线性增长）← 剩余量决定一切
             │
             ├─→ GQA / 量化 / FlashAttention / PagedAttention 增大剩余量
             │
             └─→ 剩余量决定【最大并发】 → 排队长度 → TTFT
                        │
                        └─→ 并发超限 → 抢占 → P99 ITL 恶化
                                  └─→ 踢谁由公平性策略决定

【目标函数】min 成本（= max 吞吐）  s.t.  TTFT ≤ SLO_1,  TPOT ≤ SLO_2
```

---

