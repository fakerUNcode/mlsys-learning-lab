# 学习进度

这里是全仓库的实验记录索引。状态描述仓库内材料、代码与验证证据，不代表任何读者的个人能力评价；目录或讲义已经存在本身不代表结论已验证。

## 当前进度

| 项目 | 最新状态 | 证据与边界 | 下一步 |
| --- | --- | --- | --- |
| 环境与计时 | 基础闭环已有 | 历史环境报告、Vector Add；新设备仍需重新验收 | 验证实际工具链、profiling 权限与可用显存 |
| C++ 第 00～03 节 | 已完成 | 既有 runtime、智能指针与 STL 程序及 CTest | 保留，按需回查 |
| C++ 模板与现代类型 | T04 工程与自动验证通过 | [Buffer 练习](../learning/batches/b01-environment-tensor-layout/t04-cpp-buffer-ownership/README.md)与[验证报告](../reports/stage-02/2026-10-08-t04-validation.md)；实现来源与自动检查范围见报告 | 保留回查 |
| C++20、ABI 等专题 | 按需补缺 | 不将讲义存在视为已掌握或必修 | 遇到扩展构建等问题再补 |
| PyTorch、梯度、模型前向 | B01/T01～T04 已验收；T05 数值验证通过；T06/T07 有程序证据；T08～T12 CPU 检查通过，T10 另有 GPU float64 检查 | [B 批次章节索引](../learning/batches/README.md)、[T08 Attention](../learning/batches/b03-attention/t08-scaled-dot-product-attention/README.md)、[T09 Attention](../learning/batches/b03-attention/t09-causal-mask-and-multihead/README.md)、[T10 数值与成本](../learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/README.md)、[T11 四个 Decoder 模块](../learning/batches/b04-decoder-block/t11-embedding-residual-rmsnorm-swiglu/README.md)、[T11 报告](../reports/stage-02/2026-10-13-t11-validation.md)、[T12 位置与头映射](../learning/batches/b04-decoder-block/t12-rope-gqa-position/README.md)、[T12 报告](../reports/stage-02/2026-10-14-t12-validation.md) | T13 简化 Decoder 前向与第一月复盘 |
| LLM 推理服务 00～06 | 讲义与记录已保存 | 不等同模型部署或性能实验 | 用后续模型实现补充验证 |
| CUDA 与扩展优化 | 待推进 | 笔记与模块约定；无成熟自定义 kernel 项目 | 完成模型基础后进入 |
| 模型级推理实验 | 本仓库尚无完整 runner | 目前以基础知识和小型验证为主 | 在本仓库建立可复现实现与报告 |

目录编号沿用历史；新版年度阶段不回写旧日期。既有学习日志保留，新增状态以实际确认、代码与独立解释分别记录。

## 每日记录

| 日期 | 学习主题 | 结果 |
| --- | --- | --- |
| [2026-09-08](2026-09-08.md) | 建仓与环境入口 | 建立最小工程骨架 |
| [2026-09-09](2026-09-09.md) | 阶段 0 与性能基线 | 完成环境报告和 Vector Add 基准 |
| [2026-09-10](2026-09-10.md) | 路线收缩与 Stage 1 实验 | 跑通 C++、CTest 和 Sanitizer 闭环 |
| [2026-09-11](2026-09-11.md) | C++ 语法地基 | 修正引用、所有权和生命周期认识 |
| [2026-09-12](2026-09-12.md) | 智能指针与 STL | 第 02、03 节完成，下一步进入模板与类型特性 |
| [2026-09-14](2026-09-14.md) | LLM 推理服务前置知识 | 完成 00～03，下一步学习注意力复杂度与 FlashAttention |
| [2026-09-15](2026-09-15.md) | LLM 推理服务：注意力、带宽与调度 | 完成 04～06，下一步进入系统总览 |
| [2026-09-26](2026-09-26.md) | 年度路线与仓库职责更新 | 基础学习与年度项目拆分；学习成果保持原路径 |
| [2026-09-27](2026-09-27.md) | T01 设备验收与 T02 Tensor shape | CUDA Toolkit/kernel 验收通过；完成 Tensor、索引、广播练习及独立复写 |
| [2026-10-08](2026-10-08.md) | T03/T04 收尾、T05 数值验证与程序导读重写 | T03/T04 已验收；T05 float64 对照与 gradcheck 通过、口述待验收；T04/T05 已补齐前置知识、设计决策和证据映射 |
| [2026-10-09](2026-10-09.md) | Stage-02 章节目录整理与 T06 梯度模式实验 | 章节入口按 `tNN-章节名` 归整；T06 实验与术语说明复核通过；T05 独立口述仍待验收 |
| [2026-10-11](2026-10-11.md) | T08 单头与 T09 因果多头 Attention | T08 三组 CPU float64 reference/SDPA 对照及 T09 六项 CPU 语义检查通过；T09 独立复写于 2026-10-12 验收通过 |
| [2026-10-12](2026-10-12.md) | T10 数值稳定性与 Attention 成本 | 朴素 softmax 溢出与减最大值对照、全 mask 行约定、序列翻倍资源账；5 项 CPU 检查通过 |
| [2026-10-13](2026-10-13.md) | T11 Decoder Block 基础模块 | Embedding、残差、RMSNorm、SwiGLU 四个独立模块；6 项 CPU reference 检查通过 |
| [2026-10-14](2026-10-14.md) | T12 旋转、头映射与位置语义 | 四个例子、12 项 CPU 检查；整段 / 增量一致与固定真实配置对照 |

## 更新规则

每天结束学习时记录四项内容：

1. 今天实际完成了什么；
2. 哪些命令、测试或输出可以证明；
3. 哪些概念仍然不会；
4. 下一次只推进哪一个明确任务。

总路线和阶段验收标准见 [Infra 学习指南](../Infra%20Introduction.md)，当前课程入口见 [PyTorch 与模型基础](../learning/batches/README.md)。
