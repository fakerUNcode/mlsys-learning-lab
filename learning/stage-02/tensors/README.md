# Tensor 学习资料索引

本目录按照任务和章节组织 B01 的 Tensor 学习材料。建议按 T01 → T02 → T03 的顺序推进；每个任务都有自己的学习入口、分章教程、练习记录与验证脚本。

每篇章节教程和实验记录开头都列有本篇专属的“术语与前置知识”，可以直接从该篇开始学习。统一[术语表](terms.md)汇总跨章节的词义，作为查阅和扩展参考。本任务图床链接与本地源图的对应表见[图片索引](image-index.md)。

## 任务导航

| 任务 | 内容 | 状态 | 学习入口 |
|---|---|---|---|
| B01/T01 | 当前设备环境、GPU/PyTorch、CUDA Toolkit 与 profiler | 已验收 | [环境报告](../../../reports/stage-02/2026-09-27-t01-environment.md) |
| B01/T02 | Tensor 创建、索引、广播、批量矩阵乘法 | 已验收 | [T02 练习与批阅](t02/README.md) |
| B01/T03 | 存储、stride、连续性、view/reshape、dtype/device | 进行中 | [T03 实验记录](t03/README.md) |

## T02 章节

按顺序学习：

1. [Tensor 与 shape](t02/chapters/01_tensor_and_shape.md)
2. [索引与切片](t02/chapters/02_indexing_and_slicing.md)
3. [逐元素运算与广播](t02/chapters/03_broadcasting.md)
4. [批量矩阵乘法与 shape 推导](t02/chapters/04_batched_matmul.md)

练习答案与验收记录在 [T02 记录](t02/README.md)。

## T03 章节

按顺序学习：

1. [存储、stride 与连续性](t03/chapters/01_storage_stride_contiguous.md)
2. [transpose、view、reshape 与别名](t03/chapters/02_views_reshape_aliasing.md)
3. [dtype、device 与存储检查](t03/chapters/03_dtype_device.md)

实验预测与记录在 [T03 实验表](t03/README.md)。

## 官方资料

本地分章教程是学习主线；官方页面用于术语核对和延伸阅读：

- R01：[PyTorch Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- R02：[Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- R02：[Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
