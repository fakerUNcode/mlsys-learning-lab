# STL基础

## 前置基础

1. 容器保存元素。
2. 迭代器表示范围位置。
3. 算法处理半开区间。

## 核心关系

| 角色 | 示例 | Infra 用途 |
| --- | --- | --- |
| 容器 | `vector`、`map` | 保存 Tensor 元数据或任务 |
| 迭代器 | `begin/end` | 界定一个 batch 范围 |
| 算法 | `sort`、`find_if` | 按 shape 分组或筛选任务 |
| allocator | 默认 allocator | 管理 Host 原始存储 |

范围元素数量为：

\[
N
\overset{\text{迭代器作差}}{=}
e-b
\]

## 符号说明

- \(b\)：起始迭代器。
- \(e\)：尾后迭代器。
- \(N\)：范围元素数。

标准 allocator 默认管理 Host 内存，不等于 CUDA caching allocator。二者都可能减少重复申请，但操作不同地址空间。

## 延伸阅读

[详细笔记](detailed-notes.md)还包含模板、C++17 和 allocator 的原综合讲解；拆分后的主题以各目录 README 为准。

## 程序实例

[现代类型示例](../../examples/03-modern-types/README.md)使用 vector 和 accumulate。

## 直观理解

Host 侧任务队列像送往 GPU 的货架：容器保存任务，迭代器圈定本批范围，算法负责分组。它们组织 GPU 工作，但不会自动变成 CUDA kernel。
