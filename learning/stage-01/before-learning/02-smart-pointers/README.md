# 智能指针

## 前置基础

1. 所有权表示谁保证对象存活并最终销毁它。
2. 裸指针只保存地址，不自动表达所有权。
3. 引用计数记录共享所有者数量。

## 使用边界

| 类型 | GPU/推理场景 | 边界 |
| --- | --- | --- |
| `unique_ptr` | 请求独占 workspace | 默认首选，可移动不可复制 |
| `shared_ptr` | 多请求共享模型 | 只在确需共享寿命时使用 |
| `weak_ptr` | 模型缓存与观察者 | 使用前必须 `lock()` |

对象销毁条件：

\[
n_s
\overset{\text{减至零}}{=}
0
\Rightarrow_{\text{析构对象}}
\text{释放资源}
\]

## 符号说明

- \(n_s\)：强引用数量。
- 下标 \(s\)：strong，强引用。

## 程序实例

见[指针示例](../../examples/02-smart-pointers/README.md)。它模拟两个请求共享模型，而缓存不延长模型寿命。

## 部署说明

```bash
cmake --build build/stage1 --target smart_pointer_demo
./build/stage1/02-smart-pointers/smart_pointer_demo
```

## 直观理解

`unique_ptr` 是唯一门卡，`shared_ptr` 是多人共同续租，`weak_ptr` 是不续租的通讯录。模型缓存应使用通讯录，否则过期权重可能一直占用显存。
