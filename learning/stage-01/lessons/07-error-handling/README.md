# 错误处理

## 前置基础

1. 异常沿调用栈传播。
2. error code 要求调用者显式检查。
3. RAII 在异常展开时清理已构造对象。

## 使用边界

| 场景 | 建议 |
| --- | --- |
| C++ 内部深层失败 | 异常并保留上下文 |
| 普通的“没有值” | `optional` 或结果类型 |
| CUDA/C/ABI 边界 | 检查 error code |
| 析构函数 | 不向外抛异常 |

CUDA kernel 异步执行，launch 成功不代表执行完成。必要同步点还要检查延迟暴露的错误。异常也不能直接穿过 C ABI，需要在 PyTorch/Python 边界转换。

## 延伸阅读

[详细笔记](cpp-error-handling-examples.md)保留原错误、ABI 和构建综合讲解。

## 程序实例

[阶段示例](../../examples/README.md)使用 variant 表达解析成功或失败，并用退出码通知 shell。

## 直观理解

CUDA error code 像 GPU 工位的故障编号，C++ 异常像 Host 调度链上的故障单。编号必须有人检查，故障单必须在 ABI 边界被翻译。
