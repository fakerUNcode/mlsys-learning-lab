# 模板特性

## 前置基础

1. 类型决定数据表示和可用操作。
2. 模板参数在编译期被具体类型替换。
3. traits 在编译期描述类型性质。

## 主线关系

CUDA kernel 常为 `float`、`half` 等 dtype 复用同一算法。模板生成类型对应的代码；traits 选择累加类型、向量宽度或支持路径。

```cpp
template <class T>
T twice(T value) {
  static_assert(std::is_arithmetic_v<T>);
  return value + value;
}
```

`static_assert` 在编译期拒绝不符合约束的类型，不会等到 GPU 运行时才报错。

## 程序实例

本节暂时使用讲义代码。进入 CUDA 阶段后，将为同一 elementwise kernel 增加多 dtype 实例。

## 部署说明

本节暂时没有独立程序，避免提供无法运行的占位命令。当前统一验证入口是[示例导航](../../examples/README.md)；进入多 dtype CUDA kernel 时再补模板工程。

## 直观理解

模板像不同 dtype 共用的算子模具，traits 像入模检查表。模具只写一次，编译器按 float、half 等材料生成不同成品。
