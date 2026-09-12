# 生命周期

本章只讨论资源对象如何创建、转移和销毁。

## 阅读顺序

1. [RAII](01-raii-resource-management.md)
2. [拷贝移动](02-copy-and-move-ownership-transfer.md)
3. [生命周期、拷贝与移动逐段讲解](object-lifetime-copy-move-explained.md)：保留原有逐句说明。

## 程序实例

运行[对象生命周期示例](../../examples/00-runtime-foundations/README.md)，观察 `variant` 中的 vector 如何移动到 `unique_ptr`，以及局部对象离开 `main()` 后如何自动清理。
