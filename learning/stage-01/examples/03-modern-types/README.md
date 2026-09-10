# 现代类型

这是原 `stage1_demo` 的独立项目。它用 `string_view` 观察输入，用 `variant` 表达成功或错误，用 `vector` 保存整数，用 `optional` 表达可能不存在的和。

## 运行命令

```bash
cmake -S learning/stage-01/examples -B build/stage1
cmake --build build/stage1 --target modern_types_demo modern_types_test
./build/stage1/03-modern-types/modern_types_demo 1,2,3,4
ctest --test-dir build/stage1 -R modern_types --output-on-failure
```

详细解析见[阶段报告](../../../../reports/stage-01/README.md)。报告记录的是重构前目标名 `stage1_demo/stage1_test`；源码逻辑未改变，新目标名用于区分示例。
