# 构建测试

## 前置基础

1. CMake 描述目标和依赖。
2. CTest 启动测试并检查退出状态。
3. Sanitizer 为 Host 代码插入运行时检查。

## 工具边界

| 工具 | 作用 |
| --- | --- |
| CMake | 生成构建规则 |
| 编译器 | 编译和链接 C++ |
| CTest | 登记并运行测试 |
| ASan | 检查 Host 内存问题 |
| UBSan | 检查部分未定义行为 |
| TSan | 检查 Host 数据竞争 |

ASan 不能检查 CUDA Device 越界；GPU 代码后续使用 Compute Sanitizer。Sanitizer 会改变性能，不用于 benchmark 数据。

## 程序实例

顶层[示例导航](../../examples/README.md)可一次构建三个独立工程并运行全部测试。

## 部署说明

```bash
cmake \
  -S learning/stage-01/examples \
  -B build/stage1-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON
cmake --build build/stage1-asan
ctest --test-dir build/stage1-asan --output-on-failure
```

## 直观理解

CMake 是装配清单，编译器是生产线，CTest 是验收员，Sanitizer 是故障传感器。GPU 车间使用另一套检测设备，不能拿 Host 检查结果代替。
