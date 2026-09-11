# 程序实例

本目录恢复为阶段 1 的原始单工程结构，与 reports/stage-01 中的构建日志、源码解析和运行命令保持一致。

## 文件结构

~~~text
examples/
├── CMakeLists.txt
├── include/
│   └── runtime_demo.hpp
├── src/
│   ├── runtime_demo.cpp
│   └── main.cpp
└── tests/
    └── runtime_test.cpp
~~~

| 目标 | 输入文件 | 职责 |
| --- | --- | --- |
| stage1_core | src/runtime_demo.cpp | 解析与求和静态库 |
| stage1_demo | src/main.cpp | 命令行示例程序 |
| stage1_test | tests/runtime_test.cpp | 自动化断言程序 |

## 构建运行

~~~bash
cmake \
  -S learning/stage-01/examples \
  -B build/stage1-lifetime \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build/stage1-lifetime
ctest --test-dir build/stage1-lifetime --output-on-failure
./build/stage1-lifetime/stage1_demo 1,2,3,4
~~~

## 检查内存

~~~bash
cmake \
  -S learning/stage-01/examples \
  -B build/stage1-lifetime-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON
cmake --build build/stage1-lifetime-asan
ctest --test-dir build/stage1-lifetime-asan --output-on-failure
~~~

## 预期输出

~~~text
count=4 sum=10
status=ok
~~~

## 注释规范

后续示例必须说明文件职责、关键类型、对象所有权、成功/错误路径、边界条件、API 选择原因、测试意图，以及 Host 与 Device 行为的区别。注释必须随代码同步修改，不能让历史说明与当前目标名、路径或行为脱节。

## 变更规则

任何目录、文件名、CMake 目标或程序行为变更，都必须把下面内容视为一个整体：

```text
源码
⇒（声明目标）CMake
⇒（验证行为）测试
⇒（给出入口）README
⇒（保存证据）reports
```

提交前逐项确认：

- 源码路径和头文件包含路径存在；
- CMake 目标名与构建输出一致；
- CTest 实际执行预期测试；
- README 命令可以从仓库根目录复制运行；
- 报告中的代码、目标名、路径、输出和当前工程一致；
- 若保留历史报告，必须明确标注对应 commit，不得与当前操作指南混写。

## 报告入口

- [实验总览](../../../reports/stage-01/01-overview.md)
- [构建解析](../../../reports/stage-01/02-build.md)
- [源码解析](../../../reports/stage-01/03-code.md)
- [测试检查](../../../reports/stage-01/04-test.md)
