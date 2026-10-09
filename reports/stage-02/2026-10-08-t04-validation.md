# T04 Buffer 构建与内存检查报告

日期：2026-10-08。结论：实现、构建、运行与禁止拷贝检查通过；工程验证范围见下文。

## 来源与环境

- 源码：[buffer.hpp](../../learning/stage-02/t04-cpp-buffer-ownership/buffer.hpp)、[buffer.cpp](../../learning/stage-02/t04-cpp-buffer-ownership/buffer.cpp)、[CMakeLists.txt](../../learning/stage-02/t04-cpp-buffer-ownership/CMakeLists.txt)。
- 本轮沿用工作区已有实现与验证，补齐源码中的语法、行为和原理注释并重新运行检查；核读时发现 `buffer.cpp` 文件尾部停在空状态检查中间，恢复 moved-from 复用、vector 迁移、异常展开和 `main()`，随后按下方命令重新构建验证。
- GCC：`g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`。
- CMake：3.28.3；C++17；Debug；CPU float 数组；确定性输入，无随机种子或 GPU 依赖。
- 编译告警：`-Wall -Wextra -Wpedantic`；检查：`-fsanitize=address,undefined -fno-omit-frame-pointer`。

## 复现

从仓库根目录：

```bash
cmake -S learning/stage-02/t04-cpp-buffer-ownership -B /tmp/mlsys-t04-cpp-buffer-build \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build /tmp/mlsys-t04-cpp-buffer-build
ctest --test-dir /tmp/mlsys-t04-cpp-buffer-build --output-on-failure -V
```

CTest 的环境为 `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` 和 `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`。

实际构建与测试退出码均为 0，无编译告警或 Sanitizer 报错。程序输出：

```text
PASS ownership chain, initialization and observer
PASS self move
PASS empty transitions and moved-from reuse
PASS vector relocation
PASS exception cleanup
T04 runtime checks passed
```

CTest 汇总：

```text
1/1 Test #1: t04_buffer_ownership .............   Passed
100% tests passed, 0 tests failed out of 1
```

目录整理后于 2026-10-09 使用上方新源码路径和 `/tmp/mlsys-t04-cpp-buffer-build` 重新配置、构建并运行 CTest；结果仍为 1/1 通过，ASan/UBSan 无报错。

本轮注释更新后重新运行上述命令，CTest 1/1 通过（报告运行时间 0.01 秒），两项拷贝负例重新检查也通过。该时间仅是验证程序运行记录，不是性能指标。

练习 README 中包含 `a→b→c` 的所有权与释放路径、`std::move` 的作用，以及默认移动留下非零来源长度的问题，可用于自测。

## 覆盖与限制

| 检查 | 证据 |
|---|---|
| 初始化与移动构造 | 初始值全零；原数组指针与写入值保留；来源指针为空且长度为零 |
| 覆盖非空目标的移动赋值 | `c(2)` 接管原 4 元素数组；来源为空；退出后泄漏检查无报告 |
| 自移动 | 通过同一对象引用赋值；指针、长度与数值不变 |
| 空状态转换 | 零长度构造、空→空移动、空目标接非空、非空目标接空均通过 |
| 移动后复用 | 来源重新接收新 Buffer 后可正常访问 |
| 标准容器 | `vector<Buffer>` 显式增大 capacity，数组指针与数值保留 |
| 异常路径 | 分配后抛异常并捕获，作用域清理后泄漏检查无报告 |
| 类型性质 | 静态断言确认不可拷贝、可 noexcept 移动及 const 访问类型 |

检查只涵盖实际执行的 CPU 路径。这里没有对裸指针越界、释放后访问或并发访问提供运行时保护；调用方需遵守指针生命周期。无性能指标或 GPU 资源释放结论。

## 预期编译失败

本轮分别将以下两个片段传给 GCC 的 `-fsyntax-only` 检查。两项均以非零退出码拒绝，诊断包含 `use of deleted function`：

```cpp
#include "buffer.hpp"
int main() { stage2::Buffer a(4); stage2::Buffer b(a); }
```

```cpp
#include "buffer.hpp"
int main() { stage2::Buffer a(4); stage2::Buffer b(1); b = a; }
```

复现方式为分别保存片段到临时 `.cpp` 文件，执行：

```bash
g++ -std=c++17 -I learning/stage-02/t04-cpp-buffer-ownership \
  -fsyntax-only /tmp/t04-copy-check.cpp
```

这两项要求编译失败，与正常验证程序构建成功一起说明独占契约。
