# 代码结构

本章先回答一个问题：一个 C++ 程序为什么分成头文件、源文件、库和可执行程序？

## 前置基础

1. **源代码**：写给编译器处理的文本。
2. **函数**：先声明“如何调用”，再定义“具体怎么做”。
3. **文件路径**：编译器需要知道源码和头文件所在位置。

## 四种角色

当前示例有四类文件：

| 文件 | 任务 |
| --- | --- |
| `runtime_demo.hpp` | 公布类型与函数接口 |
| `runtime_demo.cpp` | 实现解析和求和逻辑 |
| `main.cpp` | 提供程序入口并处理输入输出 |
| `runtime_test.cpp` | 调用接口并检查结果 |

头文件中的函数声明：

```cpp
ParseResult parse_numbers(std::string_view text);
```

它告诉调用者函数名、参数类型和返回类型。源文件中的函数定义：

```cpp
ParseResult parse_numbers(std::string_view text) {
  // 函数体
}
```

它提供真正执行的代码。声明与定义必须保持一致，否则编译或链接会失败。

## 构建过程

C++ 工程通常分两步得到程序：

```text
.cpp 源文件
⇒（编译）目标文件
⇒（链接）可执行程序
```

编译阶段分别检查每个 `.cpp`；链接阶段把多个目标文件和库中的函数定义接起来。若代码能通过语法检查，却提示找不到 `parse_numbers`，问题通常发生在链接阶段。

当前项目先把核心实现组成静态库：

```text
runtime_demo.cpp
⇒（编译归档）stage1_core
```

两个入口再分别使用它：

```text
main.cpp + stage1_core
⇒（链接）stage1_demo

runtime_test.cpp + stage1_core
⇒（链接）stage1_test
```

每个箭头右侧都标出了负责状态变化的构建操作。

## 头文件保护

同一个头文件可能沿着不同依赖路径被包含多次。`#pragma once` 要求编译器在一个编译单元中只处理它一次，防止同一内容重复展开。

`#include "runtime_demo.hpp"` 可以先理解为：在编译当前 `.cpp` 前，让接口声明对当前文件可见。它不会自动把 `.cpp` 中的函数实现链接进程序，因此 CMake 仍需写明目标之间的链接关系。

## 命名空间

`namespace stage1` 给接口增加范围。函数完整名字是：

```cpp
stage1::parse_numbers
```

`::` 表示从指定范围中寻找名字。这样以后其他模块也定义 `parse_numbers` 时，不会直接发生名字冲突。

## 程序入口

一个可执行程序从 `main` 开始：

```cpp
int main(int argc, char** argv)
```

- `argc` 保存命令行参数数量；
- `argv` 让程序访问每段参数文本；
- 返回 `0` 表示成功，非零值表示失败。

示例和测试各有一个 `main`，因此它们是两个独立程序，而不是一次运行中的两个入口。

## 程序实例

打开 [CMakeLists.txt](../../examples/CMakeLists.txt)，按以下顺序找目标：

1. `add_library` 建立 `stage1_core`；
2. `add_executable` 建立两个程序；
3. `target_link_libraries` 把程序与核心库连接；
4. `add_test` 把测试程序登记给 CTest。

## 部署说明

```bash
cmake --build build/stage1-lifetime --verbose
```

`--verbose` 会显示编译和链接命令。此处只要求能在输出中区分“编译某个 `.cpp`”和“链接最终程序”，不要求记忆每个编译参数。

## 直观理解

头文件像餐厅菜单，源文件像后厨配方，静态库像备好的中央厨房，`main` 像前台接单。菜单只说明能点什么；真正出餐还必须把前台与后厨连接起来。
