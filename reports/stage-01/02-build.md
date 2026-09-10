# 构建解析

整个构建关系可以浓缩成：

```
runtime_demo.cpp
      ↓
 stage1_core 库
    ↙        ↘
main.cpp   runtime_test.cpp
   ↓            ↓
stage1_demo   stage1_test
                  ↓
                CTest
```

其中最关键的三个概念是：`add_library` 创建库，`add_executable` 创建可执行程序，`target_link_libraries` 把“程序”和“库”连接起来。

## 前置基础

1. 一个 `.cpp` 文件通常先独立编译成 `.o` 目标文件。
2. 静态库是若干.o目标文件的归档，Linux 下常以 `.a` 结尾。
3. 可执行文件需要通过链接解析自己使用的函数符号。

## 目标关系

本项目的构建关系为：

```text
runtime_demo.cpp
⇒（编译）runtime_demo.cpp.o
⇒（归档）libstage1_core.a
                 ↙       ↘
     （链接库）stage1_demo  stage1_test（链接库）
```

头文件 `runtime_demo.hpp` 不单独产生目标文件。它会被预处理器展开到引用它的 `.cpp` 文件中，为编译器提供声明。

# CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)
project(stage1_runtime LANGUAGES CXX)

option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

add_library(stage1_core src/runtime_demo.cpp)
target_include_directories(stage1_core PUBLIC include)
target_compile_features(stage1_core PUBLIC cxx_std_17)

add_executable(stage1_demo src/main.cpp)
target_link_libraries(stage1_demo PRIVATE stage1_core)

add_executable(stage1_test tests/runtime_test.cpp)
target_link_libraries(stage1_test PRIVATE stage1_core)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target stage1_core stage1_demo stage1_test)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

enable_testing()
add_test(NAME stage1_test COMMAND stage1_test)

```



## 最低版本

```cmake
cmake_minimum_required(VERSION 3.16)
```

这行表示项目要求 CMake 3.16 或更高版本。它不表示使用 C++16，也不表示 GCC 版本。

## 项目声明

```cmake
project(stage1_runtime LANGUAGES CXX)
```

- `stage1_runtime` 是 CMake 项目名；
- `LANGUAGES CXX` 只启用 C++；
- 本工程没有启用 `CUDA` language，因此不会调用 `nvcc`。

如果以后加入 `.cu` 文件，项目配置需要显式处理 CUDA，而不是仅把后缀名改为 `.cu`。

## 构建选项

```cmake
option(
  ENABLE_SANITIZERS ← ① 开关的名字
  "Enable address and undefined behavior sanitizers" ← ② 给人看的说明文字
  OFF               ← ③ 默认关闭
)
```

实际文件把它写在一行，展开后包含三个信息：选项名称、帮助文字和默认值。普通构建没有传这个选项，因此值是 `OFF`；检查构建传入 `-DENABLE_SANITIZERS=ON`，值变为 `ON`。

`-D` 在这里是向 CMake cache 定义变量，不是直接向 C++ 代码定义宏。

## 核心库

```cmake
add_library(stage1_core src/runtime_demo.cpp)
```

拆开看：

```
add_library(...)           → 创建一个“库”
     │
     ├─ stage1_core        → 给这个库起名
     │
     └─ src/runtime_demo.cpp
                            → 用这个 C++ 源文件制作这个库
```

也就是说：

> **把 `runtime_demo.cpp` 里实现的 `parse_numbers()` 和 `sum_if_not_empty()` 编译并打包成一个叫 `stage1_core` 的库。**

这次实际构建输出是：

```
Linking CXX static library libstage1_core.a
```

所以最终关系是：

```
src/runtime_demo.cpp
        ↓ 编译
runtime_demo.o
        ↓ 打包
libstage1_core.a
```

之后真正的可执行程序可以链接它：

```
main.cpp
   ↓
可执行程序
   ↑
libstage1_core.a
```

## 头文件路径

```cmake
target_include_directories(stage1_core PUBLIC include)
```

这使编译器可以根据：

```cpp
#include "runtime_demo.hpp"
```

找到 `include/runtime_demo.hpp`。

`PUBLIC` 同时表达两层需求：

```text
stage1_core 自己编译
⇒（使用 PUBLIC include）能找到头文件

stage1_demo 链接 stage1_core
⇒（传播使用要求）也能找到同一头文件
```

如果使用 `PRIVATE`，包含路径只服务于库自身，不会自动传播给使用该库的目标。

## 语言标准

```
target_compile_features(stage1_core PUBLIC cxx_std_17)
```

拆开看：

```
target_compile_features(...) → 规定目标需要的编译特性
        │
        ├─ stage1_core       → 设置核心库
        ├─ PUBLIC            → 自己使用，并把要求传给依赖它的目标
        └─ cxx_std_17        → 要求 C++17
```

也就是说：

> **`stage1_core` 使用了 C++17 特性，因此要求按 C++17 编译；`PUBLIC` 还会把这个要求传给 `stage1_demo` 和 `stage1_test`。**

------

## 示例目标

```
add_executable(stage1_demo src/main.cpp)
target_link_libraries(stage1_demo PRIVATE stage1_core)
```

拆开看：

```
add_executable(...)          → 创建可执行程序
        │
        ├─ stage1_demo       → 程序名称
        └─ src/main.cpp      → 程序的源文件

target_link_libraries(...)   → 给程序链接所需的库
        │
        ├─ stage1_demo       → 哪个程序需要库
        ├─ PRIVATE           → 只表示它自身需要
        └─ stage1_core       → 要链接的核心库
```

也就是说：

> **用 `main.cpp` 生成 `stage1_demo` 程序，再链接 `stage1_core`，使程序能使用核心库里的 `parse_numbers()`、`sum_if_not_empty()` 等实现。**



## 测试目标

```cmake
add_executable(stage1_test tests/runtime_test.cpp)
target_link_libraries(stage1_test PRIVATE stage1_core)
```

拆开看：

```text
add_executable(...)          → 创建测试可执行程序
        │
        ├─ stage1_test       → 测试程序名称
        └─ runtime_test.cpp  → 测试代码

target_link_libraries(...)   → 给测试程序链接库
        │
        ├─ stage1_test       → 哪个程序需要库
        ├─ PRIVATE           → 只有它自身需要
        └─ stage1_core       → 被测试的核心库
```

也就是说：

> **把 `runtime_test.cpp` 做成独立测试程序 `stage1_test`，再链接 `stage1_core`，从而测试核心库里的函数。**

# 检查和循环

```cmake
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")  # 如果开启了 Sanitizer，并且当前 C++ 编译器是 GNU 或 Clang，就进入下面的配置

  foreach(target stage1_core stage1_demo stage1_test)  # 依次对 stage1_core、stage1_demo 和 stage1_test 三个目标执行下面的设置

    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)  # 编译时给当前目标开启 ASan、UBSan，并保留栈帧信息方便定位错误

    target_link_options(${target} PRIVATE -fsanitize=address,undefined)  # 链接时也加入 ASan 和 UBSan 所需的运行库参数

  endforeach()  # 结束 foreach 循环

endif()  # 结束 Sanitizer 条件判断
```

## 检查条件

```cmake
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
```

拆开看：

```text
if(...)                     → 满足条件才执行后面的配置
   │
   ├─ ENABLE_SANITIZERS     → Sanitizer 开关必须为 ON
   ├─ AND                   → 两个条件都必须满足
   ├─ CMAKE_CXX_COMPILER_ID → 后面跟当前 C++ 编译器的类型
   └─ MATCHES "GNU|Clang"   → 编译器必须是 GNU 或 Clang
```

也就是说：

> **只有“开启 Sanitizer”并且“使用 GNU/Clang 编译器”时，才加入后面的检查参数。**

------

## 目标循环

```cmake
foreach(target stage1_core stage1_demo stage1_test)
```

拆开看：

```text
foreach(...)          → 对多个目标重复执行相同操作
      │
      ├─ target       → 临时变量名
      ├─ stage1_core  → 核心库
      ├─ stage1_demo  → 示例程序
      └─ stage1_test  → 测试程序
```

每轮 `${target}` 分别代表：

```text
第 1 次 → stage1_core
第 2 次 → stage1_demo
第 3 次 → stage1_test
```

也就是说，接下来的操作要给这三个target都操作一遍



## 编译插桩

**给核心库、示例程序、测试程序三个目标都配置 Sanitizer，而不是只检查最终程序。**

```cmake
target_compile_options(
  ${target}
  PRIVATE
  -fsanitize=address,undefined
  -fno-omit-frame-pointer
)
```

拆开看：

```text
target_compile_options(...)    → 给编译器添加参数
        │
        ├─ ${target}           → 当前循环处理的目标
        ├─ PRIVATE             → 参数只用于当前目标
        ├─ -fsanitize=address  → 开启 ASan 内存检查
        ├─ undefined           → 开启 UBSan 未定义行为检查
        └─ -fno-omit-frame-pointer
                               → 保留栈帧信息，方便定位报错位置
```

也就是说：

> **编译三个目标时加入 ASan、UBSan 检查代码，并保留更容易阅读的错误调用栈。**

## 链接插桩

```cmake
target_link_options(
  ${target}
  PRIVATE
  -fsanitize=address,undefined
)
```

拆开看：

```text
target_link_options(...)     → 给链接阶段添加参数
        │
        ├─ ${target}         → 当前处理的目标
        ├─ PRIVATE           → 只作用于当前目标
        └─ -fsanitize=...    → 链接 ASan、UBSan 所需支持
```

也就是说：

> **编译时加入检查代码后，链接阶段也要加入 Sanitizer 参数，使最终程序能连接所需的 Sanitizer 运行库。**

------

## 测试登记

```cmake
enable_testing()
add_test(NAME stage1_test COMMAND stage1_test)
```

拆开看：

```text
enable_testing()         → 开启 CMake/CTest 测试功能

add_test(...)            → 向 CTest 注册一个测试
    │
    ├─ NAME stage1_test  → 测试名称叫 stage1_test
    └─ COMMAND stage1_test
                         → 测试时运行 stage1_test 程序
```

所以运行：

```bash
ctest
```

大致就是：

```text
CTest
  ↓
找到注册的 stage1_test
  ↓
运行 stage1_test 可执行程序
  ↓
获取运行结果
  ↓
Passed / Failed
```

也就是说：

> **`stage1_test` 是真正执行测试逻辑的程序；CTest 负责找到它、运行它并报告成功或失败。**

------

## 构建日志

构建时可能看到：

```text
Building CXX object
Linking CXX static library
Linking CXX executable
Built target
```

分别理解成：

```text
Building CXX object
→ 正在把 .cpp 编译成 .o

Linking CXX static library
→ 正在把 .o 打包成静态库 .a

Linking CXX executable
→ 正在链接生成可执行程序

Built target
→ 这个构建目标已经完成
```

因此整个过程就是：

```text
runtime_demo.cpp
      ↓ Building CXX object
runtime_demo.o
      ↓ Linking CXX static library
libstage1_core.a
      ↓
      ├── Linking CXX executable → stage1_demo
      └── Linking CXX executable → stage1_test
```

另外，`Configuring done` 和 `Generating done` 只是说明 **CMake 配置和生成构建文件完成**，并不表示 C++ 已经编译完成。

------

## 产物观察

```bash
find build/stage1-lifetime -maxdepth 2 -type f | sort
file build/stage1-lifetime/stage1_demo
nm -C build/stage1-lifetime/libstage1_core.a
```

拆开理解：

```text
find
→ 看构建目录生成了哪些文件

file stage1_demo
→ 查看 stage1_demo 是什么类型的文件

nm -C libstage1_core.a
→ 查看静态库里有哪些 C++ 函数符号
```

例如最后一个命令可以用来确认库里是否真的存在：

```text
stage1::parse_numbers
stage1::sum_if_not_empty
```

也就是说：

> **这几个命令是从构建结果反过来检查：程序生成了吗？静态库生成了吗？库里的函数真的存在吗？**

------

## GPU 关联

当前项目还是纯 C++：

```text
.cpp
 ↓
C++ 编译器
 ↓
目标文件
 ↓
链接
 ↓
程序
```

以后进入 CUDA 后，会多出 GPU 代码：

```text
main.cpp
   ↓ Host 编译器
Host 代码产物
        ↘
          最终链接 → 程序
        ↗
kernel.cu
   ↓ CUDA 工具链
GPU/Device 代码产物
```

也就是说：

> **CUDA 会让构建链增加 GPU Device 代码的编译和 CUDA 运行库链接，但现在学习的“目标、库、依赖、链接”这些 CMake 概念仍然适用。**

最后可以把整个 `CMakeLists.txt` 记成一张关系图：

```text
CMakeLists.txt
│
├─ project()                → 这是什么项目
├─ option()                 → 定义构建开关
│
├─ add_library()            → 创建 stage1_core
│   ├─ include 路径
│   └─ C++17
│
├─ add_executable()         → 创建 stage1_demo
│   └─ 链接 stage1_core
│
├─ add_executable()         → 创建 stage1_test
│   └─ 链接 stage1_core
│
├─ if + foreach             → 给三个目标加入 Sanitizer
│
└─ add_test()               → 把 stage1_test 交给 CTest 管理
```

这就是这份 `CMakeLists.txt` 从**源码 → 库 → 程序 → 测试 → 错误检查**的完整构建逻辑。
