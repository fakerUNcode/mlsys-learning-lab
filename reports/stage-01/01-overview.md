# 实验总览

## 实验目标

本次实验验证以下链路：

```text
C++ 源文件
⇒（CMake 配置）构建规则
⇒（GNU C++ 编译）目标文件
⇒（链接）静态库与可执行文件
⇒（CTest 调用）单元测试
⇒（ASan/UBSan 插桩）内存与未定义行为检查
```

同时观察三种程序结果：正常输入、非法输入和自动化测试。

## 前置基础

1. **源文件**保存 C++ 实现，编译后产生目标文件。
2. **链接**把目标文件和库组合成可执行程序。
3. **进程退出码**用整数告诉调用者程序成功还是失败；通常 `0` 表示成功，非零值表示失败。

## 实验环境

| 项目 | 本次记录 |
| --- | --- |
| 项目路径 | `/home/alibaba/projects/mlsys-learning-lab` |
| C++ 编译器 | GNU C++ 13.3.0 |
| C++ 标准 | C++17 |
| 普通构建 | `build/stage1-lifetime` |
| 检查构建 | `build/stage1-lifetime-asan`（即：加了“运行时错误检测”的版本） |
| Sanitizer（C/C++ 的运行时“错误探测器”） | AddressSanitizer（检查**越界访问、释放后继续使用、重复释放**等问题）、UndefinedBehaviorSanitizer（检查一些 C++ 中“不允许但可能仍能运行”的操作，例如**有符号整数溢出、非法移位**等） |
| GPU | 未使用 |
| CUDA | 未调用 |

终端输出不记录 CMake 的版本、操作系统版本和 Git commit。正式性能或兼容性实验应额外保存：

```bash
cmake --version
c++ --version
git rev-parse HEAD
bash scripts/check_environment.sh --purpose=stage1-lifetime
```

## 构建命令

普通 Debug 构建：注意此时它**还没有真正开始编译 C++**，而是在告诉 CMake：“请帮我准备好怎么构建”。

```bash
cmake \
  -S learning/stage-01/examples \
  -B build/stage1-lifetime \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build/stage1-lifetime
```

参数含义：

```
cmake
  │
  ├─ -S learning/stage-01/examples
  │       ↑
  │       源代码在哪
  │       这里通常有 CMakeLists.txt
  │
  ├─ -B build/stage1-lifetime
  │       ↑
  │       构建产生的中间文件、最终程序放哪里
  │
  └─ -DCMAKE_BUILD_TYPE=Debug
          ↑
          使用 Debug（调试）模式
```

源码与构建产物分离，所以不需要把临时目标文件写进 `learning/`。删除构建目录也不会删除源码。

### CMakeLists.txt

可以把 `CMakeLists.txt` 理解成：

> **项目的“构建说明书”——告诉 CMake：有哪些源码、要生成什么、怎么编译、依赖什么。**

一个最简单的例子：

```
cmake_minimum_required(VERSION 3.16)

project(MyApp)

set(CMAKE_CXX_STANDARD 17)

add_executable(my_app
    main.cpp
    model.cpp
)
```

逐句就是：

```
cmake_minimum_required(...) → 至少需要哪个版本的 CMake

project(MyApp)              → 项目叫什么

CMAKE_CXX_STANDARD 17       → 使用 C++17

add_executable(...)         → 要生成一个可执行程序
   │
   ├─ 名字：my_app
   │
   └─ 源码：main.cpp、model.cpp
```

稍复杂一点还会规定：

```
target_include_directories(my_app PRIVATE include)
# 去哪里找 .h 头文件

target_link_libraries(my_app PRIVATE pthread)
# 最终程序需要链接哪些库

target_compile_options(my_app PRIVATE -Wall)
# 给编译器添加什么选项
```

## 构建结果

CMake 成功识别编译器：

```text
The CXX compiler identification is GNU 13.3.0
Detecting CXX compiler ABI info - done
Check for working CXX compiler: /usr/bin/c++ - skipped
Detecting CXX compile features - done
```

随后完成三个目标：

```text
stage1_core
⇒（归档）libstage1_core.a

stage1_demo
⇒（链接 stage1_core）可执行程序

stage1_test
⇒（链接 stage1_core）测试程序
```

`[16%]` 到 `[100%]` 是构建任务进度，不是程序运行进度，也不是测试覆盖率。

## 测试结果

运行命令：

```bash
ctest \
  --test-dir build/stage1-lifetime \
  --output-on-failure
```

结果：

```text
1/1 Test #1: stage1_test ... Passed
100% tests passed, 0 tests failed out of 1
```

这证明 CTest 登记的唯一测试进程以成功状态结束。它不证明所有代码路径均被覆盖，也不等于“程序没有任何潜在缺陷”。

## 正常路径

命令：

```bash
./build/stage1-lifetime/stage1_demo 1,2,3,4
```

结果：

```text
count=4 sum=10
status=ok
```

数据变化按程序操作展开：

```text
"1,2,3,4"
⇒（按逗号切分）"1"、"2"、"3"、"4"
⇒（整数解析）1、2、3、4
⇒（压入 vector）[1, 2, 3, 4]
⇒（accumulate 求和）10
```

元素数量为：
$$
N
\overset{\text{统计元素}}{=}
4
$$
累加过程为：

$$
s_0
\overset{\text{初值}}{=}
0
$$

$$
s_1
\overset{\text{加入 1}}{=}
s_0+1
$$

$$
s_2
\overset{\text{加入 2}}{=}
s_1+2
$$

$$
s_3
\overset{\text{加入 3}}{=}
s_2+3
$$

$$
s_4
\overset{\text{加入 4}}{=}
s_3+4
$$

$$
s_4
\overset{\text{完成累加}}{=}
10
$$



## 符号说明

- \(N\)：解析后的整数数量。
- \(s_i\)：加入第 \(i\) 个元素后的累计值。
- 下标 \(i\)：当前已经处理的元素序号。

## 错误路径

命令：

```bash
./build/stage1-lifetime/stage1_demo 1,x,3
```

结果：

```text
status=invalid integer
```

运行过程在 `x` 处停止：

```text
"1,x,3"
⇒（解析首段）整数 1
⇒（解析第二段）发现 x 不是完整整数
⇒（返回错误分支）输出 invalid integer
⇒（main 返回 1）通知 shell 失败
```

随后执行 `echo $?` 得到 `1`。`$?` 表示上一条前台命令的退出码；它不是 C++ 异常编号，也不是解析失败元素的下标。

## 检查结果

开启 `ENABLE_SANITIZERS=ON` 后，项目再次成功构建，CTest 结果为：

```text
1/1 Test #1: stage1_test ... Passed
100% tests passed, 0 tests failed out of 1
```

这说明当前测试执行到的路径没有被 ==ASan== （之前提到的AddressSanitizer，内存错误检查器）或 ==UBSan==（UndefinedBehaviorSanitizer，非法行为检查器） 检测出问题。检查版用时从显示的 `0.00 sec` 变为 `0.01 sec`，但样本太少且显示精度有限，不能据此计算 Sanitizer 的性能开销。

## 实验结论

本次已经由运行结果支持的结论：

- CMake 能使用 GNU C++ 13.3.0 构建该 C++17 工程；
- 静态库、示例程序和测试程序均链接成功；
- 正常输入得到正确数量与和；
- 非法整数得到错误文本和退出码 `1`；
- 当前测试路径通过 ASan 与 UBSan 检查；
- 局部对象能在退出路径中自动进入析构过程。

本次尚未证明：

- 所有整数输入都能正确处理；
- 整数求和不会溢出；
- 测试覆盖所有分支；
- 多线程下不存在数据竞争；
- GPU buffer、CUDA Stream 或 Event 的生命周期正确；
- 程序具备可比较的性能优势。

## 直观理解

这次实验像验收一条小型流水线：CMake 编排工位，编译器加工零件，链接器完成组装，CTest 投放固定样品，Sanitizer 检查加工时是否踩出界。样品通过说明这条路径可用，不代表所有原料都已验证。
