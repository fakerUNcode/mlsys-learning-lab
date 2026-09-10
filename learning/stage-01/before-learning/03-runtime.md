# 运行与构建

## 前置基础

1. 编译器把每个源文件变成目标文件。
2. 链接器把目标文件和库中的符号引用连接起来。
3. 操作系统装载程序及动态库，然后把控制权交给程序。

## 错误处理

| 情况 | 建议 | 原因 |
| --- | --- | --- |
| 构造失败、深层调用失败 | 异常 | 可跨多层传递，RAII 自动清理 |
| 失败属于普通分支 | `optional` 或结果类型 | 调用者应显式处理 |
| 系统 API、禁用异常边界 | error code | 与 C ABI 或既有协议兼容 |
| 析构函数 | 不抛异常 | 异常展开时再次抛异常会终止程序 |

不要混用“返回空值”和“抛异常”表达同一种失败。先定义接口契约：哪些失败可预期，谁负责记录上下文，谁把错误转成用户信息。

CUDA API 使用 error code，例如 `cudaMalloc` 返回 `cudaError_t`；C++ 内部可以检查它并转换成带上下文的异常；异常不能直接穿过 C ABI，PyTorch 扩展还要把 C++ 异常转换成 Python 可理解的错误。kernel launch 又是异步的，发射成功不代表执行成功，必要的同步点必须检查延迟暴露的错误。

## 编译链接

```text
main.cpp
⇒（编译）main.o
⇒（链接 CUDA 与扩展库）app
⇒（装载动态库）进程映像
⇒（运行入口）程序执行
```

符号是函数或全局对象在目标文件中的可链接名字。C++ 为支持重载会进行 name mangling；`extern "C"` 可请求 C 链接名，但不会把 C++ 类型自动变成 C ABI 安全类型。

ABI 约定二进制层面的交互规则，包括调用约定、类型布局、符号命名和异常边界。源代码能编译不代表两个不同工具链产物一定能安全混用。

PyTorch C++/CUDA extension 最终通常生成 `.so`。Python 导入它时，动态装载器需要解析 PyTorch、C++ 标准库和 CUDA 相关符号；编译器 ABI、PyTorch 构建方式或 CUDA 库版本不匹配，都可能表现为 `undefined symbol`，而不是 Python 语法错误。

动态库在 Linux 常见为 `.so`。排查命令：

```bash
nm -C ./build/stage1_demo
ldd ./build/stage1_demo
readelf -d ./build/stage1_demo
```

`nm -C` 查看并反修饰 C++ 符号；`ldd` 查看运行时解析到的动态库；`readelf` 直接读取 ELF 元数据。不要对不可信二进制运行 `ldd`。

## CMake测试

CMake 描述目标和依赖，不是编译器。CTest 负责登记与运行测试。Sanitizer 是编译器插桩，用来发现内存和未定义行为问题。

阶段 1 先用纯 C++ 示例掌握工具链。进入 CUDA 阶段后，同一构建结构会增加 CUDA language、`.cu` 源文件和 GPU 架构选项；Host 内存问题用 ASan/TSan，Device 越界和竞争则需要 Compute Sanitizer 等 CUDA 工具。

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Sanitizer 会改变内存布局和速度，因此用于诊断，不用于性能数据。GCC/Clang 的 AddressSanitizer 与 UndefinedBehaviorSanitizer 可一起使用；ThreadSanitizer 通常另开一次构建。

## 程序实例

完整工程见[程序实例](../examples/README.md)。它把实现放在库目标，把入口放在可执行目标，并通过 CTest 检查解析和所有权行为。

## 部署说明

需要 CMake 3.16+ 和支持 C++17 的 GCC 或 Clang。先普通构建确认功能，再开 Sanitizer 构建。发布动态库时应控制导出符号，并明确编译器、标准库和 ABI 兼容范围。

## 直观理解

Python 调用 CUDA 算子像启动一条跨厂生产线：`.so` 是装好的设备，ABI 是接口规格，链接器接通设备，CUDA 错误码报告 GPU 工位故障，C++ 异常把故障沿 Host 调度链送回用户。
