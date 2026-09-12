# 测试检查

## 前置基础

1. 测试用确定输入检查实际行为是否满足预期。
2. Sanitizer 通过只说明本次执行到的路径未触发其可检测问题。
3. 测试退出码为 `0` 时，CTest 把该测试记录为通过。

## 测试源码

完整文件位于 [`runtime_test.cpp`](../../learning/stage-01/examples/00-runtime-foundations/tests/runtime_test.cpp)。测试程序使用标准 `assert`，没有引入第三方测试框架。

### 成功解析

```cpp
auto parsed = stage1::parse_numbers("2,3,5");
```

预期 `parsed` 保存 `vector<int>`。随后检查：

```cpp
assert(
    std::holds_alternative<std::vector<int>>(parsed)
);
```

`holds_alternative<T>` 返回 variant 当前是否持有类型 `T`。若条件为假，启用 assert 的构建会报告失败并终止测试进程。

### 取得数据

```cpp
const auto& values =
    std::get<std::vector<int>>(parsed);
```

这里使用 `const&`，因此不会复制 vector，也不会移动它。引用的有效期依赖 `parsed`：只要 `parsed` 未销毁、未切换候选且未发生使引用失效的修改，`values` 才有效。

前一条断言先确认候选类型，随后 `std::get` 不应抛出 `std::bad_variant_access`。

### 求和断言

```cpp
assert(stage1::sum_if_not_empty(values) == 10);
```

左侧是 `optional<int>`，右侧是整数。C++17 为 optional 提供相应比较，只有 optional 有值且值等于 `10` 时条件成立。

计算过程：

\[
s
\overset{\text{累加输入}}{=}
2+3+5
\]

\[
s
\overset{\text{完成加法}}{=}
10
\]

## 符号说明

- \(s\)：三个测试整数的累计和。

### 非法输入

```cpp
auto invalid = stage1::parse_numbers("2,x");
```

首段 `2` 会先成功进入局部 vector，第二段 `x` 触发错误。函数返回时局部 vector 自动析构。

```cpp
assert(
    std::get<std::string_view>(invalid) ==
    "invalid integer"
);
```

该断言同时检查 variant 是错误候选，并检查错误文本。如果 variant 实际是 vector，`std::get<string_view>` 会抛出异常，测试也不会正常通过。

### 空集求和

```cpp
assert(!stage1::sum_if_not_empty({}).has_value());
```

`{}` 构造临时空 `vector<int>`，它通过 `const&` 绑定到函数参数。函数返回后，临时 vector 在完整表达式结束时析构。

`has_value()` 对 `nullopt` 返回 `false`，逻辑非运算后得到 `true`，所以断言通过。

## 断言边界

标准 `assert` 受 `NDEBUG` 控制。若 Release 配置定义了 `NDEBUG`，断言可能被编译掉，测试即使没有真正检查内容也可能返回成功。

本次使用 `CMAKE_BUILD_TYPE=Debug`，适合运行这些断言。开源项目后续更适合引入测试框架，或使用不会在 Release 中消失的显式检查。

当前测试覆盖：

| 行为 | 状态 |
| --- | --- |
| 多个合法整数 | 已覆盖 |
| 非法字符 | 已覆盖 |
| 空 vector 求和 | 已覆盖 |
| 空字符串解析 | 未覆盖 |
| 单个整数 | 未覆盖 |
| 负数 | 未覆盖 |
| 连续逗号 | 未覆盖 |
| 末尾逗号 | 未覆盖 |
| 整数越界 | 未覆盖 |
| 求和溢出 | 未覆盖 |

## CTest流程

运行：

```bash
ctest \
  --test-dir build/stage1-lifetime \
  --output-on-failure
```

执行关系：

```text
CTest 读取测试登记
⇒（找到 stage1_test）启动测试进程
⇒（执行所有 assert）进程返回 0
⇒（记录退出状态）Test Passed
```

`--output-on-failure` 表示成功时保持输出简洁，失败时显示测试进程输出。它不会让测试失败，也不会开启更多测试。

## ASan检查

AddressSanitizer 主要用于发现执行路径中的 Host 内存问题，例如：

- heap buffer overflow；
- stack buffer overflow；
- use-after-free；
- double free；
- 部分内存泄漏。

本次构建参数：

```text
-fsanitize=address,undefined
-fno-omit-frame-pointer
```

如果 `unique_ptr` 释放 vector 后代码继续通过旧裸指针访问它，ASan 通常能够报告 use-after-free。当前代码没有保留这样的指针。

ASan 不用于检查 CUDA Device 内存越界。未来 GPU kernel 的越界需要使用 NVIDIA Compute Sanitizer 等工具。

## UBSan检查

UndefinedBehaviorSanitizer 用于发现执行路径中的部分未定义行为，例如：

- 有符号整数溢出；
- 非法移位；
- 某些未对齐访问；
- 部分无效类型或空指针操作。

本次测试数据很小，所以没有触发 `std::accumulate` 的整数溢出。UBSan 通过不代表任意长输入都不会溢出。

## 未用TSan

本项目没有启动线程，因此本次未启用 ThreadSanitizer。即使同时打开 ASan 和 UBSan，也不等于检查了数据竞争。

以后实现线程池、任务队列或并发模型缓存时，应单独建立 TSan 构建。ASan 与 TSan 通常不放在同一次构建中。

## 结果解释

普通测试耗时显示 `0.00 sec`，检查版显示 `0.01 sec`。不能据此说 Sanitizer “只增加 0.01 秒”或“慢了固定倍数”，原因包括：

- CTest 输出时间经过舍入；
- 只有一次极短测试；
- 进程启动成本占比较大；
- 没有重复采样；
- 测试目标是正确性，不是 benchmark。

性能测量应该放入 `benchmarks/`，并明确 warmup、重复次数和统计方法。

## 手动检查

可继续运行以下输入，但它们目前不是自动化测试：

```bash
./build/stage1-lifetime/stage1_demo
./build/stage1-lifetime/stage1_demo ""
./build/stage1-lifetime/stage1_demo -3,0,7
./build/stage1-lifetime/stage1_demo 12x
./build/stage1-lifetime/stage1_demo 1,
```

每次运行后立刻执行：

```bash
echo $?
```

用来观察输出与进程状态是否一致。不要连续运行其他命令后再查看 `$?`，否则得到的是最近那条命令的退出码。

## 排错顺序

如果构建失败，按以下层次定位：

1. 查看第一条编译或链接错误；
2. 确认当前目录是仓库根目录；
3. 确认 `-S` 指向示例源码；
4. 用新的构建目录排除旧 cache；
5. 检查 GCC/CMake 版本；
6. 只在需要时查看完整 verbose 命令。

查看详细构建命令：

```bash
cmake --build build/stage1-lifetime --verbose
```

若测试失败：

```bash
ctest \
  --test-dir build/stage1-lifetime \
  --output-on-failure \
  --verbose
```

若 ASan 报错，应从报告中的错误类型、首次无效访问和分配/释放调用栈开始阅读，不要只处理最后一行。

## 练习测评

以下题目不提供答案。作答后再逐步批改和评分。

### 练习一

输入 `1,2,` 时，按现有 `parse_numbers` 的每一轮 `text`、`token` 和 `comma` 状态推演结果。判断它是否把末尾空 token 当作错误，并说明依据。

### 练习二

设计一个新测试来触发 `sum_if_not_empty` 的整数溢出风险。只写测试思路、输入选择和预期 Sanitizer 行为，不要先修改业务代码。

## 直观理解

单元测试像按清单检查几种指定货物，ASan 和 UBSan 像流水线传感器：货物经过危险位置时会报警。但没送上流水线的输入不会被检查，GPU 车间还需要自己的 Compute Sanitizer。
