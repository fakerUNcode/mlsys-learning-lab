# 源码解析

## 前置基础

1. **声明**告诉编译器某个名字及其类型，**定义**提供实际实现。
2. `const T&` 可以只读访问对象而不复制它。
3. RAII 让对象析构自动触发其成员和所持资源的清理。

## 文件关系

```text
runtime_demo.hpp
├── ParseResult 类型别名
├── parse_numbers 声明
└── sum_if_not_empty 声明

runtime_demo.cpp
└── 两个函数的定义

main.cpp
└── 用户输入与输出

runtime_test.cpp
└── 自动化断言
```

## 头文件

完整文件现位于 [`runtime_demo.hpp`](../../learning/stage-01/examples/03-modern-types/include/runtime_demo.hpp)。

### 单次包含

```cpp
#pragma once
```

同一个头文件可能经由多条包含路径进入一个翻译单元。`#pragma once` 请求编译器只展开一次，避免重复声明造成问题。

### 标准组件

```cpp
#include <optional>
#include <string_view>
#include <variant>
#include <vector>
```

每个头文件对应本项目公开接口直接使用的类型：

| 头文件 | 类型 | 本项目含义 |
| --- | --- | --- |
| `<optional>` | `std::optional<int>` | 求和可能没有结果 |
| `<string_view>` | `std::string_view` | 非拥有地观察输入字符 |
| `<variant>` | `std::variant` | 结果是数据或错误之一 |
| `<vector>` | `std::vector<int>` | 动态保存解析后的整数 |

公开声明使用什么类型，头文件就应直接包含定义该类型所需的标准头，而不要依赖其他文件偶然间接包含。

### 命名空间

```cpp
namespace stage1 {
```

命名空间把本项目名字放在 `stage1::` 下，减少与其他库中同名函数的冲突。例如调用者使用：

```cpp
stage1::parse_numbers(input);
```

### 结果类型

```cpp
using ParseResult =
    std::variant<std::vector<int>, std::string_view>;
```

实际源码写在一行，上面为了阅读换行。`using` 给较长类型起别名。

这个 variant 在任意时刻只保存两个候选之一：

```text
解析成功
⇒（选择第一个候选）vector<int>

解析失败
⇒（选择第二个候选）string_view
```

这种设计让“成功值”和“错误值”处于同一个返回类型中。缺点是错误只是一段固定文本，没有错误码、位置或原始 token。

这里返回的错误 `string_view` 指向字符串字面量。字符串字面量具有静态存储期，因此返回后仍然有效。如果它指向函数内部临时 `std::string`，就会悬空。

### 函数声明

```cpp
ParseResult parse_numbers(std::string_view text);
```

输入按值传递 `string_view`。它通常只包含字符地址和长度，复制 view 不复制字符内容。函数可以修改自己的 view 范围，而不会修改调用者的字符串对象。

```cpp
std::optional<int> sum_if_not_empty(
    const std::vector<int>& values
);
```

- `const`：函数不通过该引用修改 vector；
- `&`：不复制整个 vector；
- `optional<int>`：空输入返回“无值”，非空输入返回整数和。

## 解析实现

完整文件现位于 [`runtime_demo.cpp`](../../learning/stage-01/examples/03-modern-types/src/runtime_demo.cpp)。

### 引入声明

```cpp
#include "runtime_demo.hpp"
```

实现文件首先包含自己的头文件，可以让编译器检查声明和定义是否一致。

```cpp
#include <charconv>
#include <numeric>
```

- `<charconv>` 提供 `std::from_chars`；
- `<numeric>` 提供 `std::accumulate`。

### 函数入口

```cpp
ParseResult parse_numbers(std::string_view text) {
```

`text` 是一个非拥有 view。函数不会释放它指向的字符，也不会延长原字符存储的寿命。本程序中它观察 `argv[1]`，在 `main()` 运行期间有效。

### 空串检查

```cpp
if (text.empty())
  return std::string_view{"empty input"};
```

若输入长度为零，立即构造 variant 的错误候选并返回。此路径不会创建 `values`。

命令行中直接传入空字符串可触发它：

```bash
./build/stage1-lifetime/stage1_demo ""
```

### 局部容器

```cpp
std::vector<int> values;
```

`values` 是局部对象。它本身的生命周期从声明完成后开始，到函数退出时结束。

当 `push_back` 需要空间时，vector 会管理动态数组。函数成功返回时，动态数组中的元素被移动或复制到返回的 vector；函数失败返回时，局部 vector 析构并自动释放已分配存储。

错误输入 `1,x,3` 已经先把 `1` 压入 vector。解析 `x` 失败后直接返回，但不需要手写清理：

```text
values 已持有整数 1
⇒（错误分支 return）开始离开函数
⇒（局部对象析构）vector 释放内部存储
⇒（返回 ParseResult）调用者收到错误
```

这就是本程序中比 `unique_ptr` 更基础的一次 RAII：标准容器自身已经是资源管理对象。

### 循环条件

```cpp
while (!text.empty()) {
```

只要尚有字符未处理就继续。循环不会复制剩余字符串，而是逐步缩短 view。

### 查找逗号

```cpp
const auto comma = text.find(',');
```

`auto` 由编译器推断为 `std::string_view::size_type`。结果有两种：

- 找到：`comma` 是逗号相对当前 view 起点的位置；
- 未找到：`comma` 为 `std::string_view::npos`。

`const` 表示本轮循环不再改变这个位置。

### 取得片段

```cpp
const auto token = text.substr(0, comma);
```

`substr` 返回新的 `string_view`，仍不复制字符。

若找到逗号，token 覆盖逗号前的字符；若 `comma` 为 `npos`，token 覆盖全部剩余字符。

对输入 `1,2,3,4`，各轮状态为：

| 轮次 | `text` | `token` |
| ---: | --- | --- |
| 1 | `1,2,3,4` | `1` |
| 2 | `2,3,4` | `2` |
| 3 | `3,4` | `3` |
| 4 | `4` | `4` |

### 结果存储

```cpp
int value = 0;
```

先建立一个 `int` 对象作为解析目标。初始化为零使对象始终有确定值，但成功时 `from_chars` 会写入解析结果。

### 结构化绑定

```cpp
const auto [end, error] =
    std::from_chars(
        token.data(),
        token.data() + token.size(),
        value
    );
```

实际源码只对调用部分换行。三个参数含义：

1. `token.data()`：字符范围起点；
2. `token.data() + token.size()`：尾后地址；
3. `value`：成功后写入的整数。

解析范围是左闭右开：

\[
[p_b,p_e)
\]

尾后地址为：

\[
p_e
\overset{\text{移动长度}}{=}
p_b+n
\]

## 符号说明

- \(p_b\)：begin pointer，首字符地址。
- \(p_e\)：end pointer，尾后地址。
- \(n\)：token 的字符数量。
- \([p_b,p_e)\)：包含首地址、不包含尾后地址的范围。

`from_chars` 返回一个结果对象，结构化绑定把它拆成：

- `end`：解析停止位置；
- `error`：错误状态。

它不会抛出普通解析异常，也不会自动要求整个 token 都合法，所以代码必须检查两项条件。

### 完整检查

```cpp
if (
    error != std::errc{} ||
    end != token.data() + token.size()
) {
  return std::string_view{"invalid integer"};
}
```

第一项检查转换是否报告错误；第二项检查是否消费完整 token。

例如 token 为 `12x` 时，前面的 `12` 可能成功解析，但 `end` 停在 `x`。若只检查 `error`，程序可能错误地接受带尾随字符的输入。

逻辑关系为：

```text
转换错误
⇒（逻辑 OR）返回 invalid integer

未消费完整 token
⇒（逻辑 OR）返回 invalid integer

无错误且消费完整 token
⇒（通过检查）保存 value
```

### 保存整数

```cpp
values.push_back(value);
```

把本轮整数追加到 vector。若容量不足，vector 可能申请更大存储并移动已有元素。旧的元素指针、引用和迭代器可能因此失效；本函数没有保存这些旧位置，所以不受影响。

### 结束判断

```cpp
if (comma == std::string_view::npos)
  break;
```

未找到逗号说明当前 token 是最后一段。解析并保存后退出循环。

### 缩短视图

```cpp
text.remove_prefix(comma + 1);
```

找到逗号时，移除 token 和紧随其后的逗号。它只修改 view 的起点与长度，不移动或删除原字符。

对于 `1,2`：

```text
text 观察 "1,2"
⇒（comma 为 1）token 观察 "1"
⇒（remove_prefix 2）text 改为观察 "2"
```

### 成功返回

```cpp
return values;
```

返回类型是 `ParseResult`，编译器用 `values` 构造 variant 的 `vector<int>` 候选。

局部 `values` 即将结束生命周期。编译器可以使用移动或返回值优化，避免复制整个动态数组。无论是否发生优化，资源所有权都会由合法的对象接管，不应通过裸指针手工释放。

## 求和实现

```cpp
std::optional<int> sum_if_not_empty(
    const std::vector<int>& values
) {
  if (values.empty())
    return std::nullopt;
  return std::accumulate(
      values.begin(),
      values.end(),
      0
  );
}
```

空 vector 返回 `nullopt`，明确表达“没有求和值”。非空 vector 用半开迭代器范围求和。

初始值 `0` 的类型是 `int`，所以累加结果也是 `int`。大量数值可能发生有符号整数溢出；有符号溢出属于未定义行为，UBSan 可在执行到相应路径时报告，但当前测试没有覆盖大数溢出。

## 程序入口

完整文件现位于 [`main.cpp`](../../learning/stage-01/examples/03-modern-types/src/main.cpp)。

### 入口参数

```cpp
int main(int argc, char** argv) {
```

- `argc`：命令行参数数量，包含程序名；
- `argv`：指向各参数字符串的指针数组；
- `argv[0]`：通常是程序路径；
- `argv[1]`：第一个用户参数。

### 选择输入

```cpp
const std::string_view input =
    argc > 1 ? argv[1] : "1,2,3,4";
```

条件运算符的过程：

```text
argc 大于 1
⇒（存在用户参数）观察 argv[1]

argc 不大于 1
⇒（没有用户参数）观察默认字面量
```

两种字符存储在 `main()` 使用期间都有效，所以这里的 `string_view` 不会悬空。

### 接收结果

```cpp
auto result = stage1::parse_numbers(input);
```

`result` 的实际类型是 `ParseResult`。它拥有成功分支的 vector；错误分支只观察静态字符串字面量。

### 检查错误

```cpp
if (
    const auto* error =
        std::get_if<std::string_view>(&result)
) {
```

`std::get_if` 检查 variant 当前是否保存 `string_view`：

- 是：返回指向该候选值的指针；
- 否：返回 `nullptr`。

`if` 初始化语句把 `error` 的作用域限制在整个 `if` 语句中。条件判断指针是否非空。

```cpp
std::cerr << "status=" << *error << '\n';
return 1;
```

错误写入标准错误流 `cerr`，随后以退出码 `1` 结束 `main()`。离开前，所有已经构造成功的局部对象都会析构。

### 移交数据

```cpp
auto values = std::make_unique<std::vector<int>>(
    std::get<std::vector<int>>(
        std::move(result)
    )
);
```

这是本例最重要的所有权代码，可以分四步看：

```text
result 拥有 vector
⇒（std::move 转为可移动表达式）允许取走内容
⇒（std::get 取得 vector）构造堆上的 vector
⇒（make_unique）values 独占新 vector
```

`std::move` 自身不搬运字节，它把表达式转换为允许移动的类别。真正的移动发生在 `vector` 构造过程中。

移动后仍可以析构 `result`，但不应假定它原先 vector 的内容仍然存在。

为什么这里能安全使用 `std::get<vector<int>>`？因为前面的 `get_if<string_view>` 错误分支已经返回。按照 `ParseResult` 只有两个候选的定义，继续执行时只能是 vector 分支。

教学上这里展示 `unique_ptr`，但业务上没有必须把 vector 放到堆上的理由。直接写成局部 vector 往往更简单。保留这段是为了观察唯一所有权，不能据此得出“vector 都应放进 unique_ptr”的结论。

### 调用求和

```cpp
const auto sum =
    stage1::sum_if_not_empty(*values);
```

`values` 是智能指针，`*values` 解引用得到它管理的 vector。函数使用 `const&` 读取，不获取该 vector 的所有权。

### 输出结果

```cpp
std::cout
    << "count=" << values->size()
    << " sum=" << sum.value_or(0)
    << '\n';
```

- `values->size()` 取得元素数量；
- `sum.value_or(0)` 有值时返回和值，无值时返回 `0`；
- `\n` 写入换行字符。

正常解析至少产生一个 token，所以当前成功路径通常不会得到空 vector；`value_or(0)` 仍使 optional 的读取方式明确。

### 成功退出

```cpp
std::cout << "status=ok\n";
return 0;
```

退出时局部对象按构造相反顺序销毁。与资源最相关的部分是：

```text
values 开始析构
⇒（unique_ptr 析构）delete 所管理的 vector
⇒（vector 析构）释放内部动态数组
⇒（unique_ptr 完成析构）不再持有地址
```

随后 `result`、`input` 等局部对象也结束生命周期。`string_view` 不释放字符，因为它从未拥有字符。

## GPU迁移

将来 `unique_ptr<vector<int>>` 可以类比一次请求独占的 GPU workspace，但真实实现不能直接把 `delete` 用于 `cudaMalloc` 返回的地址。需要自定义 RAII 类型或 deleter，让申请与释放 API 配对：

```text
Host vector
⇒（析构规则）delete 或 allocator 释放 Host 存储

CUDA buffer
⇒（析构规则）cudaFree 释放 Device 存储
```

还必须考虑 CUDA 异步性：Host 对象准备析构时，GPU kernel 可能尚在使用 buffer。仅仅“有析构函数”还不够，释放时机必须与 Stream 上的工作完成关系一致。

## 直观理解

`variant` 像一个只能放“货物”或“错误单”的箱子；移动把箱中货物交给唯一管理员。管理员离开作用域时自动清仓。`string_view` 只是仓库摄像头画面，能看见字符，却不拥有仓库。
