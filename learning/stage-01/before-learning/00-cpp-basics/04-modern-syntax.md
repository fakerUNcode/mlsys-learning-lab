# 现代语法

本章集中拆解当前示例中最容易形成视觉压力的 C++17 写法。目标是能读懂它们表达的状态，不要求研究标准库内部实现。

## 前置基础

1. **类型**：规定对象可以保存什么以及支持哪些操作。
2. **函数返回值**：调用完成后交还给调用者的结果。
3. **分支**：程序根据状态选择一条执行路径。

## 模板类型

模板可以把类型作为参数。阅读时先找最外层名字，再从尖括号内向外组合：

```cpp
std::vector<int>
```

最外层是 `vector`，元素类型是 `int`，所以它是一组整数。

```cpp
std::optional<int>
```

最外层是 `optional`，候选值类型是 `int`，所以它是“一个整数或无值”。

```cpp
std::variant<std::vector<int>, std::string_view>
```

最外层是 `variant`，两个候选类型分别是整数数组和字符视图。一个 `variant` 对象在任意时刻只保存其中一种。

## 类型别名

嵌套模板很长，因此头文件给它起了业务名字：

```cpp
using ParseResult =
    std::variant<std::vector<int>, std::string_view>;
```

`ParseResult` 没有创建一种行为不同的新类型，它只是原类型的别名。名字把关注点从“模板拼写”提升为“这是一次解析的结果”。

## variant分支

解析成功时返回：

```cpp
return values;
```

因为 `values` 是 `vector<int>`，它进入 `variant` 的成功分支。失败时返回 `string_view`，进入错误分支。

下面这两句代码用于处理一个可能“成功或失败”的 `std::variant` 结果。

```
std::variant<std::vector<int>, std::string_view> result;
```

它只能保存两种类型之一：

```
成功：std::vector<int>
失败：std::string_view
```

查询分支时使用：

```cpp
std::get_if<std::string_view>(&result)
```

类型匹配时得到指针，不匹配时得到空指针。作用是：

> 查询 `result` 当前是否保存着 `std::string_view`。

它返回一个指针：

```
std::string_view*   // 匹配
nullptr             // 不匹配
```



确认成功分支后，代码再用：

```cpp
std::get<std::vector<int>>(std::move(result))
```

取得整数数组。作用是：

> 确认 `result` 保存的是 `std::vector<int>` 后，取出这个整数数组。

这里的：

```c++
std::get<std::vector<int>>(...)
```

表示按类型获取 `std::vector<int>` 分支。

如果类型正确，就得到数组；如果类型不正确，就会抛出：

```c++
std::bad_variant_access
```

因此必须先排除错误分支若没有先排除错误分支就对错误结果这样调用，`std::get` 会报告分支不匹配。



`std::move(result)` 的作用

```c++
std::move(result)
```

表示：

> 把 `result` 转换成“可以被移动”的对象。

它本身不会立即搬运数据，而是允许后面的 `std::get` 返回数组的==右值==引用。

不使用 `std::move`：

```c++
auto values =
    std::get<std::vector<int>>(result);
```

获取的是数组的引用，之后可能发生复制。

使用 `std::move`：

```c++
auto values =
    std::get<std::vector<int>>(std::move(result));
```

可以把内部的 `vector<int>` 移出来，避免复制大量整数。

可以近似理解为：

```c++
std::vector<int> values =
    std::move(result 中的 vector);
```

移动之后，`result` 仍然存在，但其中的数组通常处于“已被移动”的状态，不应再依赖它原来的内容。

> #### 1. 左值：有身份、可定位的对象
>
> 左值通常代表一个已经存在、可以通过地址找到的对象。
>
> ```
> int x = 10;
> ```
>
> 这里：x是左值，因为它有名字，也有固定存储位置。
>
> 因此可以使用：
>
> ```
> x = 20;
> ```
>
> #### 2. 右值：临时值或即将被使用的值
>
> 右值通常是临时产生的值，没有持久的名字。
>
> ```
> int x = 1 + 2;
> ```
>
> 这里：
>
> ```
> 1 + 2
> ```
>
> 产生的是临时结果，属于右值。
>
> 常见右值：
>
> ```
> 42
> x + 1
> std::string{"hello"}
> create_object()
> ```
>
> 例如：
>
> ```
> int y = x + 1;
> ```
>
> `x + 1` 是一个临时结果，用完通常就不再需要。
>
> ## 区别表
>
> | 概念     | 本质                     | 常见形式                        | 通常绑定到                                 | 是否拥有对象 | 主要用途                   |
> | -------- | ------------------------ | ------------------------------- | ------------------------------------------ | ------------ | -------------------------- |
> | 左值     | 有稳定身份的表达式       | `value`、`*ptr`、`vec[0]`       | 已存在的对象                               | 不适用       | 读取或修改已有对象         |
> | 右值     | 临时值或可被移动的表达式 | `42`、`T{}`、`std::move(value)` | 临时对象或可转移资源的对象                 | 不适用       | 计算结果、初始化、移动资源 |
> | 左值引用 | 已有对象的别名           | `T&`、`const T&`                | `T&` 通常绑定左值；`const T&` 也能绑定右值 | 不拥有       | 避免复制、修改或只读借用   |
> | 右值引用 | 可移动对象的别名         | `T&&`                           | 右值                                       | 不拥有       | 实现移动语义和完美转发     |
>
> ### 左值引用
>
> ```c++
> int number = 10;
> int& reference = number;
> ```
>
> 关系为：
>
> ```
> number 存放整数
> ⇒（引用绑定）reference 成为别名
> ```
>
> ### 右值引用
>
> 左值引用用于借用仍要继续使用的对象，右值引用表示该对象的资源可以被转移。但要注意：两种引用都不拥有对象，而且右值引用本身不会移动资源。真正的移动发生在移动构造或移动赋值时。
>
> 一句话记忆：
>
> ```
> 左值引用：我借来用
> 右值引用：资源可以暂时放在我这
> ```
>
> ```c++
> std::string text = "CUDA";
> 
> std::string&& movable =
>     std::move(text);
> ```
>
> `movable` 的类型是：
>
> ```c++
> std::string&&
> ```
>
> 它绑定到可被移动的 `text`。
>
> 但此时还没有真正转移字符串资源。真正的移动发生在移动构造或移动赋值中：
>
> ```c++
> std::string target =
>     std::move(text);
> ```
>
> 过程为：
>
> ```
> text 拥有字符资源
> ⇒（std::move 转换）允许转移
> ⇒（移动构造）target 接管资源
> ```
>
> 移动以后，`text` 仍然存在并且可以安全销毁，但不能再假定它仍保存 `"CUDA"`。
>
> ### 示例程序片段
>
> ```c++
> int main() {
>   std::string text = "CUDA";        // text 是左值
> 
>   std::string& left_ref = text;     // 左值引用：绑定已有对象
>   left_ref = "GPU";                 // 修改的就是 text
> 
>   std::string&& right_ref =
>       std::move(text);              // 右值引用：允许转移 text 的资源
> 
>   std::string target =
>       std::move(right_ref);         // 此处真正执行移动构造
> 
>   std::cout << target << '\n';      // 输出 GPU
> }
> ```
>
> | 写法              | 类别     | 含义                             |
> | ----------------- | -------- | -------------------------------- |
> | `text`            | 左值     | 有名字、可持续访问的对象         |
> | `std::move(text)` | 右值     | 将 `text` 标记为可以被移动       |
> | `std::string&`    | 左值引用 | 借用已有对象，不复制             |
> | `std::string&&`   | 右值引用 | 绑定可移动的对象，不代表已经移动 |
>
> 关键过程：
>
> ```
> text 拥有字符串
> ⇒（左值引用）left_ref 借用
> ⇒（std::move）允许转移
> ⇒（移动构造）target 接管资源
> ```
>
> 注意：`std::move` 只进行类型转换，真正的资源转移发生在 `target` 的移动构造中。

## optional状态

`optional<int>` 的两种状态为：

```text
包含 int
⇒（value_or）使用该整数

不包含值
⇒（value_or）使用备用值
```

空数组返回 `std::nullopt`。`has_value()` 用来询问是否有值，`value_or(0)` 表示有值就取出，没有值就使用 `0`。

## 结构绑定

`from_chars` 一次性返回“停止位置”和“错误码”。结构化绑定把这两个部分分别命名：

```cpp
const auto [end, error] =
    std::from_chars(...);
```

这不是创建数组。它相当于把一个组合结果拆开，后面便能分别检查 `end` 和 `error`。

## 条件初始化

下面的 `error` 只在这条 `if` 语句控制的范围内有效：

```cpp
if (const auto* error =
        std::get_if<std::string_view>(&result)) {
  std::cerr << *error << '\n';
}
```

这样变量不会泄露到不需要它的后续代码中。指针为空时条件为假；非空时进入分支。

## 字符转换

`from_chars` 接收字符区间的起点和终点，并尝试写入 `value`：

```cpp
std::from_chars(
    token.data(),
    token.data() + token.size(),
    value)
```

仅检查错误码还不够。例如输入 `12x` 时可能先读出 `12`。实现还检查停止位置是否到达末尾，因此要求整个 token 都是合法整数。

```text
转换无错误
⇒（检查停止位置）完整整数或非法后缀
```

## 程序实例

按这个顺序阅读 [runtime_demo.cpp](../../examples/src/runtime_demo.cpp)：

1. `string_view` 划分当前 token；
2. 结构化绑定接收转换结果；
3. `variant` 表达成功或错误；
4. `vector` 保存全部整数；
5. `optional` 表达和值是否存在。

## 部署说明

运行两条路径并观察 `variant` 的不同分支：

```bash
./build/stage1-lifetime/stage1_demo 1,2,3
./build/stage1-lifetime/stage1_demo 1,x,3
```

第二条命令应返回非零退出码，可以紧接着执行 `echo $?` 查看。

## 直观理解

`variant` 像只能放一种物品的双格保险箱，`optional` 像可能为空的单格盒子，结构化绑定像拆开一张包含两项信息的回执。模板参数说明每个容器允许放什么。
