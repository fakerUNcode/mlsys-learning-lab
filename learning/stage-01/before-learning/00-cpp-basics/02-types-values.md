# 类型与值

C++ 的类型不仅说明“数据长什么样”，还参与约束能否修改、怎样传递以及谁负责管理资源。

## 前置基础

1. **变量**：有名字的数据对象。
2. **赋值**：把新值写入已经存在的对象。
3. **作用域**：名字能够被访问的代码范围。

## 声明拆读

先看示例中的声明：

```cpp
const std::string_view input = argc > 1 ? argv[1] : "1,2,3,4";
```

按以下顺序阅读：

1. 名字是 `input`；
2. 核心类型是 `std::string_view`；
3. `const` 表示初始化后不能通过 `input` 改变这份视图；
4. 右侧表达式决定它查看命令行参数还是默认文本。

这里的“初始化”发生在对象创建时。初始化与后续赋值不是一回事：

```cpp
int count = 4;  // 创建 count，并初始化为 4
count = 5;      // count 已存在，现在修改它的值
```

## 类型作用

几个基础类型在当前示例中的职责不同：

| 类型 | 保存内容 | 本例用途 |
| --- | --- | --- |
| `int` | 整数 | 单个输入数字与总和 |
| `string_view` | 字符地址和长度 | 只查看输入文本 |
| `vector<int>` | 一组连续整数 | 拥有解析结果 |
| `optional<int>` | 一个整数或无值 | 表达求和是否存在 |
| `variant<A, B>` | A 或 B 中的一种 | 表达解析成功或失败 |

`string_view` 与 `vector` 最重要的区别是所有权。`vector<int>` 拥有其中的整数；`string_view` 不拥有字符，只记录去哪里看。因此原字符先失效时，视图会悬空。

## const含义

`const` 要结合它修饰的对象理解：

```cpp
const int count = 4;
```

`count` 初始化后不能修改。

```cpp
const std::vector<int>& values
```

函数通过引用查看调用者的数组，并承诺不经由 `values` 修改数组。它不是把数组复制成一份新的常量数组。

## auto推断

`auto` 让编译器从初始化表达式推断类型：

```cpp
auto result = stage1::parse_numbers(input);
```

根据函数声明，`result` 的类型是 `stage1::ParseResult`。`auto` 省去重复书写已知类型，但不会让 C++ 变成无类型语言；编译完成后类型仍然确定。

学习时每看到一个 `auto`，都应该先尝试说出它推断后的类型。若说不出，就回到右侧表达式或函数声明查找。

## 表达式语句

表达式产生值或完成计算，例如：

```cpp
argc > 1
values.size()
sum.value_or(0)
```

语句规定一次完整动作，通常以分号结束：

```cpp
values.push_back(value);
return 0;
```

控制结构根据条件决定执行路径：

```cpp
if (text.empty()) {
  return std::string_view{"empty input"};
}
```

执行顺序为：

```text
检查 text.empty()
⇒（条件为真）返回错误

检查 text.empty()
⇒（条件为假）继续解析
```

## 程序实例

在 [main.cpp](../../examples/src/main.cpp) 中依次找出 `input`、`result`、`values` 和 `sum`，为每个名字写下：

- 完整类型；
- 是否拥有底层数据；
- 初始化后是否会被修改；
- 作用域在哪里结束。

## 部署说明

类型错误会在编译期被发现。修改教学代码前，可以先只构建核心目标观察诊断：

```bash
cmake --build build/stage1-lifetime --target stage1_core
```

阅读错误信息时先找文件名和行号，再找“期望类型”和“实际类型”。

## 直观理解

类型像容器标签：纸箱能装什么、是否可拆封、由谁保管都写在标签上。`auto` 是让仓库根据货物自动打印标签，并不是取消标签。
