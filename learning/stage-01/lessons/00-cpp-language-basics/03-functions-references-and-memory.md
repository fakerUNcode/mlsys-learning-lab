# 函数内存

本章建立函数调用时最重要的判断：数据是被复制、被借用，还是通过地址间接访问。

## 前置基础

1. **对象**：运行时具有类型、存储和生命周期的数据实体。
2. **地址**：对象在内存中的位置。
3. **作用域**：局部对象通常在离开其花括号时销毁。

## 按值传递

```cpp
ParseResult parse_numbers(std::string_view text);
```

`text` 按值传入，函数得到一份新的 `string_view`。复制的是“字符地址和长度”这份小型==视图信息==，不是整段字符。

所以实现中可以执行：

```cpp
text.remove_prefix(comma + 1);
```

它改变函数==内部视图==的起点，不会改变调用者的 `string_view`，也不会删除原始字符串中的字符。

## 引用传递

```cpp
std::optional<int> sum_if_not_empty(
    const std::vector<int>& values);
```

`values` 是调用者数组的别名。这里有两个关键性质：

- `&` 避免复制整个 `vector`；
- `const` 禁止函数通过该引用修改数组。

调用期间，真正的数组仍由调用者拥有。函数不能保存该引用并在数组销毁后继续使用。

## 指针访问

指针保存对象地址，也可以为空。示例中：

```cpp
const auto* error =
    std::get_if<std::string_view>(&result);
```

`&result` 取得 `result` 的地址并交给 `get_if`。返回值有两种情况：

```text
result 保存错误文字
⇒（类型匹配）error 指向错误

result 保存整数数组
⇒（类型不匹配）error 为空
```

只有确认指针非空后才能通过 `*error` 访问它指向的对象。当前代码把查询放在 `if` 条件中，条件成立时才进入函数体。

## 生命周期

局部对象从初始化完成后开始有效，在离开所属作用域时销毁：

```cpp
{
  std::vector<int> values;
  values.push_back(7);
}  // values 在这里销毁
```

`vector` 销毁时会释放自己的内部存储。把资源清理绑定到对象销毁，就是 ==RAII== 的核心性质。

这对 CUDA 同样重要：若一个对象拥有 Device 内存，它的析构函数可以调用 `cudaFree`；即使函数中途返回，只要对象正常离开作用域，清理仍会执行。

## 移动预览

`main.cpp` 中有：

```cpp
std::get<std::vector<int>>(
    std::move(result))
```

`std::move` 不会亲自搬运数据。它把 `result` 标记为“资源可以被转移”，随后 `vector` 的移动构造接管原有存储。

```text
result 拥有数组存储
⇒（移动构造）values 接管存储
⇒（移动完成）result 仍可析构
```

移动后的对象仍必须可以安全销毁和重新赋值，但不要依赖它还保存原来的内容。第 `01` 节会完整学习这套规则。

## 独占所有权

`unique_ptr<T>` 表示某个动态对象只有一个所有者：

```cpp
auto values =
    std::make_unique<std::vector<int>>(...);
```

`values` 离开作用域时，先销毁它管理的 `vector`，再释放对应内存。因为所有者只有一个，清理责任不会含糊，也不能直接复制这个 `unique_ptr`。

## 程序实例

阅读 [main.cpp](../../examples/00-runtime-foundations/src/main.cpp)，沿着下面的所有权变化口述一遍：

```text
result 拥有成功数组
⇒（std::move）unique_ptr 接管数组
⇒（只读引用）求和函数临时借用
⇒（main 返回）unique_ptr 自动清理
```

## 部署说明

使用 Sanitizer 构建可以帮助发现越界、释放后继续访问等错误：

```bash
cmake \
  -S learning/stage-01/examples \
  -B build/stage1-lifetime-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON
cmake --build build/stage1-lifetime-asan
ctest \
  --test-dir build/stage1-lifetime-asan \
  --output-on-failure
```

Sanitizer 是检查工具，不会替代对所有权和生命周期的理解。

## 直观理解

按值传递像复印资料，引用像把原件借给对方看，指针像写着存放地址的纸条，移动像正式办理物品交接。谁拿着所有权凭证，谁负责最后归还资源。
