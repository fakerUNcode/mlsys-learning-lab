// Stage 1 的核心接口：把命令行文本解析为整数，并对整数数组求和。
//
// 程序的数据流是：
//
//   "1,2,3" -> parse_numbers -> vector<int>{1, 2, 3}
//                         或 -> "invalid integer"
//
// main.cpp 用这些接口展示一次完整运行；runtime_test.cpp 用相同接口检查行为。
// 函数定义在 src/runtime_demo.cpp，本文件只写调用者必须遵守的约定。
#pragma once

#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace stage1 {

// 一次解析要么成功，要么失败，不能同时发生。variant 正好用一个对象表达这种
// “二选一”结果：
//
//   std::vector<int>    成功，例如 {1, 2, 3}
//   std::string_view    失败，例如 "invalid integer"
//
// 错误文字目前都是字符串字面量，整个程序运行期间都有效，因此返回的
// string_view 不会悬空。它不拥有字符；若以后改为动态生成错误信息，应改用
// std::string 等拥有内容的类型。
using ParseResult = std::variant<std::vector<int>, std::string_view>;

// 解析逗号分隔的十进制整数。
//
// 参数：
//   text - 输入文本，例如 "10,20,-3"。函数只读取其字符；虽然会移动形参
//          string_view 的观察范围，但不会修改原始文本。
//
// 返回：
//   - 成功：vector<int>，其中每个元素对应一个完整的分隔项；
//   - 失败：string_view 错误说明。空输入返回 "empty input"，无法完整转换或
//           超出 int 范围的项返回 "invalid integer"。
//
// 例子：
//   "2,3,5"  -> {2, 3, 5}
//   "2,x"    -> "invalid integer"
//
// 当前教学实现不是完整的 CSV 解析器：开头或中间的空项会失败，但末尾逗号不会
// 生成新的 token，因此 "1,2," 会被解析为 {1, 2}。后续可把它作为边界练习改进。
ParseResult parse_numbers(std::string_view text);

// 计算非空数组的元素总和。
//
// 参数：
//   values - 只读数组。const 引用避免复制；函数调用期间调用者必须保证数组有效。
//
// 返回：
//   - 非空数组：包含总和的 optional<int>；
//   - 空数组：std::nullopt。
//
// optional 用于区分“确实算出总和 0”（如 {-1, 1}）和“没有元素可计算”。当前以
// int 累加，极端大的输入仍可能造成整数溢出。
std::optional<int> sum_if_not_empty(const std::vector<int>& values);

}  // namespace stage1
