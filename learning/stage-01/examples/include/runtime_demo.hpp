// 本头文件声明“现代类型”示例的公共接口。
//
// 为什么接口单独放头文件：
// - demo 与 test 都需要调用同一份实现；
// - 编译器先通过声明检查调用参数和返回类型；
// - 实现留在 .cpp 中，修改实现时不必改调用方源码。

#pragma once  // 防止同一翻译单元重复展开本头文件

#include <optional>     // std::optional：表达“可能没有结果”
#include <string_view>  // std::string_view：非拥有地观察字符
#include <variant>      // std::variant：表达多个候选类型之一
#include <vector>       // std::vector：动态保存解析出的整数

namespace stage1 {

// 解析只有两种结果：
// 1. 成功：vector<int> 拥有全部解析数字；
// 2. 失败：string_view 指向一个静态错误字符串。
//
// variant 在同一时刻只保存其中一个候选。
// 当前错误 view 都指向字符串字面量，生命周期覆盖整个程序；
// 不要返回指向局部 std::string 的 string_view，否则会悬空。
using ParseResult = std::variant<std::vector<int>, std::string_view>;

// 把逗号分隔文本解析为整数数组。
// text 按值传递，但 string_view 的复制不会复制底层字符。
ParseResult parse_numbers(std::string_view text);

// 对非空数组求和；空数组返回 nullopt。
// const& 避免复制 vector，并保证函数不修改输入。
// 当前返回 int，极大输入存在有符号溢出风险，教学测试尚未覆盖。
std::optional<int> sum_if_not_empty(const std::vector<int>& values);

}  // namespace stage1
