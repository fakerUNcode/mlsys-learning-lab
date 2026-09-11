// “现代类型”核心库的最小测试。
//
// 当前覆盖：
// - 合法整数列表；
// - 非法字符；
// - 空 vector 的求和。
//
// 尚未覆盖空字符串、连续逗号、末尾逗号、整数越界和求和溢出。

#include "runtime_demo.hpp"

#include <cassert>      // assert
#include <string_view>  // 错误候选类型
#include <vector>       // 成功候选类型

int main() {
  // 成功路径：2、3、5 应被解析为 vector<int>。
  auto parsed = stage1::parse_numbers("2,3,5");

  // 先检查 variant 的当前候选，再使用 std::get 取值。
  // 如果跳过检查且候选类型错误，std::get 会抛 bad_variant_access。
  assert(std::holds_alternative<std::vector<int>>(parsed));

  // const& 避免复制 vector；引用有效期依赖 parsed。
  const auto& values = std::get<std::vector<int>>(parsed);

  // optional 有值且内容为 10 时，该比较才成立。
  assert(stage1::sum_if_not_empty(values) == 10);

  // 错误路径：x 不是完整整数，结果应保存错误 string_view。
  auto invalid = stage1::parse_numbers("2,x");
  assert(std::get<std::string_view>(invalid) == "invalid integer");

  // {} 构造一个临时空 vector；函数应返回 nullopt。
  assert(!stage1::sum_if_not_empty({}).has_value());

  // 全部断言通过后返回 0，CTest 将测试标记为 Passed。
  return 0;
}
