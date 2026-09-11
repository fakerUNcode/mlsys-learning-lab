// 这是核心库的自动测试程序。它不检查终端输出，而是直接调用接口，验证三个最重要的
// 约定：正确输入会得到数组，非法输入会得到错误，空数组没有可计算的和。
//
// assert(条件) 在条件为假时立即让测试失败；CTest 看到非零退出码就报告 Failed。
// 因为定义 NDEBUG 后 assert 可能被移除，所以本示例应使用 Debug 构建运行测试。
#include "runtime_demo.hpp"

#include <cassert>
#include <string_view>
#include <vector>

int main() {
  // 场景 1：验证正常数据流。
  // "2,3,5" 必须先进入 variant 的 vector<int> 成功分支；若类型不对，第一条
  // 断言失败。随后以引用读取该数组，验证求和结果为 10。
  auto parsed = stage1::parse_numbers("2,3,5");
  assert(std::holds_alternative<std::vector<int>>(parsed));
  const auto& values = std::get<std::vector<int>>(parsed);
  assert(stage1::sum_if_not_empty(values) == 10);

  // 场景 2：验证错误不会被忽略。
  // "x" 不是整数；若实现错误地跳过它、接受部分内容或返回成功数组，此断言都会失败。
  auto invalid = stage1::parse_numbers("2,x");
  assert(std::get<std::string_view>(invalid) == "invalid integer");

  // 场景 3：验证空数组的语义。
  // 这里要的不是数值 0，而是 optional 没有值；若误把空数组当成普通求和结果，
  // has_value() 会变为 true，此断言便会失败。
  assert(!stage1::sum_if_not_empty({}).has_value());

  return 0;
}
