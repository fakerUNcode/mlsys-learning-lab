// 命令行入口：解析逗号分隔整数并输出数量与总和。
//
// 示例：
//   modern_types_demo 1,2,3,4
//   输出 count=4 sum=10，并以退出码 0 结束。
//
//   modern_types_demo 1,x,3
//   输出 status=invalid integer，并以退出码 1 结束。

#include "runtime_demo.hpp"  // parse_numbers、sum_if_not_empty

#include <iostream>  // std::cout、std::cerr
#include <memory>    // std::unique_ptr、std::make_unique
#include <utility>   // std::move

int main(int argc, char** argv) {
  // argc 包含程序名：
  // - argc > 1：观察用户提供的 argv[1]；
  // - 否则：观察静态字符串字面量 "1,2,3,4"。
  // 两种底层字符在 main 使用期间都有效，所以 input 不悬空。
  const std::string_view input = argc > 1 ? argv[1] : "1,2,3,4";

  // result 是 variant，当前只可能保存 vector<int> 或错误 string_view。
  auto result = stage1::parse_numbers(input);

  // get_if 不抛异常：
  // - 当前候选是 string_view 时，返回指向错误文本的指针；
  // - 否则返回 nullptr。
  // error 只在 if 语句及其分支中可见。
  if (const auto* error = std::get_if<std::string_view>(&result)) {
    std::cerr << "status=" << *error << '\n';
    return 1;  // 非零退出码通知 shell：本次运行失败
  }

  // 前面的错误分支已经 return，因此这里可确定 result 保存 vector<int>。
  //
  // std::move(result) 允许从 variant 内部移动 vector；
  // make_unique 在堆上构造新 vector，并让 values 独占它。
  //
  // 这段代码用于教学 unique_ptr。业务代码没有必须把 vector 放在堆上的理由，
  // 直接使用局部 vector 往往更简单，不要形成“所有对象都套智能指针”的习惯。
  auto values = std::make_unique<std::vector<int>>(
      std::get<std::vector<int>>(std::move(result)));

  // *values 解引用得到 vector；函数通过 const& 读取，不取得所有权。
  const auto sum = stage1::sum_if_not_empty(*values);

  // 成功解析至少会保存一个整数。
  // value_or(0) 展示 optional 的安全读取：无值时使用回退值 0。
  std::cout << "count=" << values->size() << " sum=" << sum.value_or(0)
            << '\n';
  std::cout << "status=ok\n";

  // 返回前，局部对象按构造逆序析构。
  // values 的 unique_ptr 自动销毁 vector，vector 再释放内部动态数组。
  return 0;
}
