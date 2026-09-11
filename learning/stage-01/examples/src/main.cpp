// 本程序把核心库接到命令行：取得文本 -> 解析 -> 根据成功或失败分支输出结果。
// 解析与求和没有写在这里，目的是让这层只负责用户交互，核心逻辑可以复用和测试。
#include "runtime_demo.hpp"

#include <iostream>
#include <memory>
#include <utility>

int main(int argc, char** argv) {
  // 运行 `./stage1_demo 1,2,3` 时，argv[1] 指向 "1,2,3"；没有额外参数时则
  // 使用默认文本。string_view 不复制字符，只借用这两处已有字符；它们都会一直
  // 有效到 main 结束，所以在本例的使用范围内是安全的。
  const std::string_view input = argc > 1 ? argv[1] : "1,2,3,4";
  auto result = stage1::parse_numbers(input);

  // ParseResult 是 variant，内部此刻只能保存“整数数组”或“错误文字”之一。
  // get_if 安全地查询错误分支：若输入是 "1,x"，error 指向 "invalid integer"；
  // 若解析成功，它返回空指针，程序继续处理整数数组。
  if (const auto* error = std::get_if<std::string_view>(&result)) {
    std::cerr << "status=" << *error << '\n';
    // 非零退出码使 shell、CTest 或脚本都能识别本次运行失败。
    return 1;
  }

  // 此处已经排除错误分支，因此 result 保存 vector<int>。std::move 允许把该数组
  // 的资源移交给 unique_ptr 管理，而不是复制所有元素。unique_ptr 表示“唯一所有者”：
  // 本例中只有 values 负责这份数组；离开 main 时它会自动销毁数组，无需 delete。
  auto values = std::make_unique<std::vector<int>>(
      std::get<std::vector<int>>(std::move(result)));

  // sum_if_not_empty 只借用 *values 来读取数据。optional 有值时取总和；本示例
  // 使用 value_or(0) 为没有结果的情况准备显示值，正常解析出的数组不会走到该分支。
  const auto sum = stage1::sum_if_not_empty(*values);
  std::cout << "count=" << values->size() << " sum=" << sum.value_or(0)
            << '\n';
  std::cout << "status=ok\n";

  // 返回 0 表示成功。随后局部对象按生命周期自动销毁：unique_ptr 销毁 vector，
  // vector 再释放内部存储。这条自动清理链就是本示例要观察的 RAII 行为。
  return 0;
}
