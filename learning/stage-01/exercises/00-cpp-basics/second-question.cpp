// 第 00 节第二题：把命令行文本交给现有核心函数，并分别处理成功和失败结果。
// 本文件只负责组织输入、分支和输出，不修改 parse_numbers 的实现。
#include "runtime_demo.hpp"

#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
  // argv 中的字符由程序启动环境提供，在 main 运行期间保持有效。input 只是
  // 查看 argv[1]，不复制也不拥有字符，因此不能比被查看的命令行字符活得更久。
  if (argc < 2) {
    std::cerr << "status=missing input\n";
    // 返回 2 表示程序没有得到可处理的有效结果。
    return 2;
  }
  const std::string_view input = argv[1];

  auto result = stage1::parse_numbers(input);

  // get_if 查询 variant 是否处于 string_view 错误分支。类型匹配时返回指针，
  // 不匹配时返回空指针，因此只有进入 if 后才能读取 *error。
  if (const auto* error = std::get_if<std::string_view>(&result)) {
    std::cerr << "status=" << *error << '\n';
    // 返回 2 表示解析失败；shell 可以通过退出码识别失败，而不必分析输出文字。
    return 2;
  }

  // 错误分支已经排除，此时 result 保存 vector<int>。这里通过 const 引用读取
  // variant 内部的数组，不复制数组，也不取得其所有权。
  const auto& values = std::get<std::vector<int>>(result);

  // 读取首元素前必须检查数组非空，否则 values.front() 没有可访问的元素。
  if (values.empty()) {
    std::cerr << "status=empty result\n";
    return 2;
  }

  std::cout << "first=" << values.front() << '\n';
  std::cout << "status=ok\n";

  // 返回 0 表示输入成功解析，并且首元素已经打印。
  return 0;
}
