#include "runtime_demo.hpp"

#include <iostream>
#include <string_view>
#include <vector>
#include <variant>

int main(int argc, char** argv) {
  // TODO 1：检查是否提供了命令行参数。
  // 没有参数时输出 status=missing input，并返回 2。
  if (argc < 2) {
      std::cout << "status=missing input\n";
      return 2;
  }

  // TODO 2：创建 string_view input，让它查看 argv[1]。
  // - 内存来源：argv[1] 指向的字符数据是由操作系统在启动程序时分配并提供的。
// - 借用机制：input 只是一个轻量级的视图，它仅仅“借用”这块内存来观察字符，绝对不会发生数据复制，也不拥有数据的释放权。
// - 生命周期：在本程序中，input 仅在 main() 函数的作用域内被访问。由于命令行参数的内存贯穿程序的整个运行期，因此在这里使用它是完全安全且始终有效的。
  std::string_view input(argv[1]);
  auto result = stage1::parse_numbers(input);

  // 检查是否包含错误信息 (string_view)
  if (const auto* error = std::get_if<std::string_view>(&result)) {
      std::cout << "status=" << *error << '\n';
      return 2;
  }

  // TODO 1：从 result 中取得 vector<int>，使用 const 引用，不能复制数组。
  const auto& values = std::get<std::vector<int>>(result);

  // TODO 2：读取首元素前检查数组是否为空。
  // 为空时输出 status=empty result，并返回 2。
  if (values.empty()) {
      std::cout << "status=empty result\n";
      return 2;
  }

  // TODO 3：打印 first=<首元素>
  std::cout << "first=" << values.front() << '\n';

  // TODO 4：打印 status=ok，并返回 0。
  std::cout << "status=ok\n";
  return 0;
}