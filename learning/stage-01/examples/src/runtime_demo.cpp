// 核心函数的实现。本文件不关心终端打印或退出码，只完成两件事：解析文本，
// 以及对已解析的数组求和。这样同一份逻辑既能供 main.cpp 调用，也能独立测试。
#include "runtime_demo.hpp"

#include <charconv>
#include <numeric>

namespace stage1 {

ParseResult parse_numbers(std::string_view text) {
  // 第一步：先处理没有任何字符的输入。它不是一个空数组的成功结果，而是用户没有
  // 提供可解析文本，因此走错误分支。
  if (text.empty()) {
    return std::string_view{"empty input"};
  }

  // 第二步：创建成功结果的容器。vector 拥有自己存储的整数；如果后面发现错误并
  // 提前 return，局部 vector 会在离开作用域时自动清理，这就是 RAII 的实际效果。
  std::vector<int> values;

  // 第三步：反复取出一个逗号前的片段。例如 text 从 "12,7,-3" 依次变为
  // "12,7,-3"、"7,-3"、"-3"。string_view 只改变“看哪里”和“看多长”，
  // 不复制也不改写原始字符串。
  while (!text.empty()) {
    const auto comma = text.find(',');
    const auto token = text.substr(0, comma);
    int value = 0;

    // 第四步：将当前片段转换为 int。from_chars 返回两个信息：
    //
    //   end    实际停止读取的位置；
    //   error  转换是否报错。
    //
    // 两个条件都必须检查。比如 "12x" 的前缀 12 可以读成整数，但 end 停在 x
    // 前；只有 end 到达 token 末尾，才能确认整个片段都是一个整数。
    const auto [end, error] =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc{} || end != token.data() + token.size()) {
      return std::string_view{"invalid integer"};
    }

    // 第五步：转换成功后才加入结果。因此发生错误时，调用者不会收到半成品数组。
    values.push_back(value);

    // 找不到逗号说明刚处理的是最后一项；否则跳过当前逗号，继续处理余下文本。
    if (comma == std::string_view::npos) {
      break;
    }
    text.remove_prefix(comma + 1);
  }

  // values 离开函数时不会被复制到裸指针中。它的资源会作为 variant 成功分支的一部分
  // 被安全地转移给调用者，调用者最终也不需要手写释放内存。
  return values;
}

std::optional<int> sum_if_not_empty(const std::vector<int>& values) {
  // 空数组没有“一个元素求出的和”。返回 nullopt 让调用者能够与总和恰好为 0 区分。
  if (values.empty()) {
    return std::nullopt;
  }

  // accumulate 从初始值 0 开始依次累加。例如 {2, 3, 5} 的计算是
  // ((0 + 2) + 3) + 5，结果为 10。函数只读取 values，不取得其所有权。
  return std::accumulate(values.begin(), values.end(), 0);
}

}  // namespace stage1
