#include "runtime_demo.hpp"  // 引入本项目的头文件，其中声明了 ParseResult、parse_numbers 和 sum_if_not_empty

#include <charconv>  // 提供 std::from_chars，用来把字符串转换成整数
#include <numeric>   // 提供 std::accumulate，用来对一组数字求和

namespace stage1 {  // 把下面的函数放进 stage1 命名空间，完整名字会变成 stage1::函数名

ParseResult parse_numbers(std::string_view text) {  // 定义解析函数：输入一段文本，返回解析结果
  if (text.empty()) return std::string_view{"empty input"};  // 如果输入为空，直接返回“empty input”错误信息

  std::vector<int> values;  // 创建一个动态整数数组，用来保存解析出的数字

  while (!text.empty()) {  // 只要剩余文本不为空，就继续解析
    const auto comma = text.find(',');  // 查找当前文本中第一个逗号的位置
    const auto token = text.substr(0, comma);  // 取出逗号前面的这一段文本，例如从 "10,20" 取出 "10"

    int value = 0;  // 创建整数变量，用来接收转换后的数字

    const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);  // 尝试把 token 的全部字符转换成整数，并得到结束位置和错误状态

    if (error != std::errc{} || end != token.data() + token.size()) {  // 如果转换失败或没有完整吃掉 token 中的全部字符，就认为输入不是合法整数
      return std::string_view{"invalid integer"};  // 返回“invalid integer”错误信息
    }

    values.push_back(value);  // 把成功解析出的整数追加到 values 数组末尾

    if (comma == std::string_view::npos) break;  // 如果没找到逗号，说明当前 token 已经是最后一个数字，结束循环

    text.remove_prefix(comma + 1);  // 删除已经处理过的“数字+逗号”这一段，只保留后面的文本继续解析
  }

  return values;  // 所有数字都解析成功后，返回整数数组
}

std::optional<int> sum_if_not_empty(const std::vector<int>& values) {  // 定义求和函数：接收整数数组，并返回“可能存在的整数结果”
  if (values.empty()) return std::nullopt;  // 如果数组为空，就返回“没有结果”
  return std::accumulate(values.begin(), values.end(), 0);  // 从 0 开始，把数组中的所有整数依次相加并返回总和
}

}  // namespace stage1，结束 stage1 命名空间