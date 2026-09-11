// “现代类型”示例的业务实现。
//
// 主线映射：
// - string_view：读取算子配置或 shape 文本而不复制字符；
// - variant：明确表达成功值或错误值；
// - vector：在 Host 侧保存动态任务/维度列表；
// - optional：明确表达“没有可用结果”；
// - 结构化绑定：拆开 from_chars 返回的结束位置和错误码。

#include "runtime_demo.hpp"  // 先包含自己的头，检查声明与定义一致

#include <charconv>  // std::from_chars：无异常地解析整数
#include <numeric>   // std::accumulate：对迭代器范围求和

namespace stage1 {

ParseResult parse_numbers(std::string_view text) {
  // 空文本没有任何 token，直接返回错误候选。
  // 错误 view 指向字符串字面量，返回后仍然有效。
  if (text.empty()) {
    return std::string_view{"empty input"};
  }

  // vector 是 RAII 容器：
  // 成功时它被移动进返回的 variant；
  // 中途失败时它自动析构并释放已申请的 Host 内存。
  std::vector<int> values;

  // 每轮解析 text 当前开头的一个 token，然后缩短 view。
  // remove_prefix 只改变 view 的地址/长度，不修改原始 argv 字符串。
  while (!text.empty()) {
    // find 找到第一个逗号的位置；不存在时返回 string_view::npos。
    const auto comma = text.find(',');

    // substr 产生另一个非拥有 view：
    // 有逗号时观察逗号前内容；无逗号时观察全部剩余文本。
    const auto token = text.substr(0, comma);

    // from_chars 成功后把整数写入 value。
    int value = 0;

    // [begin, end) 是左闭右开的字符范围。
    // 结构化绑定把返回结果拆成：
    // - end：实际解析停止的位置；
    // - error：转换过程的错误码。
    const auto [end, error] =
        std::from_chars(token.data(), token.data() + token.size(), value);

    // 两个条件缺一不可：
    // - error 非空：空 token、超出 int 范围等转换失败；
    // - end 未到 token 末尾：例如 "12x" 只解析了前面的 "12"。
    if (error != std::errc{} || end != token.data() + token.size()) {
      return std::string_view{"invalid integer"};
    }

    // push_back 保存本轮结果。
    // 容量不足时 vector 可能扩容，旧迭代器/引用/指针可能失效；
    // 本函数没有保存这些位置，因此扩容不会破坏后续逻辑。
    values.push_back(value);

    // 没有逗号说明刚解析的是最后一个 token。
    if (comma == std::string_view::npos) {
      break;
    }

    // 跳过“当前 token + 一个逗号”，下一轮只观察剩余文本。
    text.remove_prefix(comma + 1);
  }

  // values 作为 variant 的成功候选返回。
  // 编译器可以使用移动或返回值优化，vector 的动态数组不会被手工释放。
  return values;
}

std::optional<int> sum_if_not_empty(const std::vector<int>& values) {
  // 空输入不强行定义成 0，而是用 nullopt 表达“没有和值”。
  if (values.empty()) {
    return std::nullopt;
  }

  // accumulate 处理 [begin, end) 半开区间。
  // 初始值 0 是 int，因此累加器和返回值也是 int。
  // 大整数之和可能溢出；UBSan 只有在相关路径实际执行时才能发现。
  return std::accumulate(values.begin(), values.end(), 0);
}

}  // namespace stage1
