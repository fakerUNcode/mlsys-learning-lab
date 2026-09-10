#include "runtime_demo.hpp"

#include <charconv>
#include <numeric>

namespace stage1 {

ParseResult parse_numbers(std::string_view text) {
  if (text.empty()) return std::string_view{"empty input"};

  std::vector<int> values;
  while (!text.empty()) {
    const auto comma = text.find(',');
    const auto token = text.substr(0, comma);
    int value = 0;
    const auto [end, error] =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc{} || end != token.data() + token.size()) {
      return std::string_view{"invalid integer"};
    }
    values.push_back(value);
    if (comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1);
  }
  return values;
}

std::optional<int> sum_if_not_empty(const std::vector<int>& values) {
  if (values.empty()) return std::nullopt;
  return std::accumulate(values.begin(), values.end(), 0);
}

}  // namespace stage1
