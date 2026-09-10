#pragma once

#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace stage1 {

using ParseResult = std::variant<std::vector<int>, std::string_view>;

ParseResult parse_numbers(std::string_view text);
std::optional<int> sum_if_not_empty(const std::vector<int>& values);

}  // namespace stage1
