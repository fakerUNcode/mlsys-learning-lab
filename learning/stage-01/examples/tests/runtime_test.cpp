#include "runtime_demo.hpp"

#include <cassert>
#include <string_view>
#include <vector>

int main() {
  auto parsed = stage1::parse_numbers("2,3,5");
  assert(std::holds_alternative<std::vector<int>>(parsed));
  const auto& values = std::get<std::vector<int>>(parsed);
  assert(stage1::sum_if_not_empty(values) == 10);

  auto invalid = stage1::parse_numbers("2,x");
  assert(std::get<std::string_view>(invalid) == "invalid integer");
  assert(!stage1::sum_if_not_empty({}).has_value());
}
