#include "runtime_demo.hpp"

#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  const std::string_view input = argc > 1 ? argv[1] : "1,2,3,4";
  auto result = stage1::parse_numbers(input);

  if (const auto* error = std::get_if<std::string_view>(&result)) {
    std::cerr << "status=" << *error << '\n';
    return 1;
  }

  auto values = std::make_unique<std::vector<int>>(
      std::get<std::vector<int>>(std::move(result)));
  const auto sum = stage1::sum_if_not_empty(*values);
  std::cout << "count=" << values->size() << " sum=" << sum.value_or(0)
            << '\n';
  std::cout << "status=ok\n";
  return 0;
}
