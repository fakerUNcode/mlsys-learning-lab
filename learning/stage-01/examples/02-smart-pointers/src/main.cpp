#include <iostream>
#include <memory>
#include <string>
#include <utility>

struct Model {
  explicit Model(std::string model_name) : name(std::move(model_name)) {
    std::cout << "load " << name << '\n';
  }
  ~Model() { std::cout << "unload " << name << '\n'; }
  std::string name;
};

int main() {
  std::weak_ptr<Model> cache;
  {
    auto request_a = std::make_shared<Model>("demo-model");
    cache = request_a;
    auto request_b = request_a;
    std::cout << "owners=" << request_a.use_count() << '\n';
  }
  std::cout << "cache_expired=" << std::boolalpha << cache.expired() << '\n';
}
