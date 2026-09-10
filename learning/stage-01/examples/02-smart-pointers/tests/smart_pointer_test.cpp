#include <cassert>
#include <memory>

int main() {
  std::weak_ptr<int> observer;
  {
    auto first = std::make_shared<int>(42);
    observer = first;
    auto second = first;
    assert(first.use_count() == 2);
    const auto alive = observer.lock();
    assert(alive && *alive == 42);
  }
  assert(observer.expired());
  assert(!observer.lock());
}
