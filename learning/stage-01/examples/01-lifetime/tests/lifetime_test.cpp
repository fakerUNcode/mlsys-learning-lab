#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

class Buffer {
 public:
  explicit Buffer(std::size_t size)
      : data_(std::make_unique<float[]>(size)), size_(size) {}
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept
      : data_(std::move(other.data_)), size_(std::exchange(other.size_, 0)) {}
  Buffer& operator=(Buffer&&) = delete;
  std::size_t size() const { return size_; }

 private:
  std::unique_ptr<float[]> data_;
  std::size_t size_ = 0;
};

int main() {
  Buffer source(128);
  Buffer target(std::move(source));
  assert(source.size() == 0);
  assert(target.size() == 128);
}
