#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>

class SimulatedGpuBuffer {
 public:
  explicit SimulatedGpuBuffer(std::size_t size)
      : data_(std::make_unique<float[]>(size)), size_(size) {
    std::cout << "acquire " << size_ << " elements\n";
  }
  ~SimulatedGpuBuffer() {
    if (data_) std::cout << "release " << size_ << " elements\n";
  }
  SimulatedGpuBuffer(const SimulatedGpuBuffer&) = delete;
  SimulatedGpuBuffer& operator=(const SimulatedGpuBuffer&) = delete;
  SimulatedGpuBuffer(SimulatedGpuBuffer&& other) noexcept
      : data_(std::move(other.data_)), size_(std::exchange(other.size_, 0)) {
    std::cout << "move ownership\n";
  }
  SimulatedGpuBuffer& operator=(SimulatedGpuBuffer&&) = delete;
  std::size_t size() const { return size_; }

 private:
  std::unique_ptr<float[]> data_;
  std::size_t size_ = 0;
};

int main() {
  SimulatedGpuBuffer request_workspace(1024);
  SimulatedGpuBuffer execution_workspace(std::move(request_workspace));
  std::cout << "source_size=" << request_workspace.size() << '\n';
  std::cout << "target_size=" << execution_workspace.size() << '\n';
}
