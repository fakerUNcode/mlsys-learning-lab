#pragma once // 预处理指令，保证此头文件在同一个编译单元中只被包含一次，防止重复定义

#include <cstddef>  // 引入 std::size_t
#include <memory>   // 引入现代 C++ 智能指针体系：unique_ptr, shared_ptr, weak_ptr
#include <string>
#include <utility>  // 引入 std::move
#include <vector>

//等价于早期 C++ 版本中繁琐的嵌套写法：
// namespace stage1 {
//    namespace smart_pointers {
        // xxx
//    }
// }
namespace stage1::smart_pointers {

// ============================================================================
// 【独占所有权模式】
// 场景：神经网络前向传播（Forward Pass）时需要的临时计算空间（Scratchpad）。
// Workspace 只归一个推理请求所有，适合用 std::unique_ptr 表达独占和可转移所有权。
// ============================================================================
struct Workspace {
  // explicit专门用于单参数构造函数，严禁编译器在暗中进行隐式的类型转换（如 Workspace w = 1000;）。
  // element_count 使用 std::size_t，明确表达“内存容量大小”的语义，防止负数或溢出。
  //:（冒号）：在此语境下，它是成员初始化列表的启动符，用于分隔参数列表与初始化动作。
  // scratch(element_count) [触发成员构造] 定位到结构体内部名为 scratch 的 std::vector<float> 变量，将 element_count 透传给该容器的构造函数，一步到位分配出所需数量的连续内存。
  explicit Workspace(std::size_t element_count) : scratch(element_count) {}

  // 用于存放算子（Operator）计算时的中间激活值（Activations）。
  // 在高并发服务中，这块内存通常会被 unique_ptr 独占保护，避免不同请求之间的数据发生读写冲突（Data Race）。
  std::vector<float> scratch;
};

// ============================================================================
// 【共享所有权模式】
// 场景：表示被加载到显存/主存中的神经网络模型（如权重参数）。
// 多个并发的推理请求可以同时读取同一份权重，因此非常适合由 std::shared_ptr 管理。
// ============================================================================
struct Model {
  // std::move 触发移动语义：接收外部传入的字符串时，直接“偷取”底层堆内存指针，
  // 避免了一次深拷贝，这在加载成千上万个对象时能显著降低开销。
  explicit Model(std::string model_name) : name(std::move(model_name)) {}

  std::string name;

  // 模拟神经网络的权重矩阵（Weights）。
  // 在真实系统中，这里可能是几 GB 大小的参数。通过 shared_ptr 共享，
  // 无论有多少个并发请求，内存中都始终只有这一份权重。
  std::vector<float> weights{0.25F, 0.5F, 0.75F};
};

// ============================================================================
// 【弱观察者模式】
// 场景：模型缓存管理（Model Cache）。
// 缓存系统如果直接持有 shared_ptr，会导致模型引用计数永远不为 0，占用的大内存永远无法释放。
// ============================================================================
// 缓存只使用 std::weak_ptr“观察”模型，不增加引用计数，不延长模型寿命；
// 取用时调用 lock() 会尝试取得一个强引用（shared_ptr）。

//inline[请求内联展开] 提示编译器在编译时将此函数直接嵌入调用点，消除压栈出栈的函数调用开销。
inline std::shared_ptr<Model> try_get_cached_model(
    const std::weak_ptr<Model>& cached_model) {

  // return cached_model.lock(); [执行状态提升] 调用弱引用的专属函数，生成并返回一个强引用对象。lock() 是一个原子级（Thread-Safe）的操作：
  // 1. 如果模型还在内存中（引用计数 > 0），它会返回一个有效的 shared_ptr，此时你可以安全地使用模型。
  // 2. 如果模型已经被其他模块释放（比如系统内存不足导致模型被逐出），它会返回 nullptr。
  // 这让你可以在不干扰模型生命周期的前提下，安全地探测并使用资源。
  return cached_model.lock();
}

}  // namespace stage1::smart_pointers