#include "smart_pointer_model.hpp" // [语法] 引入自定义头文件。[释义] 导入 Workspace 和 Model 的数据结构定义。
#include <iostream>                // [语法] 引入标准输入输出流。[释义] 用于在终端打印各种状态进行测试。
#include <memory>                  // [语法] 引入内存管理库。[释义] 提供现代 C++ 智能指针体系（unique/shared/weak_ptr）。
#include <utility>                 // [语法] 引入实用工具库。[释义] 提供 std::move 用于所有权的转移。

int main() {
  // [语法] using 声明语句。
  // [释义] 将指定命名空间中的特定类型引入当前作用域，避免后续代码中反复书写冗长的 stage1::smart_pointers:: 前缀。
  using stage1::smart_pointers::Model;
  using stage1::smart_pointers::Workspace;

  // =================================================================================
  // 阶段一：std::unique_ptr (独占所有权测试)
  // =================================================================================

  // [语法] std::make_unique<T>(args...) 是模板函数，在堆内存中动态分配对象并返回 unique_ptr。
  // [释义] 模拟一个新到达的推理请求。系统为其独占分配了一块容量为8 个 float（单精度浮点数）元素大小的计算暂存区 (Workspace)。
  auto request_workspace = std::make_unique<Workspace>(8);

  // [语法] std::move 将左值 request_workspace 强制转换为右值引用，触发 unique_ptr 的移动构造函数。
  // [释义] 转移所有权：原请求对象 (request_workspace) 交出内存控制权并被置空；
  //运行时的上下文 (active_workspace) 接管了这块暂存区。这一步彻底避免了内存拷贝。
  auto active_workspace = std::move(request_workspace);

  // [语法] 链式输出。std::boolalpha 强制布尔值打印为 true/false；(ptr == nullptr) 判断指针是否为空；-> 用于访问指针底层对象的成员。
  // [释义] 验证转移结果：确认原来的所有者已经被安全掏空（source_empty=true），且新的所有者能够正常访问这块 8 容量的内存。
  std::cout << "unique_ptr: source_empty=" << std::boolalpha
            << (request_workspace == nullptr)
            << ", scratch_size=" << active_workspace->scratch.size() << '\n';

  // =================================================================================
  // 阶段二：std::shared_ptr (共享所有权测试)
  // =================================================================================

  // [语法] std::make_shared<T>(args...) 在堆上同时分配控制块（用于计数的元数据）和实际对象，返回 shared_ptr。
  // [释义] 模拟系统启动，将名为 "demo-model" 的模型加载到主存中。目前只有一个“所有者”，强引用计数为 1。
  auto model_owner = std::make_shared<Model>("demo-model");

  // [语法] 隐式转换，用 shared_ptr 初始化 weak_ptr。这只会增加控制块中的“弱引用计数”，不改变“强引用计数”。
  // [释义] 将加载好的模型注册到全局缓存 (Cache) 中。缓存仅仅“旁观”这个模型，不会干涉它的释放逻辑。
  std::weak_ptr<Model> model_cache = model_owner;

  // [语法] 调用外部内联函数，底层执行了 model_cache.lock()，它原子性地返回一个新的 shared_ptr。
  // [释义] 模拟两个并发的推理任务 (request_a 和 request_b) 命中缓存并开始使用模型。此时，它们也成了合法的所有者，强引用计数上升到 3。
  auto request_a = stage1::smart_pointers::try_get_cached_model(model_cache);
  auto request_b = stage1::smart_pointers::try_get_cached_model(model_cache);

  // [语法] .use_count() 返回当前与 shared_ptr 共享底层对象所有权的数量。
  // [释义] 验证共享状态：模型正被并发读取，输出显示当前有 3 个实体在共同维持该模型的生命周期。
  std::cout << "shared_ptr: name=" << request_a->name
            << ", strong_owners=" << model_owner.use_count() << '\n';

  // =================================================================================
  // 阶段三：std::weak_ptr (弱观察者与安全销毁测试)
  // =================================================================================

  // [语法] .reset() 方法主动销毁当前的 shared_ptr 实例，强引用计数相应 -1。
  // [释义] 模拟生命周期结束：系统卸载模型，且两个并发推理请求也执行完毕。由于 3 个所有者全都放弃了控制权，强引用计数归零，模型占用的内存被操作系统真正回收。
  model_owner.reset();
  request_a.reset();
  request_b.reset();

  // [语法] 再次调用含 .lock() 的函数，将返回结果保存为不可变 (const) 变量。
  // [释义] 模拟一个新的请求尝试从缓存中获取刚才那个模型。由于原模型已被彻底销毁，lock() 防御生效，安全地返回了空指针 (nullptr)。
  const auto expired_model = stage1::smart_pointers::try_get_cached_model(model_cache);

  // [语法] .expired() 返回 bool，检查关联对象是否已被销毁；static_cast<bool> 将 shared_ptr 显式转为布尔值（空指针为 false）。
  // [释义] 验证过期机制：缓存系统成功察觉到模型已失效 (expired=true)，并且证实了刚才获取模型的尝试确实被安全拦截 (lock_succeeded=false)。
  std::cout << "weak_ptr: expired=" << model_cache.expired()
            << ", lock_succeeded=" << static_cast<bool>(expired_model) << '\n';

  // [语法] 三元条件运算符。如果 expired_model 可转换为 true（非空），则返回退出码 1；否则返回 0。
  // [释义] 防御性断言：如果本该过期的模型居然还能被取出来，说明存在内存泄漏或悬挂指针漏洞，程序将以状态码 1 (异常) 退出；如果一切符合预期，正常返回 0。
  return expired_model ? 1 : 0;
}