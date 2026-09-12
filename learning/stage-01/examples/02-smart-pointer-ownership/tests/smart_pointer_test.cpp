#include "smart_pointer_model.hpp" // [语法] 引入自定义的智能指针与模型结构定义。

#include <cassert> // [语法] 引入断言库。[释义] 用于在运行时验证假设条件，如果括号内的表达式为 false，程序会直接报错并终止，非常适合写单元测试。
#include <memory>  // [语法] 引入智能指针组件（unique_ptr, shared_ptr, weak_ptr 等）。
#include <utility> // [语法] 引入实用工具，主要为了使用 std::move。

int main() {
  // [语法] using 声明，简化后续代码，避免反复书写外层命名空间。
  using stage1::smart_pointers::Model;
  using stage1::smart_pointers::Workspace;

  // =================================================================================
  // 阶段一：独占所有权与移动语义测试 (std::unique_ptr)
  // =================================================================================

  // [语法] make_unique 在堆上分配一个容量为 4 的 Workspace，返回 unique_ptr。
  // [释义] 系统接收到一个新请求，为其专门开辟了一块包含 4 个 float 的独立工作区。
  auto original_workspace = std::make_unique<Workspace>(4);

  // [语法] std::move 将左值强制转换为右值，触发移动构造，将底层内存指针的所有权进行交接。
  // [释义] 模拟上下文切换：原变量交出内存控制权，新变量无损接管这块工作区。
  auto moved_workspace = std::move(original_workspace);

  // [语法] assert 断言原指针现在必须等于 nullptr。
  // [释义] 核心验证 1：确保被 move 之后，原来的“主人”已经被彻底掏空，防止悬挂指针。
  assert(original_workspace == nullptr);

  // [语法] assert 断言新指针不为空。
  // [释义] 核心验证 2：确保新的“主人”确实拿到了控制权。
  assert(moved_workspace != nullptr);

  // [语法] 访问新指针底层 vector 的 size() 方法，断言其大小必须为 4。
  // [释义] 核心验证 3：确保所有权转移过程中，底层数据（4个元素）毫发无损。
  assert(moved_workspace->scratch.size() == 4);

  // =================================================================================
  // 阶段二：共享所有权与弱观察者测试 (std::shared_ptr & std::weak_ptr)
  // =================================================================================

  // [语法] make_shared 在堆上分配控制块和 Model 对象，强引用计数初始化为 1。
  // [释义] 模拟系统主线程加载了一个名为 "test-model" 的模型。
  auto model_owner = std::make_shared<Model>("test-model");

  // [语法] 用 shared_ptr 给 weak_ptr 赋值，这只会增加弱引用计数，强引用计数不变。
  // [释义] 将模型挂载到系统的缓存模块中。缓存充当旁观者，不干预模型的生命周期。
  std::weak_ptr<Model> cache_entry = model_owner;

  // [语法] use_count() 获取当前强引用计数，断言其必须为 1。
  // [释义] 验证缓存的旁观者身份：确认存入缓存并没有导致所有者增加。
  assert(model_owner.use_count() == 1);

  // [语法] expired() 检查 weak_ptr 观察的对象是否已经被销毁，这里断言为 false（未销毁）。
  // [释义] 验证缓存有效性：此时模型还在内存中，缓存记录是有效的。
  assert(!cache_entry.expired());

  // [语法] 调用外部函数（内部执行了 lock()），成功提权并各自返回一个新的 shared_ptr。
  // [释义] 模拟并发场景：请求 A 和请求 B 同时命中缓存，并分别拿到了模型的使用权。
  auto request_a = stage1::smart_pointers::try_get_cached_model(cache_entry);
  auto request_b = stage1::smart_pointers::try_get_cached_model(cache_entry);

  // [语法] 断言 request_a 和 request_b 都可以作为布尔值 true（非空指针）。
  // [释义] 验证提权结果：确保两个请求都成功获取到了实际可用的模型。
  assert(request_a && request_b);

  // [语法] 再次断言强引用计数，此时必须等于 3。
  // [释义] 验证共享状态：模型此时正被“主线程、请求A、请求B”三方共同持有和保护。
  assert(model_owner.use_count() == 3);

  // =================================================================================
  // 阶段三：生命周期衰减与安全回收测试
  // =================================================================================

  // [语法] 主动销毁 model_owner 这个 shared_ptr，强引用计数 3 -> 2。
  // [释义] 模拟系统决定卸载模型，主线程放弃了该模型的所有权。
  model_owner.reset();

  // [语法] 断言缓存仍然未过期 (!expired)。
  // [释义] 验证延期释放机制：虽然主线程卸载了模型，但因为请求 A 和 B 还在用，底层内存没有被销毁。
  assert(!cache_entry.expired());

  // [语法] 销毁 request_a，强引用计数 2 -> 1。
  // [释义] 模拟请求 A 计算完成，释放资源。
  request_a.reset();

  // [语法] 再次断言缓存未过期。
  // [释义] 因为请求 B 还在运行，模型依然安全存活。
  assert(!cache_entry.expired());

  // [语法] 销毁 request_b，强引用计数 1 -> 0，触发堆内存释放。
  // [释义] 模拟最后一个请求 B 也执行完毕。随着最后一个所有者退出，模型真正从内存中被清理。
  request_b.reset();

  // [语法] 断言 expired() 为 true。
  // [释义] 验证内存安全：缓存敏锐地察觉到模型已经被彻底销毁（已过期）。
  assert(cache_entry.expired());

  // [语法] 尝试再次 lock()，断言其返回的 shared_ptr 为空（转为 bool 时为 false，取反为 true）。
  // [释义] 验证无悬挂指针：在模型完全释放后，如果再有新请求试图通过缓存拿模型，会被安全地拦截并返回空指针，彻底杜绝了程序崩溃的隐患。
  assert(!stage1::smart_pointers::try_get_cached_model(cache_entry));

  return 0;
}