// 文件职责：运行 T04 的 CPU 所有权验证；Buffer 的实现在同目录 buffer.hpp。
// 执行入口 main 依次检查移动链路、自移动、空状态、容器迁移、异常展开；成功输出 PASS，失败抛异常并返回失败码。
// CMakeLists.txt 将本文件构建为 CTest 的被测程序。
// 验证遵守借用指针生命周期，不故意访问已释放内存；ASan/UBSan 和泄漏检查用于检查实际运行的路径，不代表对所有输入或多线程场景的形式证明。
#include "buffer.hpp"  // 引入本课定义；尖括号形式的其他 include 则引入标准库头文件。

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using stage2::Buffer;  // 让本文件中可直接写 Buffer，而不重复写 stage2::Buffer。

// static_assert 在编译期验证布尔条件；不满足就拒绝构建，不产生运行时检查。
// 标准 traits 模板的 <Buffer> 指定被检查类型，_v 直接取其布尔值，! 表示取反。
// 前两项要求禁止拷贝，后三项要求移动和析构不会抛异常，保证容器迁移与清理契约。
// 以 static_assert(!std::is_copy_constructible_v<Buffer>) 为例：<Buffer> 指定被检查类型，is_copy_constructible_v 的值回答“能否由拷贝构造 Buffer”，! 将真假取反，static_assert 要求最终结果为真。
// _v 是标准库为这些模板值选择的名字后缀，不是单独的运算符；<Buffer> 这里是模板实参语法，不是“小于 Buffer 大于”的比较表达式。
// is_copy_assignable_v 检查给已有对象拷贝赋值是否可用，is_nothrow_move_constructible_v 检查移动构造是否可用且不抛异常，后面两项分别检查移动赋值与析构的同样保证。
static_assert(!std::is_copy_constructible_v<Buffer>);
static_assert(!std::is_copy_assignable_v<Buffer>);
static_assert(std::is_nothrow_move_constructible_v<Buffer>);
static_assert(std::is_nothrow_move_assignable_v<Buffer>);
static_assert(std::is_nothrow_destructible_v<Buffer>);
// declval 在不真正创建对象的类型表达式中模拟一个 const Buffer 引用；decltype 取调用 data() 的结果类型，is_same_v 确认只读对象返回 const float* 而非可写指针。
// 从内向外读下一条检查：std::declval<const Buffer&>() 提供一个“假想的只读 Buffer 引用表达式”，.data() 据此选择只读重载，decltype(...) 取得该调用表达式的类型。
// is_same_v<第一个类型, 第二个类型> 比较两个类型是否完全相同；这里两个类型应当都是 const float*，所以 static_assert 允许编译继续。
// decltype 的括号内部在这里不求值，因此不会真正创建 Buffer、调用 data() 或分配数组；declval 也不能像普通运行时函数一样单独执行。
static_assert(std::is_same_v<decltype(std::declval<const Buffer&>().data()),
                             const float*>);

namespace {  // 匿名命名空间：以下辅助函数仅供这个编译单元使用，main 在后面调用。

// 检查入口：各验证函数传入是否符合预期的 condition 和失败信息 message，无返回值。
// const char* 借用字符串字面量的地址，不管理它的寿命；字面量在整个程序期间有效。
// 条件为假时 throw 抛异常，main 统一报告失败；为真则正常返回。
// 不使用 assert，避免 Release 构建定义 NDEBUG 后关键运行检查被移除。
// void check(bool condition, const char* message) 依次声明“无返回值”的函数、函数名、两个用逗号分隔的参数；调用 check(判断式, "文字") 时，实参依次绑定 condition 和 message。
// if (!condition) 中的 ! 对布尔值取反；判断失败时进入大括号，否则跳过这一段，因此一次成功检查不会抛异常。
// throw std::runtime_error(message) 构造一个带说明文字的异常并抛出；控制流停止沿当前函数往下走，开始寻找能处理它的 catch，并清理沿途退出作用域的局部对象。
void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// 由移动与空状态验证调用；const Buffer& 借用现有对象，不拷贝它且不能改其状态。
// 验证空指针与零长度同时成立；&& 是逻辑“并且”，两个条件缺一不可，无返回值。
// buffer.data() == nullptr && buffer.size() == 0 先比较指针，为真才继续比较长度，这叫短路求值；此处 && 连接两个布尔表达式，与头文件中 Buffer&& 的引用类型语法不同。
void check_empty(const Buffer& buffer) {
  check(buffer.data() == nullptr && buffer.size() == 0,
        "empty buffer invariant failed");
}

// 由 main 调用，无参数或返回值；构造 a，再把同一数组经 b 转交 c。
// 验证地址、长度、元素值均保留；局部 Buffer 自主管理资源，p 只借用数组地址。
void ownership_chain() {
  // Buffer a(4) 是局部变量的直接初始化：Buffer 是类型，a 是变量名，(4) 把元素数传给构造函数，分号结束这条声明。
  Buffer a(4);
  // i 是元素下标：从 0 开始，每次 ++i 增加 1，到 i == size() 停止。
  // data()[i] 访问基地址后第 i 个 float；不是把字节数当下标。
  // 每轮先验证零初值，再写入 i+1，形成 [1,2,3,4]，供移动后的检查辨别是否保留了原数据。
  // for (初始化; 条件; 更新) 的两个分号分隔三个部分；i 只初始化一次，每轮先判断 i < a.size()，执行循环体后再执行 ++i，条件为假时退出。
  // a.data()[i] 先以 . 调用 a 的 data() 函数取得 float*，再用 [i] 取得第 i 个元素；左侧用于赋值时，改变的是数组元素而非指针或长度。
  for (std::size_t i = 0; i < a.size(); ++i) {
    check(a.data()[i] == 0.0F, "array was not value-initialized");
    // static_cast<float> 明确把整数转为 float；本题的 1～4 均能被 float 精确表示。
    // static_cast<目标类型>(原表达式) 中，<float> 指定要转换成的类型，(i + 1) 是待转换的值；这与 make_unique<float[]>(size) 的“传入类型再传入参数”外观相似，但 static_cast 是语言内建转换语法。
    a.data()[i] = static_cast<float>(i + 1);
  }
  float* p = a.data();  // 只观察，不接管，也不手动 delete。
  // Buffer b(std::move(a)) 中，外层括号为 b 的构造传参，内层括号调用 std::move(a)，随后编译器据参数形式选择 Buffer 的移动构造函数。
  Buffer b(std::move(a));  // 新建 b，调用移动构造；数组地址保留，a 变空。
  check_empty(a);
  // b.data() == p 比较两个指针是否保存同一地址，不是比较数组内容；还需另行逐元素检查，才能确认移动后数值保留。
  check(b.data() == p && b.size() == 4, "move construction lost ownership");
  Buffer c(2);
  c.data()[0] = 99.0F;
  c = std::move(b);  // 旧的 2 元素数组应在这里释放；泄漏由 ASan 检查。
  check_empty(b);
  check(c.data() == p && c.size() == 4, "move assignment lost ownership");
  // 逐元素核对：目标长度仍为 4，所以检查恰好覆盖原数组，且不会越界。
  for (std::size_t i = 0; i < c.size(); ++i) {
    check(c.data()[i] == static_cast<float>(i + 1), "move lost data");
  }
  // 所有者变了，但数组还活着，因此原借用指针 p 仍可用。
  // F 后缀指定 float 字面量。
  p[1] = 42.0F;
  check(c.data()[1] == 42.0F, "observer does not refer to transferred array");
  // 离开函数逆序析构 c、b、a：c 释放 4 元素数组，b/a 为空，无数组可释放。
  // 此后 p 所保存的地址已悬空，不能再解引用；局部裸指针结束寿命不负责释放数组。
}

// 由 main 调用，无参数或返回值；同一个对象作为移动赋值的来源与目标。
// 本类约定自移动保持数据，所以借用其指针并核对赋值前后的地址、长度和值。
void self_move() {
  Buffer buffer(3);
  buffer.data()[0] = 7.0F;
  float* p = buffer.data();
  Buffer& same = buffer;  // & 在声明中表示引用：same 是 buffer 的别名，不新建对象。
  // same 绑定后不能改为引用另一个对象；std::move(same) 仍表示同一个 buffer 对象，所以赋值函数中的 this 与 &other 相等，会跳过真正转移的分支。
  buffer = std::move(same);
  check(buffer.data() == p && buffer.size() == 3 && buffer.data()[0] == 7.0F,
        "self move changed the buffer");
}

// 由 main 调用，无参数或返回值；覆盖零长度、空/非空来源与目标，以及移动后复用。
// 所有数组都由局部 Buffer 管理；每次转移后检查来源，防止只清指针却遗留旧长度。
void empty_transitions_and_reuse() {
  Buffer empty(0);
  check_empty(empty);
  Buffer moved_empty(std::move(empty));
  check_empty(empty);
  check_empty(moved_empty);

  // 空目标接管非空来源；随后接管空来源，必须同时放弃并释放此前的 3 元素数组。
  Buffer full(3);
  moved_empty = std::move(full);
  check_empty(full);
  check(moved_empty.size() == 3, "empty target did not acquire array");
  moved_empty = std::move(empty);
  check_empty(moved_empty);  // 非空目标接收空来源，也必须释放旧数组。
  check_empty(empty);
  moved_empty = Buffer(2);  // 临时 Buffer 拥有新数组；赋值移动后，临时对象为空并可安全析构。
  moved_empty.data()[1] = 8.0F;
  check(moved_empty.size() == 2 && moved_empty.data()[1] == 8.0F,
        "moved-from object could not be reused");
}

// 由 main 调用，验证 vector 扩容重新安置 Buffer 时会移动其所有权。
// 先记录实际 capacity，再请求比它多一个槽位；标准保证容量增大时重新分配，原 Buffer 对象因此会被迁移。
// 原数组由迁移后的元素继续管理，数组本身不搬家，所以借用地址和值都应保留。
void vector_relocation() {
  std::vector<Buffer> buffers;
  buffers.reserve(1);  // reserve 增加可容纳元素的槽位但不构造元素；此时 size 为 0、capacity 至少为 1。
  buffers.emplace_back(2);  // 在 vector 内直接构造 Buffer(2)，避免先造临时对象再移动。
  // vector 的 [0] 访问第一个 Buffer；其 data()[0] 再访问该 Buffer 数组的第一个 float。
  buffers[0].data()[0] = 13.0F;
  float* original_data = buffers[0].data();

  const std::size_t old_capacity = buffers.capacity();  // const 让快照保持只读，不会在发起扩容前被意外改写。
  // capacity() 返回当前槽位数；请求 old_capacity+1 严格大于当前容量，标准因此要求重新分配。
  buffers.reserve(old_capacity + 1);
  check(buffers.size() == 1 && buffers[0].data() == original_data &&
            buffers[0].data()[0] == 13.0F,
        "vector relocation changed the owned array");
}

// 由 main 调用，故意在 Buffer 存活期间抛出异常，观察栈展开是否调用析构函数。
// catch 只接住本函数故意抛出的标记异常；若 Buffer 的析构未释放数组，泄漏检查会报告。
void exception_cleanup() {
  bool caught_expected = false;  // bool 保存 true/false；初值 false 表示还没有接住预期异常。
  // try 大括号中一旦 throw，后续语句跳过；局部 temporary 先析构，再进入匹配的 catch。
  try {
    Buffer temporary(5);
    temporary.data()[0] = 17.0F;
    throw std::runtime_error("intentional T04 unwind");
  } catch (const std::runtime_error& error) {
    // error 是异常对象的只读引用；what() 借出说明文字，string 将其转成可按内容比较的字符串。
    caught_expected = std::string(error.what()) == "intentional T04 unwind";
  }
  check(caught_expected, "intentional exception was not caught");
}

}  // namespace

// 程序入口：按顺序执行所有运行期检查；成功返回 0，任一检查抛异常就打印错误并返回失败码。
// int 是 main 的返回类型；argc/argv 本练习不需要，因此用空参数列表；EXIT_SUCCESS/EXIT_FAILURE 来自 <cstdlib>。
int main() {
  // 顺序调用各个无参数检查函数；任一 check 抛出的异常都会跳到同一 catch，避免漏报错误。
  try {
    ownership_chain();
    std::cout << "PASS ownership chain, initialization and observer\n";
    self_move();
    std::cout << "PASS self move\n";
    empty_transitions_and_reuse();
    std::cout << "PASS empty transitions and moved-from reuse\n";
    vector_relocation();
    std::cout << "PASS vector relocation\n";
    exception_cleanup();
    std::cout << "PASS exception cleanup\n";
  } catch (const std::exception& error) {
    // std::exception 是标准异常的基类引用；what() 提供失败说明，cerr 写到错误输出流。
    std::cerr << "FAIL T04 runtime check: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "T04 runtime checks passed\n";
  return EXIT_SUCCESS;
}
