# T04：Buffer 所有权短验收与完整程序导读

状态：T04 的工程检查于 2026-10-08 通过。本版把材料整理成前置知识、类实现、验证程序三段，供第一次接触这份 C++ 程序的读者从头学习。

产物：[Buffer 类](buffer.hpp)、[完整运行程序](buffer.cpp)、[独立构建配置](CMakeLists.txt)、[验证报告](../../../../reports/stage-02/2026-10-08-t04-validation.md)。

## 阅读路线与程序做什么

先读本页“前置知识”。读完后，按这个顺序打开源码：

1. [buffer.hpp](buffer.hpp)：读数组所有权约定、构造/移动/访问接口和私有成员。
2. [buffer.cpp](buffer.cpp)：从 `#include` 和编译期检查开始，依次读通用检查函数、四个运行场景，最后读 `main()`。
3. [CMakeLists.txt](CMakeLists.txt)：了解如何把 `buffer.cpp` 编译成程序并交给 CTest 运行。

程序本身是一组 CPU 检查，不是完整容器库。`Buffer` 独占一个 `float` 动态数组并记录长度；程序检查移动后所有权和数据是否保留、移动来源是否为空、自移动、空对象转换、vector 扩容和异常清理。程序没有 CUDA、并发或越界防护。

## 前置知识：读完这些就能逐段读源码

以下内容是读这份程序所需的完整最小集合。每节给出语法、它在本程序里操作什么，以及这样写的原因。遇到相同语法后文不再重复定义，只说明本处的新用途。

### 1. C++ 文件、语句与函数

```cpp
#include <iostream>
int double_value(int input) {
  return input * 2;
}
```

- `#include <iostream>` 在编译前引入标准库的输入输出声明，之后才能用 `std::cout`。引号形式如 `#include "buffer.hpp"` 用于引入本项目头文件。
- `int` 是类型；`double_value` 是函数名；括号里声明输入参数 `input`；花括号包住函数执行内容；`return` 把结果交回调用处。`void` 表示函数不返回值。
- 一条普通语句通常以分号结束。逗号可分隔参数或成员初始化项；大括号界定函数、类、命名空间或控制分支的范围。
- `std::` 表示去标准库的 `std` 命名空间查找名字，例如 `std::size_t`、`std::vector`。T04 自己把类型放进 `stage2` 命名空间，因此全名为 `stage2::Buffer`。

### 2. 变量、类型、条件、循环和下标

```cpp
std::size_t count = 4;
if (count != 0) { /* ... */ }
for (std::size_t i = 0; i < count; ++i) { /* ... */ }
```

- `std::size_t count = 4;` 声明长度变量。`=` 在声明里提供初始值；在 `c = std::move(b);` 中则是赋值。
- `if (条件)` 在条件为真时执行紧随其后的语句或大括号代码。`==`、`!=`、`<` 比较值并产生真假；`!` 取反；`&&` 是逻辑“并且”，两侧都真时结果才真，并会在左侧为假时跳过右侧（短路求值）。
- `for (初始化; 继续条件; 每轮更新)` 先初始化一次，再判断条件；条件为真就执行循环体，随后执行更新再判断。T04 中 `++i` 每轮将索引加一，`i < size()` 使最后一次合法下标是 `size()-1`。
- `data()[i]` 先调用 `data()` 取得首元素地址，再用 `[i]` 访问从零开始的第 `i` 个 `float`。它不是字节偏移；只有 `0 <= i < size()` 才能访问。
- `static_cast<float>(i + 1)` 明确把整数表达式转换成 `float`。T04 的 1 到 4 可被 float 精确表示，因此可直接比较。
- 数值字面量后缀 `F`（例如 `0.0F`）表示 `float`；不带后缀的 `0.0` 通常是 `double`。

### 3. 对象、作用域、类与访问权限

```cpp
Buffer a(4);
```

这里声明一个名为 `a` 的局部对象，并用 `4` 直接调用 `Buffer` 构造函数。对象的作用域是它所在的大括号区域；离开区域时自动销毁。局部对象通常放在自动存储期，不需要手动调用析构。

`class Buffer { public: ... private: ... };` 定义一种类型。`public:` 后的成员可以由外部代码调用；`private:` 后的成员只能由类自身访问。类末尾必须有分号。成员函数通过 `a.data()` 的点号调用；类的成员函数也能访问另一个同类对象的私有成员，如移动时的 `other.data_`。

构造函数名与类相同、不写返回类型，在新对象创建时自动运行；析构函数写作 `~Buffer()`，在对象销毁时自动运行。`explicit` 阻止某些隐式转换，例如不允许把整数 `4` 悄悄当作 `Buffer`；直接写 `Buffer a(4)` 仍然可以。

### 4. 指针、引用、const 与借用

```cpp
float* p = a.data();
Buffer& alias = a;
const Buffer& read_only = a;
```

- `float*` 是保存 `float` 地址的指针类型。`float* p` 中的星号属于类型；`p[0]` 才访问地址所指的数组元素。`&` 若出现在 `Buffer&` 这样的类型里表示引用；若写作 `&object` 则是取对象地址。
- 引用是已有对象的另一个名字，不会复制对象，也不能之后改绑到别的对象。`Buffer& alias=a` 因而让 `alias` 与 `a` 是同一个对象。
- `const Buffer&` 是不复制对象的只读引用，常用来把现有对象传给函数，同时避免函数改动它。`const float*` 表示不能通过此指针改元素；`float* const` 则是指针本身不能改指向，含义不同。
- `const std::size_t old_capacity = ...;` 中，类型名前的 `const` 让这个局部长度快照初始化后不能再被赋新值；它确保扩容请求仍基于刚读取的容量。
- `nullptr` 表示空指针。空指针没有可访问元素。裸指针如 `p` 只是地址观察者，不拥有也不释放数组；它只能在拥有数组的 `Buffer` 仍然存活时使用。
- `this` 是成员函数中指向当前对象的指针；`*this` 表示当前对象。单独的 `*` 在 `float*` 中声明指针，在 `*this` 中解引用指针，需看它所在语法位置。

### 5. 动态数组、RAII 与生命周期

数组需在运行时按传入长度创建。手动 `new[]/delete[]` 容易在异常或多条返回路径漏释放或重复释放；本程序将资源交给 `std::unique_ptr<float[]>`。这是一种只能由一个智能指针拥有的数组管理对象：离开作用域时自动用匹配的 `delete[]` 释放。

这就是本例的 RAII：对象构造时取得资源，对象析构时归还资源。资源释放绑定在对象生命周期上，因此正常执行和异常退出局部作用域都会清理。`std::make_unique<float[]>(size)` 创建并返回拥有 `size` 个 float 的智能指针；数组元素值初始化为 `0.0F`。本类约定 `size==0` 时不分配，保存 `nullptr` 和长度 0。

`std::unique_ptr<float[]>` 的类型参数 `float[]` 说明它管理数组，而运行时长度放在函数调用括号里：`make_unique<float[]>(size)`。模板尖括号 `<...>` 填类型信息，圆括号 `(...)` 传具体运行值。

### 6. 拷贝、移动与重载

`Buffer b(a)` 要从现有 `a` 创建新对象，属于拷贝构造；`c = a` 则给已存在的 `c` 赋值，属于拷贝赋值。二者不是同一个操作。若浅拷贝指针，两个 Buffer 会误以为都拥有同一数组，析构时可能重复释放；本程序选择显式禁止两种拷贝。

移动是把资源管理责任交给另一个对象，不逐个复制数组元素。`std::move(a)` 本身只把表达式转换成可用于选择移动操作的形式，不会自行转移资源；真正执行转移的是匹配到的移动构造或移动赋值函数。类会用 `Buffer(Buffer&& other)` 接收移动构造来源，用 `Buffer& operator=(Buffer&& other)` 接收移动赋值来源。这里类型里的 `&&` 是右值引用，不是逻辑“并且”。同名函数依参数类型区分，叫重载。

`std::move` 后的来源对象仍存在，必须可安全析构。本类额外明确约定来源成为 `data()==nullptr`、`size()==0`。`std::exchange(old, 0)` 返回旧值并把 `old` 改成 0，正好用于移动长度。`noexcept` 表示函数承诺不把异常传出；只有内部操作确实不会抛出时才能这样承诺。

### 7. 函数返回、编译期检查与模板

- 函数参数写在括号里，调用时按位置接收实参。`const Buffer&` 参数借用现有对象；`const char*` 在 `check` 中借用字符串字面量，字面量的生命周期覆盖整个程序。
- `Buffer&` 是返回引用的类型。移动赋值返回 `*this`，即返回刚被赋值的目标对象本身，避免复制 Buffer。
- `[[nodiscard]]` 是编译器提示属性，提醒调用方不要无意忽略返回值；`noexcept` 是异常承诺，二者不改变核心的所有权逻辑。
- `static_assert(条件);` 在编译时验证条件，条件为假则拒绝构建。`std::is_copy_constructible_v<Buffer>` 等标准库 trait 是编译期布尔值，回答某个类型是否可拷贝、是否可无异常移动等问题。`!` 在这里取布尔值反。
- `std::declval<T>()` 只用于编译期类型表达式，假设有一个 T 类型表达式而不真的构造对象；`decltype(表达式)` 取得表达式类型，`std::is_same_v<A,B>` 比较两种类型是否相同。T04 用它确认只读 Buffer 的 `data()` 返回 `const float*`。模板类型放在尖括号里；不要把此处 `<Buffer>` 读成小于号和大于号比较。

### 8. 异常、容器和程序入口

- `throw std::runtime_error("说明")` 抛出带错误信息的异常。`try { ... } catch (const std::runtime_error& error) { ... }` 执行 try 代码；发生匹配异常时，先销毁离开作用域的局部对象，再运行 catch。这个自动清理过程叫栈展开。
- `std::vector<Buffer>` 是可增长的 Buffer 顺序容器。`reserve(n)` 请求至少 n 个元素的容量；当它重新分配内部存储时，原有元素需要移动到新位置。`emplace_back(2)` 在容器位置上直接调用 `Buffer(2)` 构造元素，避免额外临时对象。
- `namespace { ... }` 是匿名命名空间，使其中的辅助函数只在当前 `.cpp` 编译单元可见。`using stage2::Buffer;` 让本文件后面可以写 `Buffer`，不用每次写全名。
- `int main()` 是程序入口。返回 `EXIT_SUCCESS`（通常为 0）表示成功；`EXIT_FAILURE` 表示失败。`std::cout` 输出普通信息，`std::cerr` 输出错误；`<<` 把右侧文字或数值送入左侧输出流，可以连续串接；`\n` 换行。

## 前置知识之后：设计选择如何落实为程序保证

下面按 `buffer.hpp` 的代码顺序编号。表中的“没这样写”是针对本程序的具体替代方式说明；它不表示所有 C++ 类都必须采用相同设计。

| 编号 | 本程序的写法 | 它提供的具体行为 | 如果省略或换写法 |
|---|---|---|---|
| H1 | `explicit Buffer(std::size_t size)` | 要求调用方明确写 `Buffer a(4)`；整数不能隐式转换成 Buffer。 | 若移除 `explicit`，`Buffer a = 4` 可能隐式调用分配数组的构造函数，容易把整数误当资源长度。 |
| H2 | `size == 0 ? nullptr : std::make_unique<float[]>(size)` | 零长度约定为空指针；非零长度分配数组并把 float 元素值初始化为 0。 | 若改成默认初始化的 `new float[size]`，基本类型元素没有零初值；若直接用单对象 unique_ptr，则类型和释放方式都不匹配数组。 |
| H3 | `Buffer(const Buffer&) = delete` | 明确禁止从已有 Buffer 创建拷贝。 | `unique_ptr` 成员本身也会使编译器生成的拷贝构造不可用；显式删除让本类接口意图清晰且诊断直接。若自行浅拷贝裸地址，两个对象会错误共享唯一资源。 |
| H4 | `Buffer& operator=(const Buffer&) = delete` | 明确禁止给已有 Buffer 做拷贝赋值。 | `unique_ptr` 同样不可拷贝，因此默认拷贝赋值不可用；若手写浅拷贝赋值，会丢失原目标管理的地址并制造两个假所有者。 |
| H5 | `Buffer(Buffer&& other) noexcept` | 为新目标提供移动构造，并声明此操作不会把异常传出。 | 删除此移动接口后，`Buffer b(std::move(a))` 会尝试匹配被删除的拷贝构造而失败。`noexcept` 是本类对成员操作的承诺，必须由真实实现支撑。 |
| H6 | `data_(std::move(other.data_))` | 让 unique_ptr 的移动构造把同一数组的管理权转给新对象；数组元素和地址不变。 | 若把它写成从 `other.data_` 普通拷贝，因 unique_ptr 不可拷贝而编译失败。默认化整个 Buffer 移动构造也能移动这个成员，但还需要处理长度成员的不变量（见 H7）。 |
| H7 | `size_(std::exchange(other.size_, 0))` | 目标取得旧长度，同时将来源长度清零，让来源满足“空指针 + 长度 0”。 | 默认成员移动对 `std::size_t` 只会复制数字；指针成员虽为空，来源长度仍可能是旧值，形成内部状态不一致。 |
| H8 | `data_ = std::move(other.data_)` 后更新长度 | 给已存在目标赋值时，unique_ptr 移动赋值负责释放目标旧数组并接管来源数组；接着长度一起转移。 | 若只替换长度而不正确交接 data_，可能泄漏目标旧数组或让指针与长度描述不同资源。 |
| H9 | `if (this != &other)` | 同一对象作为来源和目标时跳过资源转移；本类约定自移动保留原值。 | 移除判断后，代码就不再主动保证自移动保持原值；也不应依赖某个成员类型自身的自移动细节来替代类级契约。 |
| H10 | `~Buffer() = default` | 离开作用域时销毁成员；unique_ptr 对仍拥有的数组执行匹配的数组释放。 | 不需要手写 `delete[]`；若又释放 `data()` 借出的地址，就会和 unique_ptr 重复释放。 |
| H11 | `float* data()` 与 `const float* data() const` | 普通对象可经返回指针写元素；只读 Buffer 只能得到只读元素地址。 | 只有只读版本就不能经普通 Buffer 写入；只有可写版本则只读对象无法安全调用。返回地址始终只是借用，不转移所有权。 |
| H12 | `size() const noexcept` 按值返回 `std::size_t` | 返回一个长度副本；读取不改 Buffer，且不把成员变量的引用交给调用者。 | 返回引用会让调用方获得与对象生命周期绑定的别名；对这个小整数没有必要承担该耦合。 |
| H13 | `std::unique_ptr<float[]> data_` 与 `std::size_t size_` 一起作为私有成员 | unique_ptr 是唯一资源所有者，长度与所拥有数组一起封装；外部只能经受控接口访问。 | 若以裸指针拥有数组，类必须自行维护释放，并正确处理拷贝、移动、异常和析构；少一处都可能泄漏或重复释放。 |
| H14 | `static_assert` 检查 traits 与只读返回类型 | 在构建时拒绝违反本类类型契约的实现，例如可拷贝、不能无异常移动或 const 访问器返回错类型。 | 只靠运行测试无法在每种调用路径都及时暴露接口性质错误；不过编译期断言也不能证明运行期值和资源转移正确。 |

一个容易误解的点：`std::unique_ptr<float[]>` 支持移动，因此若显式把 `Buffer(Buffer&&)` 默认化，指针成员本身可以正确移动；本类手写移动构造的关键原因，是还要把普通整数成员 `size_` 清成 0，维持来源对象的不变量。不要把“unique_ptr 不可拷贝”推成“unique_ptr 不可移动”。

`noexcept` 与 vector 的关系要按本例边界理解：本程序同时禁止拷贝、提供无异常移动，并用 `static_assert` 确认这一性质；vector 扩容运行场景验证当前类型可以迁移。不要把它概括为“任何容器只要没有 noexcept 就不能编译”——标准操作及类型是否可拷贝会影响容器的异常保证和可用策略。

## 定义到验证：每条契约由哪里检查

| `buffer.hpp` 中的定义 | `buffer.cpp` 对应位置 | 检查类型 | 证据能说明什么 |
|---|---|---|---|
| H3/H4 禁止拷贝 | 文件顶部 `static_assert`；验证报告中的两个 `-fsyntax-only` 负例 | 编译期 | 类型 traits 为 false，实际拷贝构造/赋值代码被编译器拒绝。 |
| H5/H6/H7 移动来源并保持成员一致 | `ownership_chain()`、`empty_transitions_and_reuse()` | 运行期 | 在这些输入场景中，数组地址和值保留，来源变成空对象。 |
| H8 目标旧资源清理 | `ownership_chain()` 与空/非空转移；退出时 ASan 泄漏检查 | 运行期 | 覆盖了本程序执行的覆盖赋值路径，旧数组没有报告泄漏；不等于证明所有异常/输入组合。 |
| H9 自移动约定 | `self_move()` | 运行期 | `same` 是 `buffer` 的引用别名，实际走到地址相同的自移动场景，值、长度和地址不变。 |
| H10 自动析构 | `exception_cleanup()`；各函数离开作用域时的 Sanitizer 检查 | 运行期 | 异常展开和正常退出中的已执行对象均被清理；未故意解引用悬空指针。 |
| H11/H12 访问接口 | `ownership_chain()` 的地址、元素访问和长度检查；只读返回类型 `static_assert` | 编译期 + 运行期 | 返回类型受编译期约束；运行期检查所选路径上的借用地址与长度。 |
| H13 所有权成员支持 vector 迁移 | `vector_relocation()` | 运行期 | 扩容后数组地址和值仍保留；和无异常移动 traits 一起证明这条小场景。 |
| `check` 与 `main` 的失败路径 | 故意失败时 `check` 抛异常，`main` 捕获并返回失败码 | 运行期 | CTest 可通过进程退出码识别失败；本轮成功输出五条 PASS。 |

这些证据分工不同：编译期类型约束不验证数组运行期内容；运行期用例只验证实际执行路径；ASan/UBSan 是动态检查工具，不是对所有输入或并发场景的形式证明。

## 验证函数职责速查

| 函数 | 输入与动作 | 关联契约和可见结果 |
|---|---|---|
| `check(condition, message)` | 接收真假条件与错误文字；假时抛 `runtime_error`。 | 失败立即停止当前用例，让 `main` 统一报告；不是一个测试场景。 |
| `check_empty(buffer)` | 借用一个只读 Buffer，检查空指针与长度 0 同时成立。 | 验证空状态不变量；不修改也不接管对象。 |
| `ownership_chain()` | 创建 a、写入四个值、移动到 b，再赋给已有 c。 | 移动地址、长度、数值、来源为空、目标旧数组释放、借用指针仍指向存活数组。 |
| `self_move()` | 通过引用别名将 buffer 移动赋值给自身。 | 检查 H9 的本类自移动约定。 |
| `empty_transitions_and_reuse()` | 运行空→空、空目标接非空、非空目标接空，再给移动后对象重新分配。 | 检查来源长度同步归零、旧目标资源释放和对象可复用。 |
| `vector_relocation()` | 保存元素数组地址，要求 vector 容量严格增长并检查元素迁移结果。 | 验证本类可由容器无异常移动，数组本身不搬家。 |
| `exception_cleanup()` | Buffer 局部存活时主动抛异常并在外层接住。 | 验证栈展开调用析构；泄漏检查观察该实际路径。 |
| `main()` | 依序调用各场景，成功打印 PASS；异常打印 FAIL 并返回非零。 | 是整个程序和 CTest 的执行入口。 |

## 第一部分：读懂 Buffer 类（buffer.hpp）

### 文件边界和成员不变量

头部注释说明本文件只定义 CPU 数组所有者，`buffer.cpp` 才启动程序；`#pragma once` 防止同一编译单元重复处理头文件。三个成员/接口关系要始终保持：

| 情况 | `data_` | `size_` | 谁负责释放 |
|---|---|---:|---|
| 非空 Buffer | 拥有数组的 `unique_ptr` | 元素数 | 这个 Buffer 的 `unique_ptr` |
| 空 Buffer | `nullptr` | 0 | 无资源需释放 |
| 移动后来源 | 移动走后为空 | 清成 0 | 新目标对象 |

`data_` 与 `size_` 都是 `private`，防止调用方只改指针或只改长度，破坏不变量。调用方通过 `data()` 和 `size()` 访问信息。

### 构造函数

```cpp
explicit Buffer(std::size_t size)
    : data_(size == 0 ? nullptr : std::make_unique<float[]>(size)),
      size_(size) {}
```

从左往右读：构造函数接收元素个数 `size`；冒号引出成员初始化列表；先根据长度初始化 `data_`，再初始化 `size_`。虽然这里的顺序与声明一致，C++ 实际总按成员在类里声明的顺序初始化。

`条件 ? 值A : 值B` 是条件表达式，只求值被选中的一个分支。长度为 0 时直接用空指针；否则分配数组。最后 `{}` 是空函数体，成员已在进入函数体之前初始化完成。分配抛出异常时构造失败，不会产生一个可使用的半成品 Buffer。

### 禁止拷贝

```cpp
Buffer(const Buffer&) = delete;
Buffer& operator=(const Buffer&) = delete;
```

第一行禁止创建新 Buffer 时从另一个 Buffer 拷贝；第二行禁止把一个已有 Buffer 的内容拷贝到另一个。`= delete` 是编译器识别的函数声明语法，不是释放内存的 `delete` 运算符。既然唯一指针不允许复制，本类也不自行实现深拷贝。

### 移动构造：初始化一个新目标

```cpp
Buffer(Buffer&& other) noexcept
    : data_(std::move(other.data_)),
      size_(std::exchange(other.size_, 0)) {}
```

调用示例是 `Buffer b(std::move(a));`：新建的 `b` 是目标，`other` 引用原对象 `a`。初始化 `data_` 时把唯一指针成员移动给 b，数组地址不变、元素不复制，a 的 unique_ptr 变为空。初始化长度时 `exchange` 把 a 的旧长度交给 b，并把 a 清成 0。

若只移动 unique_ptr 而普通复制 `size_t`，来源指针会空但长度仍旧非零。因此这里同时更新两个成员，确保来源的“空指针 + 长度 0”不变量。`noexcept` 依赖成员移动与长度交换不抛异常。

### 移动赋值：覆盖一个已存在的目标

```cpp
Buffer& operator=(Buffer&& other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}
```

`c = std::move(b)` 中 c 已存在，`this` 指向 c，`other` 引用 b。先检查地址是否相同，若是自移动就跳过，符合本类“自移动保留原数据”的约定。若是不同对象，unique_ptr 的移动赋值会释放 c 原有数组并接管 b 的数组；随后长度也从 b 转给 c，b 变空。

`return *this` 返回 c 本身的引用，不复制它；所以赋值表达式的结果仍指向目标 c。析构函数写作 `~Buffer() = default`，表示由编译器自动析构成员；unique_ptr 会释放当前拥有的数组，空指针不释放任何数组。

### 访问器与私有存储

两个 `data()` 是重载：可写 Buffer 返回 `float*`，只读 Buffer 返回 `const float*`。函数末尾的 `const` 表示该函数承诺不改 Buffer 成员；它和返回类型里的 `const` 限制的是两件事。`data_.get()` 借出原始地址，unique_ptr 仍是所有者。`size()` 返回长度整数的副本。

最后的 `std::unique_ptr<float[]> data_;` 管数组生命期，`std::size_t size_;` 只记元素数。类把二者放在一起，使每个对象都能同时表达“资源在哪”和“有多少元素”。

## 第二部分：按 buffer.cpp 执行顺序读完整程序

### 1. 头文件与编译期检查

`#include "buffer.hpp"` 引入本课类定义；尖括号 include 引入标准库。`using stage2::Buffer;` 建立本文件内的简写。

在任何运行前，连续的 `static_assert` 检查 Buffer 不可拷贝、可无异常移动且可无异常析构。最后一个类型检查通过 `declval<const Buffer&>().data()` 让编译器按只读对象选择 `data()` 重载，再以 `decltype` 取出返回类型，确认它精确为 `const float*`。这些检查只在编译时运行，不会分配数组。

### 2. 通用运行检查函数

`check(condition, message)` 接受一个真假条件和失败文字。条件为假时抛出 `runtime_error`，因此不会只打印了失败却继续运行。`check_empty(buffer)` 通过 `data()==nullptr && size()==0` 一起检查空对象；必须两个条件都成立。

### 3. `ownership_chain()`：从 a 经 b 移交给 c

循环先确认构造出来的 4 个元素均为零，再填入 `[1,2,3,4]`。保存 `p=a.data()` 后，p 仅观察地址。移动构造创建 b 后，程序检查 a 为空、b 仍使用 p 指向的地址且长度为 4。

然后新建 `c(2)` 并写入标记值。`c=std::move(b)` 覆盖旧目标：c 原来的 2 元素数组由 unique_ptr 释放，原 4 元素数组由 c 接管。逐元素检查证明内容仍是 `[1,2,3,4]`；经 p 修改第二个元素后，再通过 c 读取，证明地址仍指向同一块数组。函数结束按创建的逆序销毁局部对象；c 释放 4 元素数组，b/a 已空。p 此后悬空，不能再读写。

### 4. `self_move()`：同一对象作为来源和目标

`Buffer& same=buffer` 让 `same` 成为 buffer 的别名。于是 `buffer=std::move(same)` 的目标和来源地址相同；移动赋值的 `if (this != &other)` 为假，跳过成员转移。程序检查地址、长度和值都保持原样。

### 5. `empty_transitions_and_reuse()`：空与非空状态组合

`Buffer empty(0)` 构造明确的空对象。移动空对象仍得到两个空对象。接着把 `full(3)` 移入空目标，目标获得数组、来源变空；再把空来源赋给非空目标，目标的旧数组需要释放并成为空对象。末尾 `moved_empty = Buffer(2)` 通过临时 Buffer 移动赋值重新填充对象，然后检查长度和写入值。这证明移动后的空对象仍可再次使用。

临时对象在完整表达式结束时就会销毁；赋值后它已成为空对象，所以不会与接收者重复释放。

### 6. `vector_relocation()`：容器移动元素

`reserve(1)` 先要求至少一个槽位；`emplace_back(2)` 在槽位中构造一个两元素 Buffer。记录数组地址和值后，程序读取容器实际的 `capacity()` 并请求 `old_capacity+1` 个槽位。请求值严格大于当前容量，因此标准要求 vector 重新分配元素存储并迁移已有 Buffer。程序确认 vector 仍有一个元素，Buffer 移动后管理同一数组地址，数组值也保留。这样不假定某个标准库一定把 `reserve(1)` 精确分配成 1 个槽位。

这段验证的是 Buffer 对象被 vector 搬到新的对象位置时，数组资源仍安全转移。`noexcept` 移动契约使标准容器可以在重新分配时可靠地移动 Buffer。

### 7. `exception_cleanup()`：异常离开作用域

`try` 块里创建拥有 5 个 float 的局部对象，然后主动抛出标记异常。控制流离开 try 块前会先析构 `temporary`，unique_ptr 因而释放数组，再进入匹配的 `catch`。catch 通过 `error.what()` 读取异常说明并与预期文字比较；`std::string(...)` 把字符指针内容构造成字符串，`==` 比较文字内容。泄漏检查负责发现这条路径下是否仍有未释放数组。

### 8. `main()`：串起验证并报告结果

`main` 依次调用以上场景。每个场景正常返回后输出一行 PASS。若任意 `check` 抛出异常，控制跳到最外层 catch，向 `std::cerr` 写入 FAIL 和原因，返回失败码；全部通过才输出最终通过文字并返回成功码。运行成功因此代表每项检查都实际执行到了。

## 构建和运行

从仓库根目录直接编译并开启 GCC/Clang Sanitizer：

```bash
g++ -std=c++17 -Wall -Wextra -Wpedantic -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  learning/stage-02/cpp-ownership/t04/buffer.cpp -o /tmp/mlsys-t04-buffer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 /tmp/mlsys-t04-buffer
```

或使用同目录 CMake/CTest 配置：

```bash
cmake -S learning/stage-02/cpp-ownership/t04 -B /tmp/mlsys-t04-build \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build /tmp/mlsys-t04-build
ctest --test-dir /tmp/mlsys-t04-build --output-on-failure -V
```

C++17 为本程序启用所用语言特性；警告选项帮助发现可疑代码；ASan 检查实际执行路径中的越界/释放问题，UBSan 检查部分未定义行为，泄漏选项检查退出时遗留的堆资源。它们不是对所有可能输入的证明，也没有检查并发访问。

本轮恢复被截断的 `buffer.cpp` 尾部后重新构建运行，实际结果与[验证报告](../../../../reports/stage-02/2026-10-08-t04-validation.md)的五项 PASS 对应。验证只能说明这些场景；报告还记录编译器、CTest 和负例结果。

<a id="t04-oral-record"></a>

## 所有权路径自测

可以先遮住源码，自行解释以下问题，再对照类实现与前文说明：

1. 在 `a → b → c` 链路中，说明移动构造和移动赋值的目标区别、c 的旧数组何时释放、最终谁释放原数组。
2. 说明独占资源为何禁止浅拷贝，以及 `std::move` 本身是否转移资源。
3. 若移动构造直接 `= default`，说明 unique_ptr 和长度成员各自如何变化，为什么本题还需要清零来源长度。

检查这些问题时要留意几个边界：本类显式删除拷贝，但可另行实现安全深拷贝；`std::move` 不创建临时对象；本类空状态是 `Buffer(0)`；自移动是否保持原值取决于类自身实现约定。
