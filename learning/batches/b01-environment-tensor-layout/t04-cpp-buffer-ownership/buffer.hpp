// 文件职责：定义只拥有一块 CPU float 数组的 Buffer，供 buffer.cpp 验证所有权。
// 执行流程：buffer.cpp 的 main 调用各验证函数；函数创建/移动 Buffer；离开作用域时析构 Buffer，由其 unique_ptr 成员释放数组。
// CMakeLists.txt 负责构建与内存检查。
// 本文件只定义资源管理接口，不负责输入数据或启动程序。
// 生命周期约定：非空对象独占数组；空对象为 nullptr + 长度 0；移动来源成为空对象。
// 所有操作都是普通 CPU 操作，不提供并发同步。
// 调用方必须避免在移动/析构期间通过其他线程或裸指针访问数组，且访问下标必须小于 size()。
#pragma once  // 同一编译单元多次包含本头文件时，只处理一次定义。

#include <cstddef>
#include <memory>
#include <utility>

// namespace 将本课的名字放入 stage2 作用域，避免与其他课程的 Buffer 同名冲突。
// namespace stage2 { ... } 把大括号内声明的 Buffer 命名为 stage2::Buffer；:: 表示到左侧命名空间或类型中查找右侧名字，不是访问某个对象的成员。
namespace stage2 {

// class 把数组所有者和元素数封装在一起。
// public 是调用方可以使用的接口，private 成员只能由本类操作，避免外部代码单独修改指针或长度而破坏一致性。
// class Buffer { ... }; 定义一种名为 Buffer 的类型；大括号内放成员声明和函数定义，类定义末尾的分号不能省略。
// public: 与 private: 中的冒号用于标记后续成员的访问权限；权限持续生效到下一个权限标记，本身不会执行任何操作。
class Buffer {
 public:
  // 构造函数：由 Buffer a(n) 调用，输入 size 是元素数，不是字节数，无返回值。
  // explicit 禁止把整数暗中转换为 Buffer，例如 Buffer a = 4；须明确写 a(4)。
  // 冒号后的成员初始化列表直接构造成员，初始化次序遵循文件末尾的成员声明次序。
  // ?: 是条件表达式：零长度使用空指针；否则 make_unique<float[]>(size) 分配 size 个 float 并进行值初始化，所以初值全是 0.0F。
  // float[] 指明数组版本，使最终释放使用 delete[] 而不是单对象 delete。
  // std:: 是标准库名字的限定前缀。
  // 分配失败会抛异常，Buffer 构造不成功；调用方不能把失败对象当作有效数组使用。
  // 逐项读 explicit Buffer(std::size_t size)：explicit 是构造限制，Buffer 与类同名所以它是构造函数，括号声明一个参数，std::size_t 是参数类型，size 是参数名。
  // 构造函数不写返回类型，不等于普通函数的 void 返回类型；Buffer a(4) 的意思是声明变量 a 并把 4 传给这个构造函数。
  // 本类没有无参构造函数，所以 Buffer a; 不成立；要构造空对象须写 Buffer a(0)，explicit 仍允许这种直接初始化。
  // 参数括号后面的单个 : 引出成员初始化列表，不是 std:: 中的 ::，也不是函数体内的赋值语句。
  // data_(表达式), size_(size) 中，左边 data_、size_ 是成员名，括号中是各自的初始值；size_(size) 中两个名字分别指成员和传入参数。
  // size == 0 ? nullptr : std::make_unique<float[]>(size) 依次读为“检查长度是否等于零；是则取空指针；否则调用分配函数”，两个分支只执行被选中的一个。
  // == 比较是否相等；单个 = 在 c = std::move(b) 这种语句中表示赋值，在 float* p = a.data() 这种声明中引出初始化值；nullptr 是表示“不指向对象”的空指针值，不是一块可访问的数组。
  // make_unique 后的 <float[]> 传入类型“float 数组”，后面的 (size) 传入运行时元素数；例如 size 为 4 时得到拥有 4 个 float 的 unique_ptr 对象。
  // 初始化列表之后的 {} 是空函数体，说明没有额外操作；成员在进入这个空函数体之前就已经构造完成。
  explicit Buffer(std::size_t size)
      : data_(size == 0 ? nullptr : std::make_unique<float[]>(size)),
        size_(size) {}

  // const Buffer& 表示只读的现有对象引用，不复制该参数对象；这里对应拷贝构造。
  // = delete 表示显式禁止这个操作，调用它会编译失败。
  // 独占资源不能靠浅拷贝地址生成第二个所有者，否则两次析构可能释放同一数组。
  // 本类也不隐式做深拷贝。
  // Buffer(const Buffer&) 中，const 修饰被引用的 Buffer，& 与类型一起表示引用；这里没有参数名，因为被删除的函数没有实现需要使用参数。
  // 把 Buffer b(a) 中的 a 传给 const Buffer& 参数本身不会复制对象；若该构造可用，是构造函数负责建立 b，本类用 = delete 禁止这次构造。
  // = delete 是函数声明上的特殊语法，不是调用 delete 释放内存；它表示“存在这个接口，但任何尝试调用它的代码都应编译失败”。
  Buffer(const Buffer&) = delete;
  // operator= 是赋值运算符重载；这个版本用于已存在对象之间的拷贝，亦禁止。
  // Buffer& 返回类型表示返回目标对象本身的引用；参数无名字是因为没有函数体。
  // 逐项读 Buffer& operator=(const Buffer&)：开头 Buffer& 是返回类型，operator= 是赋值函数的名字，括号里的 const Buffer& 是来源参数类型。
  // 若 c 已存在，c = b 会查找 c 的 operator=；若写 Buffer c = b，则 c 正在被创建，需要拷贝构造而不是拷贝赋值，本类两种形式都禁止。
  Buffer& operator=(const Buffer&) = delete;

  // 移动构造：由 Buffer b(std::move(a)) 调用，创建新的目标对象，无返回值。
  // Buffer&& 是右值引用，使该重载可以接收被 std::move 标记为可移动的对象。
  // std::move 本身只转换表达式的值类别；下面 unique_ptr 的移动构造才转移所有权，不复制数组元素，并把来源 unique_ptr 置空。
  // other 是有名字的参数表达式，因此对其成员仍需使用 std::move，不能直接尝试拷贝 unique_ptr。
  // exchange(other.size_, 0) 返回来源旧长度，同时将来源长度设为 0。
  // 若整个移动构造写成 = default，标量长度只会被复制，来源会留下非零长度，与其已为空的指针不一致。
  // 这里主动维护本题的空对象约定。
  // noexcept 承诺不抛异常；默认删除器的 unique_ptr 移动与标量操作满足该条件。
  // Buffer&& other 中的 && 是右值引用类型的一部分，不是两个布尔条件之间的逻辑“并且”；other 是对来源对象的引用，没有凭空生成另一个来源对象。
  // a 是有名字、可以继续定位到的对象表达式；std::move(a) 把该表达式标为可供移动重载使用，但 a 的生命周期继续存在。
  // “重载”是同一函数名具有不同参数形式；Buffer(a) 会匹配 const Buffer& 版本，Buffer(std::move(a)) 可匹配 Buffer&& 版本，本类前者被禁止而后者有实现。
  // other.data_ 中的 . 表示访问 other 这个对象的 data_ 成员；同一个类的函数可以访问另一个同类对象的 private 成员。
  // std::exchange(other.size_, 0) 是一次带两个实参的函数调用；假设来源长度为 4，它返回 4 供目标 size_ 初始化，同时把来源 size_ 写成 0。
  // 函数参数括号后的 noexcept 是异常规格；若仍有异常逃出该函数，程序会终止，而不是交给调用方的普通 catch 继续处理，所以必须确认函数体内操作确实符合承诺。
  Buffer(Buffer&& other) noexcept
      : data_(std::move(other.data_)), size_(std::exchange(other.size_, 0)) {}

  // 移动赋值：由 c = std::move(b) 调用；目标 c 已存在，可能已经拥有旧数组。
  // 输入 other 是来源，返回 *this 的引用以支持连续赋值。
  // 来源变空，目标接管数组。
  // 与移动构造不同，这里需要处理目标的旧资源；成员移动赋值会完成配对释放。
  // 逐项读 Buffer& operator=(Buffer&& other) noexcept：返回当前 Buffer 的引用，重载赋值运算符，接收可移动的来源引用，并承诺调用不会抛出异常。
  // c = std::move(b) 对应在 c 上调用这个成员函数并把 b 绑定到 other；this 指向 c，而 &other 得到 b 的地址。
  Buffer& operator=(Buffer&& other) noexcept {
    // this 是指向当前目标对象的指针；&other 取来源对象地址。
    // 地址相同就是自移动。
    // 本题要求自移动保留原数据，因此只在两个对象不同的时候执行转移。
    // & 在 &other 这个表达式中是“取地址”，在 Buffer& 这个类型中才是“声明引用”；!= 比较地址是否不同，if (...) 只在比较为真时执行随后的大括号代码。
    if (this != &other) {
      // 本语句结束时目标旧数组已被释放，来源数组改由目标拥有，来源指针为空。
      // 不手动 delete[]：所有释放责任交给 unique_ptr，避免重复释放。
      data_ = std::move(other.data_);
      // 再转移长度并清零来源，使移动完成后的两个对象都满足各自状态约定。
      size_ = std::exchange(other.size_, 0);
    }
    // *this 解引用目标对象指针，返回当前对象本身，不创建另一份 Buffer。
    // * 在 *this 表达式中表示“取得该地址指向的对象”，不同于 float* 类型声明中的“指针类型”；由于函数返回 Buffer&，return *this 返回的是当前对象的别名。
    // 如果返回类型改为 Buffer，这里就需要按值产生返回对象，可能尝试本类禁止的拷贝；返回引用也使 (c = std::move(b)).size() 可以直接访问赋值后的 c。
    return *this;
  }

  // 析构函数在局部对象离开作用域或异常展开时自动调用，不带参数、没有返回值。
  // = default 让编译器生成成员清理：按声明逆序销毁 size_、data_；size_ 无资源，data_ 非空则用 delete[] 释放数组，空则不释放。
  // 不要另外手动释放 data().
  // ~Buffer() 中的 ~ 与类名组合构成析构函数名，空括号表示没有参数；这里的 ~ 不是对整数进行按位取反。
  // = default 请求编译器生成该特殊成员函数，不代表“不执行清理”，也不是“把对象设成默认值”；成员的析构函数仍会被自动调用。
  ~Buffer() = default;

  // 以下访问器由验证程序调用，不转移所有权、不分配存储；noexcept 表示不会抛异常。
  // [[nodiscard]] 提醒调用方别无意忽略返回值。
  // float* 是可写元素的地址；get() 只借出地址，unique_ptr 仍是唯一所有者。
  // 调用方不能 delete[] 这个借用指针。
  // 下标访问须满足 0 <= i < size()；数组被所有者释放后，旧借用指针不能再访问。
  // 逐项读 [[nodiscard]] float* data() noexcept { return data_.get(); }：属性建议别忽略结果，float* 是返回类型，data 是函数名，() 表示无参数，{} 是函数体，return 将地址交给调用方。
  // data_ 是 unique_ptr 成员对象，data() 是本类的成员函数，两者不是同一个名字；data_.get() 用 . 在这个成员对象上调用 get()，取得其管理的原始 float*。
  // float* p = a.data() 声明一个保存 float 地址的变量 p；p[0] 访问首元素，p[1] 访问下一个 float，移动的是元素索引而不是字节地址的增量 1。
  [[nodiscard]] float* data() noexcept { return data_.get(); }
  // 末尾 const 表示可在只读 Buffer 上调用且不改变对象；返回 const float*，使调用者无法经此接口修改元素。
  // 它限制该访问路径，并不让所有其他别名也只读。
  // 本行有两个 const：开头 const float* 的 const 限制返回地址所指向的元素不可通过它修改，参数括号后的 const 限制函数对当前 Buffer 成员的修改。
  // 对普通对象 a 调用 a.data() 选择前一个可写版本；对 const Buffer& r 调用 r.data() 选择本行只读版本，所以 float* p = r.data() 不成立。
  // const float* 是“指向只读 float 的指针”，不同于 float* const 的“本身不能改指向的指针”；这里返回的是前一种类型。
  [[nodiscard]] const float* data() const noexcept { return data_.get(); }
  // 返回元素个数的副本。
  // std::size_t 是容纳对象大小的无符号整数类型，不能表示负数。
  // std::size_t size() const noexcept 的返回类型位于函数名前；return size_ 复制一个长度整数给调用方，调用方修改这个副本不会改变成员 size_。
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  // <float[]> 是模板实参：选择“float 数组”的 unique_ptr 实例。
  // 这个成员管理数组寿命；size_ 只记长度，不负责资源。
  // 成员尾部的 _ 用于区分成员和函数参数。
  // std::unique_ptr<float[]> data_; 声明一个名为 data_ 的智能指针对象；<...> 给模板指定所管理的类型，float[] 表示数组，但此处的 [] 不写运行时长度。
  // 智能指针对象本身是 Buffer 的一个成员，拥有的数组另外分配在动态存储中；移动这个成员转移管理责任，并不逐个搬运数组元素。
  // std::size_t size_; 只声明一个长度成员；它实际取什么值由前面的构造函数初始化列表决定，声明本身不会把长度自动设为零。
  std::unique_ptr<float[]> data_;
  std::size_t size_;
};

}  // namespace stage2
