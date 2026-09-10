# 生命周期

## 前置基础

1. 构造函数让一块存储进入“有效对象”状态。
2. 析构函数在对象生命结束时清理其持有资源。
3. 指针保存地址；指针消失不等于它指向的对象已被释放。

## 生命周期

RAII 的核心不是“只能管理内存”，而是把资源寿命绑在对象寿命上。在本学习主线里，最重要的资源包括 GPU buffer、CUDA Stream、CUDA Event、线程锁和动态库句柄。

以 GPU buffer 为例，`cudaMalloc` 成功后必须在所有 kernel 不再使用它时调用 `cudaFree`。如果中间的 shape 检查、kernel launch 或其他步骤失败，手写在函数末尾的 `cudaFree` 很容易被跳过；RAII 包装器能让析构函数承担清理责任。

```cpp
class File {
    // 创建一种“文件对象”
    // 以后可以这样使用：
    // File f("test.txt");

public:

    // 构造函数
    // 例如：
    // 写下File f("hello.txt");程序会自动调用： File("hello.txt")
    // 作用： 创建 File 对象的时候，顺便打开文件

    // explicit：
    // 防止 C++ 自动帮你做类型转换

    // 如果没有 explicit： File f = "test.txt";  C++ 可能偷偷理解成： File f("test.txt");
    // 加了 explicit 后： 必须明确写： File f("test.txt");

    File(const char* path)
        // const char* path：
        // 接收一个字符串，也就是文件路径

        // 初始化列表 这里是在创建对象的时候，直接给成员变量 handle_ 赋值，等价于：
        // File(const char* path)
        // {
        //     handle_ = std::fopen(path, "r");
        // }
        //
        // 只是 C++ 推荐写法

        : handle_(std::fopen(path, "r"))
        // std::fopen：
        // 调用 C 语言的文件打开函数
        // "r" 表示 read，只读方式打开
        // 成功：返回一个文件指针；失败：返回 nullptr
        // 然后把返回值保存到 handle_


    {
        // 判断文件是否打开成功
        if (!handle_)
            // 抛出异常
            throw std::runtime_error("open failed");
    }


    // 析构函数
    // 当 File 对象被销毁时，会自动调用这个函数
    // 例如：
    // {
    //     File f("a.txt");
    // }
    //
    // 离开大括号（生命域）时，f 自动销毁，然后调用 ~File()

    ~File()
    {
        // fclose：
        // 关闭文件，释放操作系统占用的文件资源
        // 这就是 RAII：
        // 创建对象 -> 获取资源(open)
        // 销毁对象 -> 释放资源(close)
        std::fclose(handle_);
    }



    // 禁止复制构造函数
    //
    // 防止：
    //
    // File a("a.txt");
    // File b = a;
    //
    // 因为复制会导致：
    //
    // a --------\
    //             ---> 同一个文件
    // b --------/
    //
    // 两个对象管理同一个文件，
    // 最后可能重复关闭文件。

    File(const File&) = delete;



    // 禁止赋值操作
    //
    // 防止：
    //
    // File a("a.txt");
    // File b("b.txt");
    //
    // b = a;
    //
    // 这样会让两个对象管理同一个文件。

    File& operator=(const File&) = delete;



private:
    // 私有成员变量

    // 外部代码不能直接访问它
    // 例如：
    // File f("a.txt");
    // f.handle_ = nullptr;  // 不允许
    //
    // 必须让 File 自己管理这个变量。


    // std::FILE*：
    //
    // FILE：
    // C 语言表示文件的一种结构
    //
    // FILE*：
    // 指向这个文件结构的指针
    //
    // 可以理解成：
    // 操作系统给这个打开文件的“编号/地址”

    std::FILE* handle_;
};
```

上面的 `File` 是最小语法示例。把 `std::fopen和std::fclose` 换成 `cudaMalloc和cudaFree`，就得到 GPU buffer 包装器的基本骨架。真实项目还必须检查 CUDA 返回码，并考虑异步工作是否已经完成。

对象离开作用域时，语言保证调用析构函数；即使中途抛出异常，已经成功构造的局部对象仍会按相反顺序析构。

```text
进入作用域
⇒（调用构造）持有资源
⇒（正常返回或异常展开）离开作用域
⇒（调用析构）释放资源
```

## 拷贝移动

==拷贝==产生两个独立对象；==移动==则把资源的控制权转交给新对象。移动后的旧对象仍必须可以析构和重新赋值，但它的业务内容通常不应再被假定。

```text
源对象 A 持有资源 R
⇒（拷贝构造）A 持有 R，B 持有 R 的副本

源对象 A 持有资源 R
⇒（移动构造）B 持有 R，A 处于有效未指定状态
```

“有效未指定”表示语言允许安全析构和赋新值，但不承诺原来的内容仍在。

GPU buffer 通常不能按普通指针做浅拷贝，否则两个对象会对同一地址调用两次 `cudaFree`。常见设计是禁止拷贝、允许移动：buffer 可以从临时执行上下文移交给 Tensor 或缓存，但始终只有一个释放者。

> **浅拷贝（shallow copy）**指的是：复制对象时，只复制“表面上的值”，不会复制它指向的底层资源。
>
> 例如：
>
> ```
> Buffer a;
> Buffer b = a;
> ```
>
> 如果 `Buffer` 内部有一个指针：
>
> ```
> void* data_;
> ```
>
> 浅拷贝会变成：
>
> ```
> a.data_ ──┐
>           ├──> GPU显存地址 R
> b.data_ ──┘
> ```
>
> 也就是说，`a` 和 `b` 表面是两个对象，但实际上管理的是**同一份资源**。
>
> 问题是：当 `a` 析构时：
>
> ```
> cudaFree(data_);
> ```
>
> 释放了 GPU 内存；之后 `b` 析构时又执行：
>
> ```
> cudaFree(data_);
> ```
>
> 就会出现**重复释放（double free）**。
>
> 所以资源管理类通常不能直接浅拷贝，而要：
>
> - **深拷贝**：复制一份新的资源；
> - **移动**：转移资源所有权，让旧对象放弃管理权。

## 所有权

| 工具 | 所有者数量 | 使用边界 |
| --- | ---: | --- |
| `unique_ptr<T>` | 1 | 默认首选；资源只有一个负责人 |
| `shared_ptr<T>` | 多个 | 多方确实需要共同延长寿命 |
| `weak_ptr<T>` | 0 | 观察共享对象；打破引用环 |

“所有权”可以理解为：**谁有责任保证对象活着，以及最终谁负责销毁它。**在推理 Runtime 中，可以这样理解：

- `unique_ptr`：**只有我需要它**。我活着它就活着，我销毁它也销毁。适合请求内部的临时对象、独占资源。
- `shared_ptr`：**大家都需要它**。只要还有一个人在用，对象就不能销毁。适合多个线程/请求共享同一个模型。
- `weak_ptr`：**我想用它，但不负责养着它**。对象可能已经被别人销毁，所以使用前要 `lock()` 检查。适合缓存、观察者。

核心区别：**unique = 独占，shared = 共同负责生命周期，weak = 只使用/观察，不管生命周期。**

设强引用计数为 \(n_s\)。对象销毁条件是：

$$
n_s
\overset{\text{减至零}}{=}
0
\Rightarrow_{\text{释放对象}}
\text{析构对象}
$$


## 符号说明

- \(n_s\)：`shared_ptr` 的强引用数量。
- 下标 \(s\)：strong，强引用。
- `weak_ptr` 不增加 \(n_s\)。

`shared_ptr` 不是“更安全的默认指针”。它有控制块和原子计数成本，而且相互持有会形成环。观察者使用 `weak_ptr::lock()` 临时取得 `shared_ptr`，并检查对象是否仍存在。

## 程序实例

```cpp
// 创建一个 GpuBuffer 对象，并让 owner 独占它
auto owner = std::make_unique<GpuBuffer>(bytes);
// ↑                   ↑          ↑
//自动推断类型          类型        构造参数
//
// 等价于：
// std::unique_ptr<GpuBuffer> owner =
//     std::make_unique<GpuBuffer>(bytes);

// ❌ unique_ptr 不能复制
// auto copy = owner;
// 意思是：不能让 copy 和 owner 同时“独占”同一个对象。


// 把 owner 的所有权“转让”给 next，这个操作则合法
auto next = std::move(owner);
//
// 转移前：owner ───→ GpuBuffer
// 转移后：owner → 空
//         next  ───→ GpuBuffer

// 创建 Model 对象
auto shared = std::make_shared<Model>(weights_path);
//
// shared 的实际类型：std::shared_ptr<Model>
// shared ───→ Model。Model 可以被多个 shared_ptr 共同拥有

// 创建一个 weak_ptr，用来“观察” shared 指向的 Model
std::weak_ptr<Model> observer = shared;
//  ↑         ↑       ↑
// 类型       Model    变量名

//检查 Model 还活着吗？如果活着，就得到一个临时 shared_ptr，放进 alive
if (auto alive = observer.lock()) {
  alive->infer(input);
}
```

这里的 `GpuBuffer` 与 `Model` 表示后续项目要实现的资源类，用来展示所有权关系，不属于本阶段可执行示例中的真实 CUDA 类型。

可运行版本见[生命周期示例](../../examples/01-lifetime/README.md)和[指针示例](../../examples/02-smart-pointers/README.md)。

## 部署说明

本节只需要支持 C++17 的编译器。进入实例目录后使用 CMake 构建；开启 Sanitizer 可检查越界、释放后使用和部分未定义行为。

## 直观理解

一块临时显存像只有一把钥匙的 GPU 仓位，交给下个执行阶段后原阶段不能再释放；**模型权重（模型推理所需的核心参数数据）**像共享仓库，最后一个请求离开才卸载；缓存只是地址簿，不应让旧模型永远占着显存。
