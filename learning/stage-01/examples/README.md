# 示例导航

每个目录是一套独立示例，拥有自己的源码、测试和 CMake 目标。顶层 CMake 可以一次构建全部示例。

## 示例列表

| 编号 | 示例 | 核心知识 | 程序 |
| --- | --- | --- | --- |
| 01 | [生命周期](01-lifetime/README.md) | RAII、构造、析构、移动 | `lifetime_demo` |
| 02 | [智能指针](02-smart-pointers/README.md) | unique/shared/weak | `smart_pointer_demo` |
| 03 | [现代类型](03-modern-types/README.md) | STL、optional、variant、string_view | `modern_types_demo` |

## 全部构建

从仓库根目录执行：

```bash
cmake \
  -S learning/stage-01/examples \
  -B build/stage1 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON
cmake --build build/stage1
ctest --test-dir build/stage1 --output-on-failure
```

## 单独运行

```bash
./build/stage1/01-lifetime/lifetime_demo
./build/stage1/02-smart-pointers/smart_pointer_demo
./build/stage1/03-modern-types/modern_types_demo 1,2,3,4
```
