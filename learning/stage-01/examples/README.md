# Stage 1 按主题组织的示例程序

每个子目录是一份独立主题工程，包含自己的源码和 CMake 配置；需要自动化验证的主题还包含测试。仓库根目录的 `CMakeLists.txt` 统一配置与构建，CTest 汇总已有测试。

## 目录地图

| 目录 | 学习主题 | 主要目标 |
| --- | --- | --- |
| [`00-runtime-foundations`](00-runtime-foundations/) | C++ 运行时基础与解析错误路径 | `stage1_demo`、`stage1_test` |
| [`02-smart-pointer-ownership`](02-smart-pointer-ownership/) | `unique_ptr`、`shared_ptr`、`weak_ptr` 的所有权 | `stage1_smart_pointer_demo`、`stage1_smart_pointer_test` |
| [`03-stl-containers-and-algorithms`](03-stl-containers-and-algorithms/) | `vector`、迭代器和常用算法 | `stage1_stl_demo` |

源码目录使用明确职责名：`include/` 放接口，`src/` 放实现，`tests/` 放自动化测试。新的专题示例应使用 `序号-主题` 目录，并自带 README、CMakeLists 和源码；需要时再添加测试。

## 构建全部示例

从仓库根目录运行：

```bash
cmake -S learning/stage-01/examples -B build/stage1-examples -DCMAKE_BUILD_TYPE=Debug
cmake --build build/stage1-examples
ctest --test-dir build/stage1-examples --output-on-failure
```

运行智能指针示例：

```bash
./build/stage1-examples/02-smart-pointer-ownership/stage1_smart_pointer_demo
```

运行 STL 示例：

```bash
./build/stage1-examples/03-stl-containers-and-algorithms/stage1_stl_demo
```

运行时基础示例的历史构建说明见 [`00-runtime-foundations/README.md`](00-runtime-foundations/README.md)。
