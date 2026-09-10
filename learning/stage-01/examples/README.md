# 程序实例

此实例先用纯 C++ 覆盖 RAII、移动、`unique_ptr`、STL、`optional`、`variant`、`string_view`、结构化绑定、CMake、CTest 和 Sanitizer。它不要求本机有 GPU；目的是先练熟将来构建 CUDA/PyTorch extension 时仍会使用的 Host 侧工具链。

输入 `1,2,3,4` 可以暂时理解为一组推理任务的规模配置：解析成功后进入执行路径，解析失败则返回结构化错误。后续会把这里的普通整数替换为 Tensor 元数据和 GPU 任务。

## 构建运行

```bash
cd learning/stage-01/examples
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/stage1_demo 1,2,3,4
```

## 检查内存

```bash
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

## 查看链接

```bash
nm -C build/stage1_demo | grep parse_numbers
ldd build/stage1_demo
```

## 预期输出

```text
count=4 sum=10
status=ok
```
