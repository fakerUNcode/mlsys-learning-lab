# 智能指针

模拟推理系统中的模型权重：请求通过 `shared_ptr` 共同持有模型，缓存只保存 `weak_ptr`，因此缓存不会阻止旧模型卸载。

## 运行命令

```bash
cmake -S learning/stage-01/examples -B build/stage1
cmake --build build/stage1 --target smart_pointer_demo smart_pointer_test
./build/stage1/02-smart-pointers/smart_pointer_demo
ctest --test-dir build/stage1 -R smart_pointer --output-on-failure
```
