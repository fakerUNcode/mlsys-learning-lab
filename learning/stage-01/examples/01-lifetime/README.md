# 生命周期

用模拟 GPU buffer 展示 RAII、禁止拷贝和移动所有权。示例只分配 Host 内存，不要求 CUDA；接口命名用于连接后续 GPU 主线。

## 运行命令

```bash
cmake -S learning/stage-01/examples -B build/stage1
cmake --build build/stage1 --target lifetime_demo lifetime_test
./build/stage1/01-lifetime/lifetime_demo
ctest --test-dir build/stage1 -R lifetime --output-on-failure
```

## 观察重点

```text
构造 buffer
⇒（进入作用域）持有资源
⇒（移动构造）转交资源
⇒（离开作用域）析构并释放
```
