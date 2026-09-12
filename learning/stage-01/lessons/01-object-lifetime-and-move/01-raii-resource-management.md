# RAII

## 前置基础

1. 构造函数建立有效对象。
2. 析构函数结束对象生命并清理资源。
3. 作用域决定局部对象何时自动析构。

## 核心过程

```text
申请资源
⇒（构造成功）对象持有资源
⇒（离开作用域）调用析构函数
⇒（配对释放）资源归还系统
```

对 CUDA 而言，资源可以是 Device buffer、Stream 或 Event。RAII 的价值是让正常返回和错误退出走同一条清理路径。真实 GPU buffer 还要确认异步 kernel 已不再使用该地址。

## 程序实例

见[阶段示例](../../examples/README.md)。示例使用 `vector` 和 `unique_ptr` 管理 Host 内存，不需要 CUDA 环境；GPU buffer 仍是后续扩展方向。

## 部署说明

```bash
cmake -S learning/stage-01/examples -B build/stage1-lifetime
cmake --build build/stage1-lifetime --target stage1_demo
./build/stage1-lifetime/stage1_demo 1,2,3,4
```

## 直观理解

RAII 像 GPU 机房的借用登记：取得显存时生成责任人，责任人离场时自动归还。中途报错不会让清理步骤被遗忘，但异步任务是否结束仍需单独确认。
