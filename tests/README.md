# Tests

保存跨模块回归测试和公共测试约定。靠近具体实现的局部测试放在对应模块，只有共享行为或端到端流程放在这里。

## 当前状态

根目录测试套件尚在建设中。阶段 1 C++ 示例已有独立 CTest，位于 [`learning/stage-01/examples`](../learning/stage-01/examples/)。T04 的 CPU Buffer 所有权验证也有[独立 CTest 入口](../learning/stage-02/cpp-ownership/t04/README.md)，保持在练习目录中。

## 运行方式

Python 测试：

```bash
pytest -q
```

阶段 1 C++ 测试：

```bash
cmake -S learning/stage-01/examples -B /tmp/mlsys-stage1-build
cmake --build /tmp/mlsys-stage1-build
ctest --test-dir /tmp/mlsys-stage1-build --output-on-failure
```

T04 Buffer 所有权及内存检查（GCC/Clang）：

```bash
cmake -S learning/stage-02/cpp-ownership/t04 -B /tmp/mlsys-t04-build \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build /tmp/mlsys-t04-build
ctest --test-dir /tmp/mlsys-t04-build --output-on-failure -V
```

此 CTest 入口启用 ASan、UBSan 和泄漏检查，覆盖移动、自移动、空对象、容器迁移与异常清理；编译器禁止拷贝的验证方法见[报告](../reports/stage-02/2026-10-08-t04-validation.md)。

## 分类约定

| 类型 | 要求 |
| --- | --- |
| CPU | 默认可运行，速度足够快 |
| GPU | 无兼容 GPU 时明确 skip |
| 工具链 | 缺少编译器或 Toolkit 时说明原因 |
| 数值 | 明确 dtype、绝对误差和相对误差 |
| 性能 | 不作为普通单元测试的硬性时间断言 |

## 覆盖重点

- 正常输入、空输入和最小输入；
- 非整除线程块尺寸等边界 shape；
- dtype、device、layout 与非连续 Tensor；
- 错误输入和可操作的错误信息；
- CPU reference 与 GPU candidate 的数值一致性；
- RAII 清理、异常路径和并发停止流程。

性能回归容易受硬件和系统负载影响，应由独立 benchmark 记录分布并设定环境明确的阈值。
