# tests/

## 职责

承载跨模块的回归测试和测试约定；模块内部的局部测试可放在对应目录。测试分为纯 CPU、可选 GPU 和需要编译工具链的类别。

## 推荐布局

```text
tests/
├── README.md
├── test_imports.py
├── test_numerics.py
└── conftest.py
```

GPU 测试应支持无 GPU 环境下自动 skip，并对 dtype、shape、边界值和误差阈值做显式断言。运行入口是 `pytest -q`。
