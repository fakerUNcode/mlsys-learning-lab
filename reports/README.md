# reports/

## 职责

沉淀实验过程和结论。报告不是运行日志的堆积，而是让另一位读者能复现、判断证据质量并理解下一步选择的技术记录。

## 推荐布局

```text
reports/
├── README.md
├── <YYYY-MM-DD>-<topic>.md
├── figures/
└── data/                   # 大文件可改为外部 artifact，保留获取方法
```

## 报告模板

问题与假设 → 环境快照 → baseline → 实现细节 → 正确性结果 → 性能方法 → 原始数据/图表 → 分析 → 局限性 → 结论与下一步。必须写清 commit、命令、参数、随机种子、GPU、驱动、CUDA、PyTorch 和编译器版本。
