# ABI链接

## 前置基础

1. 编译产生目标文件。
2. 链接器解析跨文件符号。
3. 动态装载器在运行前查找共享库。

## 核心过程

```text
源文件
⇒（编译）目标文件
⇒（链接）可执行文件或 .so
⇒（动态装载）进程可调用符号
```

ABI 约定调用方式、类型布局、符号命名和异常边界。PyTorch extension 的编译器、C++ 标准库或依赖版本不兼容时，Python 导入可能表现为 `undefined symbol`。

## 排查命令

```bash
nm -C ./build/stage1-lifetime/stage1_demo
ldd ./build/stage1-lifetime/stage1_demo
readelf -d ./build/stage1-lifetime/stage1_demo
```

不要对来源不可信的二进制执行 `ldd`。

## 直观理解

编译加工零件，链接完成组装，ABI 规定接口尺寸，动态装载负责开机接线。接口不匹配时，即使源码看起来正确，程序也可能找不到符号。
