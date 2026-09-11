# 第二题记录

本目录保存第 `00` 节入门测评第二题的完成代码与运行证据。程序复用现有 `parse_numbers`，没有修改核心函数。

## 构建命令

从仓库根目录执行：

```bash
c++ \
  -std=c++17 \
  -Ilearning/stage-01/examples/include \
  learning/stage-01/exercises/00-cpp-basics/second-question.cpp \
  learning/stage-01/examples/src/runtime_demo.cpp \
  -o build/stage1-checkpoint-q2
```

本次练习直接调用编译器生成临时程序，避免改动正式示例的 CMake 目标。

## 合法输入

```bash
./build/stage1-checkpoint-q2 8,13,21
echo $?
```

实际输出：

```text
first=8
status=ok
0
```

退出码 `0` 表示文本成功解析，程序已经安全读取并打印首元素。

## 非法输入

```bash
./build/stage1-checkpoint-q2 8,x,21
echo $?
```

实际输出：

```text
status=invalid integer
2
```

退出码 `2` 表示没有得到可使用的解析结果。错误文字来自 `ParseResult` 的 `string_view` 分支。

## 数据流程

```text
argv[1]
⇒（string_view 借用）input
⇒（parse_numbers）result
⇒（get_if）错误或成功
⇒（非空检查）首元素
```

## 直观理解

`result` 像装着“货物”或“退货单”的箱子。先确认箱中是哪一种，再读取内容；拿到数组的引用只是查看箱内货物，箱子销毁后不能继续查看。
