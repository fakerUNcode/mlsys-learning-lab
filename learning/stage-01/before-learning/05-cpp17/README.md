# C++17

## 前置基础

1. 有些接口可能没有结果。
2. 一个值可能属于若干候选类型之一。
3. 非拥有视图不能延长底层数据寿命。

## 常用类型

| 类型 | Runtime 含义 | 风险 |
| --- | --- | --- |
| `optional<T>` | 设备配置或结果可能缺失 | 读取前检查 |
| `variant<A,B>` | CPU/GPU 后端或成功/失败 | 处理所有候选 |
| `string_view` | 无复制查看算子名 | 原字符必须仍存活 |
| 结构化绑定 | 拆开 shape、stride 或结果 | 注意值与引用 |

## 程序实例

[现代类型示例](../../examples/03-modern-types/README.md)将输入解析为成功 vector 或错误 view，并把空集求和表达为 `nullopt`。

## 部署说明

```bash
cmake --build build/stage1 --target modern_types_demo
./build/stage1/03-modern-types/modern_types_demo 1,2,3,4
```

## 直观理解

`optional` 像可能空着的显卡槽位，`variant` 像只能装一种货物的多规格箱，`string_view` 是监控画面：能看数据，但不拥有数据。
