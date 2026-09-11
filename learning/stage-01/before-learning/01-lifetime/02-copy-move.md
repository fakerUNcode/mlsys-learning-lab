# 拷贝移动

## 前置基础

1. 拷贝产生独立对象。
2. 浅拷贝只复制地址，可能让两个对象管理同一资源。
3. 移动将资源控制权交给目标对象。

## 核心过程

```text
source 持有 buffer
⇒（移动构造）target 接管 buffer
⇒（清空源状态）source 可析构
```

GPU buffer 管理类通常禁止拷贝并允许移动。否则两个包装对象可能对同一 Device 地址调用两次 `cudaFree`。移动构造常标记 `noexcept`，便于标准容器安全选择移动路径。

## 程序实例

当前示例把 `variant` 中的 `vector<int>` 移动到 `unique_ptr<vector<int>>` 管理的新对象。它展示所有权转移，但没有检查移动后源 vector 的具体内容；“源大小为零”只属于讲义推演，不能算当前测试结论。

## 部署说明

```bash
cmake --build build/stage1-lifetime --target stage1_test
ctest --test-dir build/stage1-lifetime --output-on-failure
```

## 直观理解

拷贝像另开一间同规格仓库；浅拷贝却只是复印同一仓库钥匙，容易重复退仓。移动是把唯一钥匙交给下一位负责人，原负责人不再处置资源。
