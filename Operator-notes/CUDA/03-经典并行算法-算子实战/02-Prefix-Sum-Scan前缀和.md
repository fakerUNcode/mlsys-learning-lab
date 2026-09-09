

并行算法中最具智慧、也是很多高级算子（如基数排序 Radix Sort、动态张量掩码过滤 Stream Compaction、可逆操作）的核心基石——**Prefix Sum / Scan（前缀和扫描）**。

如果你觉得前面的“归约（Reduction）”是把一个数组折叠成一个单一数值，那么“前缀和”就是要**为数组中的每一个元素，都算出它前面所有元素的总累加值**。



# 基础概念复习

在写代码前，我们必须先在脑海中理清前缀和的定义和分类：

1. **什么是“前缀和（Prefix Sum / Scan）”？**

   - 给定一个输入数组 $X = [x_0, x_1, x_2, \dots, x_{n-1}]$。
   - 输出一个同等长度的数组 $Y = [y_0, y_1, y_2, \dots, y_{n-1}]$。
   - 每一个位置的 $y_i$，等于当前位置以及（或仅）其前面所有元素的和。

2. **包容性（Inclusive）与独占性（Exclusive）：**

   - **Inclusive Scan（包含当前元素）：**

     $$y_i = \sum_{j=0}^{i} x_j$$

     *例如：* 输入 `[1, 2, 3, 4]`，输出 `[1, 3, 6, 10]`。

   - **Exclusive Scan（不包含当前元素，从 0 开始）：**

     $$y_0 = 0, \quad y_i = \sum_{j=0}^{i-1} x_j$$

     *例如：* 输入 `[1, 2, 3, 4]`，输出 `[0, 1, 3, 6]`。

     *(从最直观的 **Inclusive Scan** 开始深入。)*

3. **串行思维 vs 并行思维的冲突：**

   - **串行做法（极度依赖前一步）：**

     在 CPU 上，我们写 `y[i] = y[i-1] + x[i]`。每一步都死死依赖于前一步算出来的结果。这在串行上极其简单，耗时 $O(N)$。

   - **并行的巨大挑战：**

     既然下一步必须等上一步算完，成千上万个线程怎么可能“同时”去算前缀和呢？这就需要用到经典的 **Kogge-Stone 算法（逐步距离倍增算法）**。





# Kogge-Stone 并行前缀和算法公式推导

Kogge-Stone 算法的核心思想是：**多轮迭代，每一轮让跨度（Offset / Stride）翻倍。**



## 公式

在第 $k$ 轮（跨度为 $offset = 2^k$）中，对于满足条件的每个线程 $tid$：



$$Index_{target} = tid$$

$$Index_{source} = tid - offset$$

$$val_{new} = s_{data}[Index_{target}] + s_{data}[Index_{source}]$$

## 符号全翻译

- $k$：当前扫描迭代的轮次序号（从 0 开始递增）。

- $offset$：当前轮次的向左回溯跨度，取值为 $1, 2, 4, 8, \dots$。

- $Index_{target}$：当前线程在共享内存中的写入槽位，等于自己的局部工号 $tid$。

- $Index_{source}$：当前线程需要向左回溯抓取数据的来源槽位。

- $tid$：当前线程在线程块内的局部编号（对应代码中的 `threadIdx.x`）。

- $s_{data}$：存放在高速共享内存中的当前数据数组。

- $val_{new}$：两数相加后的临时过渡新值。

  

##  详细推导

为了防止数据竞争（读写冲突），每一轮的加法必须安全进行：



$$offset = 1$$

(初始化第一轮跨度为 1)

(即每个元素加上它左边紧邻的元素)



$$Index_{source} = tid - offset$$

(如果 tid >= offset，计算左侧邻居的槽位)



$$val_{new} = s_{data}[tid] + s_{data}[Index_{source}] \\ = s_{data}[tid] + s_{data}[tid - offset] \\ = s_{data}[tid] + s_{data}[tid - 1]$$

(当前位置的值加上左侧邻居的值)

($\Rightarrow$ 此时已完成相邻 2 个元素的局部累加)



$$\text{进入第二轮: } offset = 2$$

(跨度倍增为 2)



$$val_{new} = s_{data}[tid] + s_{data}[tid - 2]$$

(当前已经累加了 2 个元素的值)

(再加上向左偏移 2 个位置的元素、同样已累加了 2 个元素的值)

($\Rightarrow$ 当前位置已经汇聚了前面 4 个元素的完整总和)

![image-20260906110853703](https://fakercodes.oss-cn-hangzhou.aliyuncs.com/sl/image-20260906110853703.png)

......

经过 $\log_2(N)$ 轮之后，所有位置上的元素都会自动包含其左侧全部数字的总和！



# 前缀和程序 (`scan_demo.cu`)

下面是一个基于共享内存和 Kogge-Stone 算法的单 Block 前缀和完整实现。请在 WSL 中新建 `scan_demo.cu` 并写入以下代码：



```c++
#include <iostream>
#include <cuda_runtime.h>

#define SECTION_SIZE 256 // 处理一个块大小的前缀和 (256 个元素)

// ==========================================
// 模块一：Kogge-Stone 包含式前缀和算子 (Inclusive Scan)
// ==========================================
__global__ void koggeStoneScanKernel(float *d_out, const float *d_in, int n) {
    // 声明共享内存工作台
    __shared__ float s_data[SECTION_SIZE];

    int tid = threadIdx.x;

    // 1. 协作搬运：从全局内存把数据载入共享内存
    if (tid < n) {
        s_data[tid] = d_in[tid];
    } else {
        s_data[tid] = 0.0f;
    }
    __syncthreads(); // 等待所有线程搬运完毕

    // 2. Kogge-Stone 核心迭代：offset 依次为 1, 2, 4, 8, 16, 32, 64, 128
    for (int offset = 1; offset < SECTION_SIZE; offset *= 2) {
        float temp = 0.0f;
        
        // 只有当自己左边有足够的元素时，才去拿左边的数据
        if (tid >= offset) {
            temp = s_data[tid - offset];
        }

        // 极其重要的一步：同步！
        // 必须确保所有线程都已经把 s_data[tid - offset] 读出来了，
        // 才能更新 s_data[tid]，否则先算完的线程会覆盖掉其他线程还要读的数据。
        __syncthreads();

        // 原地更新共享内存槽位
        if (tid >= offset) {
            s_data[tid] += temp;
        }

        // 再次同步：等待所有槽位更新完毕，再开启下一轮倍增
        __syncthreads();
    }

    // 3. 将计算完毕的前缀和结果写回全局内存
    if (tid < n) {
        d_out[tid] = s_data[tid];
    }
}

// ==========================================
// 模块二：CPU 主控程序
// ==========================================
int main() {
    int n = 8; // 为了演示清晰，我们先用 8 个数字做验证
    size_t size = n * sizeof(float);

    float h_in[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_out[8] = {0.0f};

    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    std::cout << "输入原始数组: ";
    for (int i = 0; i < n; ++i) std::cout << h_in[i] << " ";
    std::cout << std::endl;

    // 启动算子 (启动 1 个 Block，包含 256 个线程)
    koggeStoneScanKernel<<<1, SECTION_SIZE>>>(d_out, d_in, n);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "前缀和扫描结果: ";
    for (int i = 0; i < n; ++i) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    std::cout << "期望标准答案:   1 3 6 10 15 21 28 36" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
```

## WSL 编译与运行流程

在 WSL 终端中依次输入以下命令：

```bash
nvcc scan_demo.cu -o scan_demo
./scan_demo
```

**预期输出：**

```
输入原始数组: 1 2 3 4 5 6 7 8 
前缀和扫描结果: 1 3 6 10 15 21 28 36 
期望标准答案:   1 3 6 10 15 21 28 36 
```

