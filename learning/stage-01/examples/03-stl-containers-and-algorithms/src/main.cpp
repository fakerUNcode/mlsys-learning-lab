// 本程序用一个简化的“推理任务队列”学习 STL 的三个核心角色：
//
//   容器 std::vector                 保存一组任务；
//   迭代器 begin()/end()             描述要处理的半开区间 [begin, end)；
//   算法 sort/find_if/count_if/
//        accumulate                  在该范围上完成排序、查找、统计和归约。
//
// 这些代码全部在 CPU（Host）上运行。即使真实任务引用 GPU Tensor，使用 STL
// 算法也不会自动产生 CUDA kernel；STL 在这里负责组织和调度任务元数据。

#include <algorithm>  // std::sort、std::find_if、std::count_if
#include <iostream>   // std::cout
#include <numeric>    // std::accumulate
#include <string>     // std::string
#include <vector>     // std::vector

// InferenceTask 是一种自定义类型，用来把一次推理任务的相关字段组合在一起。
// `struct` 的成员默认是 public，所以 main() 和 Lambda 可以直接读取这些字段。
struct InferenceTask {
  // std::string 拥有算子名的字符数据，离开作用域时会自动释放内部存储。
  std::string operator_name;

  // batch_size 表示这项任务一次处理的样本数量。
  int batch_size;

  // priority 数值越大表示任务越应优先执行；这是本示例约定的业务规则。
  int priority;
};

int main() {
  // std::vector<InferenceTask> 表示元素类型为 InferenceTask 的动态连续数组。
  // 花括号是列表初始化：外层初始化 vector，内层每组值初始化一个结构体。
  // `const` 表示变量名 all_tasks 不能被重新赋值，也不能增删或修改其中的任务。
  const std::vector<InferenceTask> all_tasks{
      {"embedding", 8, 1},
      {"decoder", 2, 3},
      {"prefill", 16, 2},
      {"decoder", 4, 3},
  };

  // 这里有意复制一份容器。all_tasks 保留原始输入顺序，scheduled_tasks 用于排序。
  // vector 的复制会逐个复制其中的 InferenceTask；实际大型系统应考虑这份成本。
  auto scheduled_tasks = all_tasks;

  // std::sort 接收两个随机访问迭代器，处理 [begin, end) 半开区间：
  // begin() 指向第一个任务；end() 指向最后一个任务之后，不能被解引用。
  // 第三个参数是比较器 Lambda。`const InferenceTask&` 以只读引用借用元素，
  // 避免每次比较都复制包含 std::string 的任务对象。
  std::sort(scheduled_tasks.begin(), scheduled_tasks.end(),
            //sort第三个参数为比较器，说明排序规则
            // []：捕获列表为空，表示这个 Lambda 函数不需要借用外部作用域的任何变量。
            // (const ...)：参数列表。sort 算法在内部做比较时，会不断把两个任务分别当作 left 和 right 传进来。
            // 加 const 和 & 是为了安全和高效（只读、不拷贝）。
            [](const InferenceTask& left, const InferenceTask& right) {
              // 如果两者的优先级不相等，则比较数值大小。
              // 使用 `>` 代表降序：如果 left 的优先级数值大于 right，就返回 true。
              // 这样数值越大的任务（越紧急），就会被推到数组越靠前的位置。
              if (left.priority != right.priority) {
                return left.priority > right.priority;
              }

              // 第二重判断：副规则（处理平局，比较 batch_size）
              // 如果程序能运行到这里，说明 left 和 right 的 priority 绝对相等。
              // 此时采用 `<` 代表升序：如果 left 的批量大小更小，就返回 true。
              // 这保证了在优先级完全一样的情况下，batch_size 较小的任务会排在前面。
              return left.batch_size < right.batch_size;
            });

  std::cout << "scheduled tasks:\n";

  // 范围 for 会依次访问容器中的每个元素。这里的 `const auto&` 表示：
  //   auto  让编译器推导元素类型为 InferenceTask；
  //   &     引用原元素而不复制；
  //   const 循环体只能读取任务，不能修改排序后的容器。
  for (const auto& task : scheduled_tasks) {
    std::cout << "  op=" << task.operator_name
              << ", batch=" << task.batch_size
              << ", priority=" << task.priority << '\n';
  }

  // std::find_if 从前向后查找第一个让谓词返回 true 的元素。
  // `high_priority_task` 的类型是 vector<InferenceTask>::iterator。
  // Lambda 的 [] 是捕获列表；这里为空，因为判断逻辑不需要外部变量。
  const auto high_priority_task = std::find_if(
      scheduled_tasks.begin(), scheduled_tasks.end(),
      [](const InferenceTask& task) { return task.priority >= 3; });

  // 查找失败时 find_if 返回 end()。必须先比较再解引用，不能读取 *end()。
  if (high_priority_task != scheduled_tasks.end()) {
    // `->` 通过迭代器访问其所指 InferenceTask 的成员，类似指针成员访问。
    std::cout << "first high-priority task: "
              << high_priority_task->operator_name << '\n';
  }

  // count_if 返回范围内满足谓词的元素数量。
  // operator_name == "decoder" 使用 std::string 的内容比较，不是地址比较。
  const auto decoder_count = std::count_if(
      scheduled_tasks.begin(), scheduled_tasks.end(),
      [](const InferenceTask& task) {
        return task.operator_name == "decoder";
      });

  // accumulate 通常用于数值归约，但也允许提供自定义累加规则：
  //   前两个参数：要处理的 [begin, end)；
  //   第三个参数：初始值 0，同时让累加结果类型成为 int；
  //   第四个参数：把当前总和与当前任务合并的 Lambda。
  // 每轮计算都等价于 total = total + task.batch_size。
  const int total_batch_size = std::accumulate(
      scheduled_tasks.begin(), scheduled_tasks.end(), 0,
      [](int total, const InferenceTask& task) {
        return total + task.batch_size;
      });

  std::cout << "decoder task count: " << decoder_count << '\n';
  std::cout << "total batch size: " << total_batch_size << '\n';

  // main 返回 0，向操作系统表示程序正常结束。随后局部 vector 和其中的 string
  // 按作用域逆序自动析构，不需要手写 delete，这同样符合 RAII。
  return 0;
}
