"""T05 的可运行验证程序：比较三种方式得到的梯度。

文件职责与执行位置：从仓库根目录运行本文件后，底部入口调用 ``main``；
``main`` 依次运行标量复合函数和两输入、两输出线性层。每个例子都比较：
1. README 中逐步推导的手算结果；
2. PyTorch autograd 根据计算图反向得到的梯度；
3. ``central_difference`` 通过扰动输入、重新计算损失得到的数值近似。

相关文件：``README.md`` 解释公式、形状和推导；
``reports/stage-02/2026-10-08-t05-validation.md`` 记录一次运行的环境与误差。
本脚本只负责产生可复算的数值证据，不训练模型。输入都在 CPU 上由固定常量
构造并使用 float64，因此不依赖随机初始化或 GPU；float64 仍有舍入误差，
所以有限差分结果只应视为近似值。
"""

# torch 提供 Tensor、autograd 和数值检查接口；torch.nn.functional 是函数模块，
# 这里用 ``as F`` 给它一个短别名，后面的 F.linear 就是调用这个模块中的 linear。
import torch
import torch.nn.functional as F


# 所有实验 Tensor 使用 64 位浮点数。中心差分会相减两个很接近的损失值，
# 相比 float32，float64 通常能保留更多有效数字；它并不能消除浮点舍入误差。
DTYPE = torch.float64

# 每个坐标的扰动量 h。中心差分用 parameter+h 和 parameter-h 两次前向值，
# 再除以 2h。这个值是本例的设置，不保证适用于数值尺度不同的其他函数。
STEP = 1e-6


def scalar_loss(x: torch.Tensor) -> torch.Tensor:
    """计算标量复合函数 u=x*x+2*x+1、f=u^3，并返回标量 Tensor。

    参数 x 是由调用方创建的零维 Tensor；如果它开启了梯度跟踪，下面的
    square、乘加和 pow 运算会连入 autograd 计算图。inner 是中间值
    u=x^2+2x+1，返回值是 u^3。函数只计算前向结果，不修改 x。

    签名中的 ``x: torch.Tensor`` 和 ``-> torch.Tensor`` 是类型注解，分别提示
    参数和返回值预期为 Tensor；Python 通常不会仅凭类型注解自动拒绝其他类型。
    """
    # square() 对标量 x 求平方；乘加形成中间节点 u。每个运算都产生一个
    # 新 Tensor，并在需要时记录其与 x 的依赖，供反向传播使用。
    inner = x.square() + 2.0 * x + 1.0

    # pow(3) 表示对 inner 逐元素取三次方；本例 inner 是标量，返回也为标量。
    return inner.pow(3)


def central_difference(loss_fn, parameter: torch.Tensor, step: float) -> torch.Tensor:
    """逐坐标用中心差分近似参数梯度，并在返回前恢复参数。

    ``loss_fn`` 是无参数可调用对象；每次调用都必须从当前参数重新计算并返回
    一个单元素实数 Tensor。``parameter`` 是待求导 Tensor，形状可以是标量、
    向量或矩阵；``step`` 是正的扰动量 h。返回值 ``estimate`` 与 parameter
    同形状，每个位置保存该坐标的差分近似：

        (L(flat_parameter[k]+h) - L(flat_parameter[k]-h)) / (2h)

    每次只扰动一个坐标，其余输入和坐标保持不变。中心差分不是解析导数：
    光滑处截断误差通常为 O(h^2)，h 太小时又会受浮点相减舍入影响。

    所有权与副作用：本函数会原地改写调用者传入的 parameter，但每次成功
    计算后会把被改坐标恢复为原值。``view(-1)`` 要求 parameter 连续，且这个
    视图与原 Tensor 共用存储，因此写入视图就是写入原参数。当前固定例子
    都是连续 Tensor。``no_grad`` 让这些临时赋值和前向检查不进入 autograd 图。
    若 loss_fn 抛异常，当前实现没有 finally 清理，坐标可能停留在扰动值；
    因此这里只用于本脚本确定、无异常的前向函数，不是通用安全的差分工具。
    """
    # empty_like 只分配同形状、同 dtype、同设备的未初始化存储；下面循环必须
    # 写满每个坐标后才能读取 estimate。参数和 estimate 在本例中都是连续 Tensor。
    estimate = torch.empty_like(parameter)

    # no_grad 是上下文管理器：缩进块执行完毕后自动恢复原来的梯度跟踪状态。
    # 这很重要，因为 parameter 是 requires_grad=True 的叶子 Tensor，差分时需要
    # 临时原地赋值；这些检查计算也不需要建立第二张 autograd 图。
    with torch.no_grad():
        # numel() 返回总元素数；range(n) 依次产生 0 到 n-1，遍历展平后的每个坐标。
        for index in range(parameter.numel()):
            # view(-1) 把连续 Tensor 暂时看作一维视图；-1 表示自动推导元素数。
            # item() 将单元素 Tensor 读成 Python float，保存原参数值以供恢复。
            original = parameter.view(-1)[index].item()

            # 正向扰动：只把当前坐标改为 θ+h，其他坐标不动，再重新计算标量损失。
            parameter.view(-1)[index] = original + step
            plus = loss_fn().item()

            # 负向扰动：从同一个原值出发改为 θ-h，而非在 θ+h 基础上再减 h。
            # 因此两次采样点间距是 2h，形成对称差分并抵消一部分偶次误差。
            parameter.view(-1)[index] = original - step
            minus = loss_fn().item()

            # 恢复调用者的参数，再把本坐标的差分写进同位置的结果 Tensor。
            # 先恢复再写 estimate，确保下一轮仍从同一个基准参数点开始。
            parameter.view(-1)[index] = original
            estimate.view(-1)[index] = (plus - minus) / (2.0 * step)
    return estimate


def max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    """返回两个同形状 Tensor 的最大逐元素绝对误差。

    left 和 right 分别是要比较的两组梯度；相减、取绝对值、取最大值后，
    item() 将单元素 Tensor 转成 Python float 供格式化输出。此函数只汇总误差，
    不决定误差是否可接受；通过标准由调用处的 assert_close 设置。
    """
    return (left - right).abs().max().item()


def check_scalar() -> None:
    """检查标量函数在 x=0.5 处的手算、autograd 和差分梯度。"""
    # 这里的 x 是 check_scalar 的局部标量 Tensor；它只用于 scalar_loss，
    # 与 check_linear_layer 中同名的长度为 2 的输入向量没有共享关系。
    # torch.tensor 创建零维 CPU Tensor。requires_grad=True 请求 autograd 跟踪 x；
    # 直接创建且没有上游运算的 x 是叶子 Tensor，scalar_loss 的运算会记录 grad_fn。
    x = torch.tensor(0.5, dtype=DTYPE, requires_grad=True)
    loss = scalar_loss(x)

    # inputs 是只有一个元素的 tuple，逗号用于表示 tuple。autograd.grad 返回 tuple；
    # 左侧 (autograd_x,) 用单元素 tuple 解包语法取出唯一梯度，结果是 d(loss)/dx。
    (autograd_x,) = torch.autograd.grad(loss, (x,))

    # README 中的手算先得 u=2.25、du/dx=3、df/du=15.1875，链式相乘得到
    # df/dx=45.5625。此 Tensor 是独立的预期答案，不参与 loss 的计算图。
    manual_x = torch.tensor(45.5625, dtype=DTYPE)

    # lambda: ... 是一个无参数匿名函数；它捕获当前作用域里的 x。差分函数临时
    # 改 x 后调用 lambda，lambda 再执行 scalar_loss(x)，所以两侧损失会重新计算。
    finite_x = central_difference(lambda: scalar_loss(x), x, STEP)

    # rtol=0 关闭相对误差，只检查绝对误差。手算是解析值，要求 1e-12；有限差分
    # 是近似值，允许 1e-7。超出容差会抛异常，中止程序，不会静默报告通过。
    torch.testing.assert_close(autograd_x, manual_x, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(finite_x, autograd_x, rtol=0.0, atol=1e-7)

    # item() 把标量 Tensor 转成普通数字；格式说明 .12g 表示最多 12 位有效数字，
    # .3e 表示科学记数法并保留 3 位小数。
    print("scalar: x=0.5, loss=", f"{loss.item():.12g}")
    print("  manual dx=", f"{manual_x.item():.12g}")
    print("  autograd dx=", f"{autograd_x.item():.12g}")
    print("  finite-difference dx=", f"{finite_x.item():.12g}")
    print("  max |FD - autograd|=", f"{max_abs_error(finite_x, autograd_x):.3e}")
    # detach() 生成不连接原计算图的 Tensor 视图（与 x 共用数值存储）；
    # requires_grad_() 原地开启该视图的梯度跟踪，使其作为新的检查输入。
    # gradcheck 会逐坐标数值扰动输入并与解析反向梯度比较，eps 使用同一 h。
    scalar_gradcheck = torch.autograd.gradcheck(
        scalar_loss, (x.detach().requires_grad_(),), eps=STEP
    )
    print("  gradcheck=", scalar_gradcheck)
    assert scalar_gradcheck, "scalar gradcheck failed"


def linear_loss(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """计算两输出线性层的固定目标半平方误差，返回标量 Tensor。

    x 的形状为 (2,)，weight 的形状为 (2,2)，bias 和 target 的形状为 (2,)。
    F.linear 按 x @ weight.T + bias 计算预测；weight 的行索引对应输出，列索引
    对应输入。target 是本教学例子给定的标签，不由线性层产生。返回的单个 loss
    才是本例要对 x、weight、bias 求导的标量目标。
    """
    # 固定目标与输入使用相同 dtype，且默认创建在 CPU；与本脚本的输入设备一致。
    target = torch.tensor([1.0, 2.0], dtype=DTYPE)

    # residual 定义为“预测减目标”，形状是 (2,)。square 逐元素平方，sum 把两项
    # 合成标量。0.5 与平方求导产生的系数 2 抵消，所以 dL/dy 等于 residual。
    residual = F.linear(x, weight, bias) - target
    return 0.5 * residual.square().sum()


def check_linear_layer() -> None:
    """验证小线性层的输入、权重和偏置三组梯度。"""
    # 这里重新创建一个局部 x 向量，形状为 (2,)，不是上一个标量例子的 x=0.5。
    # 这些固定数值使每一步都能手算复核。x=(2,)；weight=(2,2)，每行对应一个
    # 输出、每列对应一个输入；bias=(2,)。F.linear 计算 x @ weight.T + bias。
    # requires_grad=True 让这三个直接创建的 Tensor 都成为可求导叶子。
    x = torch.tensor([2.0, -1.0], dtype=DTYPE, requires_grad=True)
    weight = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=DTYPE, requires_grad=True)
    bias = torch.tensor([0.5, -0.5], dtype=DTYPE, requires_grad=True)
    loss = linear_loss(x, weight, bias)

    # inputs tuple 的顺序是 x、weight、bias，因此返回 tuple 按同一顺序解包。
    # 三个梯度形状分别是 (2,)、(2,2)、(2,)，与各自输入参数形状相同。
    grad_x, grad_weight, grad_bias = torch.autograd.grad(loss, (x, weight, bias))

    # 按 README 的手算：y=[0.5,-3.5]，r=y-target=[-0.5,-5.5]，且 dL/dy=r。
    # dL/db=r；dL/dW[i,j]=r[i]*x[j]，所以外积 r[:,None]*x[None,:] 给出
    # [[-1,0.5],[-11,5.5]]；dL/dx=weight.T @ r=[5,-6.5]。下面三项是常量答案，
    # 用来独立核对 autograd，而不是由 autograd 计算出来的副本。
    manual_x = torch.tensor([5.0, -6.5], dtype=DTYPE)
    manual_weight = torch.tensor([[-1.0, 0.5], [-11.0, 5.5]], dtype=DTYPE)
    manual_bias = torch.tensor([-0.5, -5.5], dtype=DTYPE)

    # lambda 定义不接收显式参数的匿名函数。它闭包捕获 x、weight、bias 这三个
    # Tensor 对象；central_difference 原地暂时改其中一个坐标时，闭包读取到的就是
    # 当前扰动值，并重新计算同一个标量 loss。差分函数返回前会恢复这些坐标。
    loss_fn = lambda: linear_loss(x, weight, bias)
    finite_x = central_difference(loss_fn, x, STEP)
    finite_weight = central_difference(loss_fn, weight, STEP)
    finite_bias = central_difference(loss_fn, bias, STEP)
    # 每个 tuple 把同一参数组的手算、autograd、有限差分结果配在一起。
    # for 循环的 tuple 解包依次绑定三个名字，使下面同一套断言检查所有参数组。
    for manual, automatic, finite in (
        (manual_x, grad_x, finite_x),
        (manual_weight, grad_weight, finite_weight),
        (manual_bias, grad_bias, finite_bias),
    ):
        torch.testing.assert_close(automatic, manual, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(finite, automatic, rtol=0.0, atol=1e-7)

    print("linear: y=[0.5, -3.5], loss=", f"{loss.item():.12g}")

    # tolist() 把 Tensor 转为 Python 列表，便于完整展示向量和矩阵；格式化误差
    # 用科学记数法，便于比较小量级的有限差分误差。
    for name, manual, automatic, finite in (
        ("dx", manual_x, grad_x, finite_x),
        ("dW", manual_weight, grad_weight, finite_weight),
        ("db", manual_bias, grad_bias, finite_bias),
    ):
        print(f"  manual {name}=", manual.tolist())
        print(f"  autograd {name}=", automatic.tolist())
        print(f"  finite-difference {name}=", finite.tolist())
        print(f"  max |FD - autograd| ({name})=", f"{max_abs_error(finite, automatic):.3e}")

    # detach() 为每个参数建立脱离原计算图的 Tensor 视图，requires_grad_() 让它们
    # 成为新的 autograd 输入。tuple comprehension 保留 x、weight、bias 的顺序。
    # gradcheck 对每个输入坐标做数值扰动，与 autograd 梯度比较；float64 适合这类检查。
    inputs = tuple(value.detach().requires_grad_() for value in (x, weight, bias))
    linear_gradcheck = torch.autograd.gradcheck(linear_loss, inputs, eps=STEP)
    print("  gradcheck=", linear_gradcheck)
    assert linear_gradcheck, "linear-layer gradcheck failed"


def main() -> None:
    """程序总入口：打印配置，再依次运行标量例子和线性层例子。"""
    # __version__ 来自当前导入的 PyTorch；打印配置可让复现报告记录实际环境。
    print("torch version:", torch.__version__)
    print("dtype:", DTYPE, "step:", STEP, "device: cpu")
    check_scalar()
    check_linear_layer()


# Python 直接运行本文件时将 __name__ 设为 "__main__"，条件成立并开始实验；
# 若其他模块 import 本文件，条件不成立，只定义函数而不会自动执行检查。
if __name__ == "__main__":
    main()
