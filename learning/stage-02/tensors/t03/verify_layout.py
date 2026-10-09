"""Run T03 layout experiments after recording predictions in t03/README.md."""

from itertools import product

import torch


def describe(name, tensor):
    shape = tuple(tensor.shape)
    coordinates = product(*(range(size) for size in shape))
    slots = {
        index: tensor.storage_offset()
        + sum(position * stride for position, stride in zip(index, tensor.stride()))
        for index in coordinates
    }
    print(
        f"{name}: value={tensor.tolist()}, shape={shape}, "
        f"stride={tensor.stride()}, contiguous={tensor.is_contiguous()}, "
        f"dtype={tensor.dtype}, device={tensor.device}, "
        f"storage_offset={tensor.storage_offset()}, "
        f"storage_slots={slots}, "
        f"storage_ptr={tensor.untyped_storage().data_ptr()}"
    )


def main():
    print("=== A. Shape, stride, contiguous ===")
    x = torch.arange(12).reshape(3, 4)
    t = x.transpose(0, 1)
    describe("x", x)
    describe("t = x.transpose(0, 1)", t)

    print("\n=== B. view after transpose ===")
    try:
        describe("t.view(12)", t.view(12))
    except RuntimeError as exc:
        print("t.view(12): expected RuntimeError:", str(exc).splitlines()[0])

    print("\n=== C. reshape may or may not copy ===")
    a = x.reshape(2, 6)
    describe("a = x.reshape(2, 6)", a)
    print("a shares x storage:", a.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())
    before_x = x.clone()
    a[0, 0] = -100
    print("after a[0,0] = -100, x changed:", not torch.equal(x, before_x))
    a[0, 0] = before_x[0, 0]
    print("restored x after alias check:", torch.equal(x, before_x))

    b = t.reshape(12)
    describe("b = t.reshape(12)", b)
    print("b shares t storage:", b.untyped_storage().data_ptr() == t.untyped_storage().data_ptr())
    before_t = t.clone()
    b[0] = -200
    print("after b[0] = -200, t changed:", not torch.equal(t, before_t))

    print("\n=== D. contiguous and clone ===")
    c = t.contiguous()
    describe("c = t.contiguous()", c)
    print("c shares t storage:", c.untyped_storage().data_ptr() == t.untyped_storage().data_ptr())
    cloned = x.clone()
    describe("cloned = x.clone()", cloned)
    print("clone shares x storage:", cloned.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())

    print("\n=== E1. dtype conversion on CPU ===")
    f32 = torch.tensor([1.25, 2.5], dtype=torch.float32)
    f64 = f32.to(torch.float64)
    describe("f32", f32)
    describe("f64 = f32.to(torch.float64)", f64)
    print("dtype conversion shares storage:", f32.untyped_storage().data_ptr() == f64.untyped_storage().data_ptr())
    print("dtype conversion values close:", torch.allclose(f32, f64.to(torch.float32)))

    print("\n=== E2. CPU/GPU transfer ===")
    if torch.cuda.is_available():
        cpu = torch.arange(6, dtype=torch.float32)
        gpu = cpu.to("cuda")
        torch.cuda.synchronize()
        back = gpu.to("cpu")
        describe("cpu", cpu)
        describe("gpu", gpu)
        describe("back on cpu", back)
        print("round-trip values equal:", torch.equal(cpu, back))
        print("CPU and GPU storage pointers are different domains; do not compare as a sharing test.")
    else:
        print("CUDA unavailable: GPU transfer portion blocked; CPU/dtype checks above remain valid.")

    print("\n=== F. Create, transpose, contiguous, view ===")
    base = torch.arange(6).reshape(2, 3)
    transposed = base.transpose(0, 1)
    contiguous = transposed.contiguous()
    flattened = contiguous.view(6)
    describe("base", base)
    describe("transposed", transposed)
    describe("contiguous", contiguous)
    describe("flattened = contiguous.view(6)", flattened)
    print("transpose shares base storage:", transposed.untyped_storage().data_ptr() == base.untyped_storage().data_ptr())
    print("contiguous shares transpose storage:", contiguous.untyped_storage().data_ptr() == transposed.untyped_storage().data_ptr())
    print("view shares contiguous storage:", flattened.untyped_storage().data_ptr() == contiguous.untyped_storage().data_ptr())

    print("\n=== F. A view write changes its source ===")
    alias_source = torch.arange(6).reshape(2, 3)
    alias_view = alias_source.view(6)
    alias_view[0] = -1
    print("source after view write:", alias_source.tolist())
    print("view and source share storage:", alias_view.untyped_storage().data_ptr() == alias_source.untyped_storage().data_ptr())


if __name__ == "__main__":
    main()
