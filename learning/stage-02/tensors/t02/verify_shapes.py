"""Run the T02 shape exercises after writing predictions in t02/README.md."""

import torch


def show(name, fn):
    try:
        result = fn()
        if isinstance(result, torch.Tensor):
            print(f"{name}: shape={tuple(result.shape)}, ndim={result.ndim}, numel={result.numel()}")
        else:
            print(f"{name}: {result}")
    except RuntimeError as exc:
        first_line = str(exc).splitlines()[0]
        print(f"{name}: RuntimeError: {first_line}")


def main():
    x = torch.arange(24).reshape(2, 3, 4)

    show("1 scalar", lambda: torch.tensor(3.5))
    show("2 reshape", lambda: torch.arange(24).reshape(2, 3, 4))
    show("3 integer index x[1]", lambda: x[1])
    show("4a slice x[:, 1, :]", lambda: x[:, 1, :])
    show("4b ellipsis x[..., 2]", lambda: x[..., 2])
    show("5 broadcast (3,1)+(1,4)", lambda: torch.zeros(3, 1) + torch.zeros(1, 4))
    show("6 broadcast (5,3,1)+(1,4)", lambda: torch.zeros(5, 3, 1) + torch.zeros(1, 4))
    show("7 incompatible (2,3)+(3,2)", lambda: torch.zeros(2, 3) + torch.zeros(3, 2))
    show("8 incompatible (2,3,4)+(2,4)", lambda: torch.zeros(2, 3, 4) + torch.zeros(2, 4))
    show("9 scalar broadcast", lambda: torch.tensor(1.0) + torch.zeros(2, 1, 3))
    show("10 batched matmul", lambda: torch.zeros(2, 3, 4) @ torch.zeros(2, 4, 5))
    show("11 broadcasted batch matmul", lambda: torch.zeros(7, 2, 3, 4) @ torch.zeros(1, 4, 6))
    show("12 incompatible batch matmul", lambda: torch.zeros(2, 3, 4) @ torch.zeros(5, 4, 6))


if __name__ == "__main__":
    main()
