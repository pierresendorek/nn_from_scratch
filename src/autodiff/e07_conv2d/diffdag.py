import numpy as np

from .main import Bilinear, Mul, Op, Variable, exp


class Func(Op):
    def __init__(self, arg: Op, name: str):
        self.arg = arg
        self.name = name

    def diff(self):
        return Mul(Func(self.arg, name=f"D{self.name}"), self.arg.diff())

    def __repr__(self):
        return f"{self.name}({self.arg})"


if __name__ == "__main__":
    x = Variable("x", value=np.random.randn(3, 3), derivable=True)

    Ax = Bilinear(
        repr=lambda l, r: f"({l} @ {r})",
        left=Variable("A", value=np.random.randn(3, 3), derivable=False),
        right=x,
        operation=lambda l, r: l @ r,
    )

    f = Func(Ax, name="f")

    B = Bilinear(
        repr=lambda l, r: f"({l} @ {r})",
        left=Variable("A", value=np.random.randn(3, 3), derivable=False),
        right=f,
        operation=lambda l, r: l @ r,
    )

    g = Func(f, name="g")

    print("Expression:", g)
    print("Derivative:", g.diff())
