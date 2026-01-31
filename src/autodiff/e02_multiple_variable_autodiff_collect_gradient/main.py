from __future__ import annotations

from collections import defaultdict
from math import exp


class Op:
    def eval(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def diff(self, wrt: Variable) -> Op:
        raise NotImplementedError("Subclasses should implement this method.")

    def collect_gradient(self) -> defaultdict:
        raise NotImplementedError("Subclasses should implement this method.")

    def __add__(self, other: "Op"):
        return Add(self, other)

    def __mul__(self, other: "Op"):
        return Mul(self, other)

    def __sub__(self, other: "Op"):
        return Sub(self, other)


class Variable(Op):
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"({self.name}={self.value})"

    def eval(self):
        return self.value

    def diff(self, wrt: Variable) -> Op:
        if self == wrt:
            return Variable(name="one", value=1.0)
        else:
            return Variable(name="zero", value=0.0)


class Add(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} + {self.right})"

    def diff(self, wrt: Variable) -> Op:
        return Add(self.left.diff(wrt), self.right.diff(wrt))

    def eval(self):
        return self.left.eval() + self.right.eval()


class Sub(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} - {self.right})"

    def diff(self, wrt: Variable) -> Op:
        return Sub(self.left.diff(wrt), self.right.diff(wrt))

    def eval(self):
        return self.left.eval() - self.right.eval()


class Exp(Op):
    def __init__(self, arg: Op):
        self.arg = arg

    def diff(self, wrt: Variable) -> Op:
        return Mul(Exp(self.arg), self.arg.diff(wrt))

    def __repr__(self):
        return f"exp({self.arg})"

    def eval(self):
        return exp(self.arg.eval())


class Mul(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} * {self.right})"

    def eval(self):
        return self.left.eval() * self.right.eval()

    def diff(self, wrt: Variable) -> Op:
        left_diff = self.left.diff(wrt)
        right_diff = self.right.diff(wrt)
        return Add(Mul(left_diff, self.right), Mul(self.left, right_diff))


if __name__ == "__main__":
    # Constants (considered as such)
    a = Variable("a", 3.0)
    b = Variable("b", 2.0)

    # Variables
    x = Variable(name="x", value=0.1)
    y = Variable(name="y", value=0.2)

    expr = a * x * x + b * y
    print(f"Expression: {expr}")
    print(f"Evaluated: {expr.eval()}")
    print(f"Derivative: {expr.diff(wrt=x)}")  # should be 2ax
    print(f"Derivative: {expr.diff(wrt=y)}")  # should be b
