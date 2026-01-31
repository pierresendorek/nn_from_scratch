from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from math import exp

import numpy as np


class Op:
    name = "Op"

    def eval(self) -> np.ndarray | float:
        raise NotImplementedError("Subclasses should implement this method.")

    def diff(self, wrt: Variable, direction: Variable) -> "Op":
        raise NotImplementedError("Subclasses should implement this method.")

    def collect_gradient(self) -> defaultdict:
        raise NotImplementedError("Subclasses should implement this method.")

    def __add__(self, other: "Op"):
        return Add(self, other)

    def __mul__(self, other: "Op"):
        return Mul(self, other)

    def __matmul__(self, other: "Op"):
        return MatMul(self, other)

    def __sub__(self, other: "Op"):
        return Sub(self, other)


class Variable(Op):
    def __init__(self, name: str, value: np.ndarray):
        self.name = name
        self.value = value

    def __repr__(self):
        return self.name

    def eval(self):
        return self.value

    def diff(self, wrt: Variable, direction: Variable) -> Op:
        if self == wrt:
            return direction
        else:
            return Variable(name="zero", value=np.zeros_like(self.value))


class Add(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} + {self.right})"

    def diff(self, wrt: Variable, direction: Variable) -> Op:
        left_diff = self.left.diff(wrt, direction)
        right_diff = self.right.diff(wrt, direction)

        if left_diff.name == "zero" and right_diff.name == "zero":
            return Variable(name="zero", value=np.zeros_like(wrt.value))
        if left_diff.name == "zero":
            return right_diff
        if right_diff.name == "zero":
            return left_diff

        return Add(left_diff, right_diff)

    def eval(self):
        return self.left.eval() + self.right.eval()


class Sub(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} - {self.right})"

    def diff(self, wrt: Variable, direction: Variable) -> Op:
        return Sub(self.left.diff(wrt, direction), self.right.diff(wrt, direction))

    def eval(self):
        return self.left.eval() - self.right.eval()


class Linear(Op):
    def __init__(self, arg: Op, repr: str, linear_operation: Callable):
        self.operation = linear_operation
        self.arg = arg
        self.repr = repr

    def __repr__(self):
        return f"{self.repr}({self.arg})"

    def eval(self):
        return self.operation(self.arg.eval())

    def diff(self, wrt: Variable, direction: Variable) -> Op:
        arg_diff = self.arg.diff(wrt, direction)

        # if isinstance(arg_diff, ConstantZero):
        #     return ConstantZero()

        # else
        return Linear(
            repr=self.repr,
            arg=arg_diff,
            linear_operation=self.operation,
        )


class Bilinear(Op):
    def __init__(self, left: Op, right: Op, repr: Callable, operation: Callable):
        self.operation = operation
        self.repr = repr
        self.left = left
        self.right = right

    def __repr__(self):
        return self.repr(self.left, self.right)

    def eval(self):
        left_eval = self.left.eval()
        right_eval = self.right.eval()

        if isinstance(left_eval, float) or isinstance(right_eval, float):
            return left_eval * right_eval
        return self.operation(left_eval, right_eval)

    def diff(self, wrt: Variable, direction: Variable) -> Op:
        left_diff = self.left.diff(wrt, direction)
        right_diff = self.right.diff(wrt, direction)

        # if isinstance(left_diff, ConstantZero) and isinstance(right_diff, ConstantZero):
        #     return ConstantZero()

        # if isinstance(left_diff, ConstantZero):
        #     return Bilinear(
        #         repr=self.repr,
        #         left=self.left,
        #         right=right_diff,
        #         operation=self.operation,
        #     )
        # if isinstance(right_diff, ConstantZero):
        #     return Bilinear(
        #         repr=self.repr,
        #         left=left_diff,
        #         right=self.right,
        #         operation=self.operation,
        #     )

        return Add(
            Bilinear(
                repr=self.repr,
                left=left_diff,
                right=self.right,
                operation=self.operation,
            ),
            Bilinear(
                repr=self.repr,
                left=self.left,
                right=right_diff,
                operation=self.operation,
            ),
        )


class Mul(Bilinear):
    def __init__(self, left: Op, right: Op):
        super().__init__(
            repr=lambda l, r: f"({l} . {r})",
            left=left,
            right=right,
            operation=lambda l, r: l * r,
        )


class MatMul(Bilinear):
    def __init__(self, left: Op, right: Op):
        def repr(a, b):
            return f"({a} @ {b})"

        super().__init__(
            repr=repr,
            left=left,
            right=right,
            operation=lambda l, r: l @ r,
        )


class Tanh(Op):
    def __init__(self, arg: Op):
        self.arg = arg

    def eval(self, perturbed_variable=None):
        arg = self.arg.eval()
        e = np.exp(2 * arg)
        return (e - 1) / (e + 1)

    def diff(self, wrt: Variable, direction: Variable) -> Op:
        return Sech2(self.arg) * self.arg.diff(wrt, direction)

    def __repr__(self):
        return f"Tanh({self.arg})"


class Sech2(Op):
    def __init__(self, arg: Op):
        self.arg = arg

    def eval(self):
        arg = self.arg.eval()
        e_pos = np.exp(arg)
        e_neg = np.exp(-arg)
        return 4 / (e_pos + e_neg)

    def __repr__(self):
        return f"Sech2({self.arg})"

    # diff is purposedly undefined, but can be implemented if needed


def fully_connected(input: Op, weight: Op, bias: Op) -> Op:
    """_summary_

    Args:
        input (Op): tensor with shape (batch_size, nb_channels_input)
        weight (Op): tensor with shape (nb_channels_input, nb_channels_output)
        bias (Op): tensor with shape (1, nb_channels_output)

    Returns:
        Op: tensor with shape (batch_size, nb_channels_output)
    """
    return MatMul(input, weight) + bias


def square(x: Op) -> Op:
    return Mul(x, x)


# # class Div(Op):
# #     def __init__(self, numerator, denominator):
# #         self.numerator = numerator
# #         self.denominator = denominator

# #     def __repr__(self):
# #         return f"({self.numerator})/({self.denominator})"

# #     def eval(self, perturbed_variable=None):
# #         return self.numerator.eval(perturbed_variable) / self.numerator.eval(perturbed_variable)

# #     def diff(self, perturbed_variable):
# #         return super().diff()

if __name__ == "__main__":
    a = Variable(name="a", value=np.random.randn(3, 2))
    b = Variable(name="b", value=np.random.randn(2, 4))

    expr = Bilinear(
        a, b, repr=lambda l, r: f"B({l}, {r})", operation=lambda l, r: l @ r
    )

    print(
        expr.diff(
            wrt=a,
            direction=Variable(name="delta_a", value=np.ones((3, 2))),
        )
    )
