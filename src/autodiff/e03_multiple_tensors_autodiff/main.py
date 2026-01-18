from collections import defaultdict
from math import exp
import numpy as np


class Op:
    def eval(self, perturbed_variable=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def diff(self) -> "Op":
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
        return Add(self, Mul(other, Variable("minus_one", value=-1.0, derivable=False)))


class Delta(Op):
    def __init__(self, perturbed_variable: "Variable"):
        self.perturbed_variable = perturbed_variable

    def __repr__(self):
        return f"Delta_{self.perturbed_variable.name}"

    def eval(self, perturbed_variable=None):
        if self.perturbed_variable == perturbed_variable:
            return 1.0
        return 0.0


class Linear(Op):
    def __init__(self, repr: callable, left: Op, right: Op, operation: callable):
        self.operation = operation
        self.repr = repr
        self.left = left
        self.right = right

    def __repr__(self):
        return self.repr(self.left, self.right)

    def eval(self, perturbed_variable: Delta):
        left_eval = self.left.eval(perturbed_variable)
        right_eval = self.right.eval(perturbed_variable)

        if isinstance(left_eval, float) or isinstance(right_eval, float):
            return left_eval * right_eval

        return self.operation(left_eval, right_eval)

    def diff(self):
        left_diff = self.left.diff()
        right_diff = self.right.diff()

        if is_zero(left_diff) and is_zero(right_diff):
            return Variable(name="zero", value=0.0, derivable=False)

        if is_zero(left_diff):
            return Linear(
                repr=self.repr,
                left=self.left,
                right=right_diff,
                operation=self.operation,
            )

        if is_zero(right_diff):
            Linear(
                repr=self.repr,
                left=left_diff,
                right=self.right,
                operation=self.operation,
            )

        return Add(
            Linear(
                repr=self.repr,
                left=left_diff,
                right=self.right,
                operation=self.operation,
            ),
            Linear(
                repr=self.repr,
                left=self.left,
                right=right_diff,
                operation=self.operation,
            ),
        )


class Mul(Linear):
    def __init__(self, left: Op, right: Op):
        super().__init__(
            repr=lambda l, r: f"({l} . {r})",
            left=left,
            right=right,
            operation=lambda l, r: l * r,
        )


class Matrix(Linear):
    @staticmethod
    def operation(input: Op, matrix: Op) -> Op:
        """_summary_
            input (Op): tensor with shape (batch_size, nb_channels_input,...)
            matrix (Op): matrix with shape (nb_channels_output, nb_channels_input)
        Returns:
            Op: out(b, j, ...) = sum over i of matrix(j, i) * input(b, i, ...)
        """
        return np.einsum("bi...,ji->bj...")

    def __init__(self, left: Op, right: Op):
        super().__init__(
            repr=lambda l, r: f"({l} ° {r})",
            left=left,
            right=right,
            operation=self.operation,
        )


class MatMul(Linear):
    def __init__(self, left: Op, right: Op):
        def repr(a, b):
            return f"({a} @ {b})"

        super().__init__(
            repr=repr,
            left=left,
            right=right,
            operation=lambda l, r: l @ r,
        )


class Conv2D(Linear):
    def __init__(self, input: Op, kernel: Op, stride: int = 1):
        """
        2d convolution operation with given stride
        Only supports valid padding and single channel for now
        """
        self.input = input
        self.kernel = kernel
        self.stride = stride

    def __repr__(self):
        return f"Conv2D({self.input}, {self.kernel}, stride={self.stride})"

    def eval(self, perturbed_variable=None):
        input_array = self.input.eval(
            perturbed_variable
        )  # shape (batch, channels, height, width)

        kernel_array = self.kernel.eval(
            perturbed_variable
        )  # shape (out_channels, in_channels, kernel_height, kernel_width)

        output_height = (input_array.shape[0] - kernel_array.shape[0]) // self.stride
        output_width = (input_array.shape[1] - kernel_array.shape[1]) // self.stride

        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                input_patch = input_array[
                    :,
                    i * self.stride : i * self.stride + kernel_array.shape[0],
                    j * self.stride : j * self.stride + kernel_array.shape[1],
                ]
                output[:, :, i, j] = np.einsum("bchw,ochw", input_patch, kernel_array)
        # TODO wip
        return output


class Variable(Op):
    def __init__(self, name: str, value: np.ndarray, derivable: bool):
        self.name = name
        self.value = value
        self.derivable = derivable

    def __repr__(self):
        return self.name

    def eval(self, perturbed_variable=None):
        return self.value

    def diff(self):
        if self.derivable:
            return Delta(perturbed_variable=self)
        else:
            return Variable("zero", value=0.0, derivable=False)


def is_zero(expr: Op):
    return isinstance(expr, Variable) and not expr.derivable and expr.value == 0


class Add(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} + {self.right})"

    def diff(self):
        return Add(self.left.diff(), self.right.diff())

    def eval(self, perturbed_variable=None):
        return self.left.eval(perturbed_variable) + self.right.eval(perturbed_variable)


class Tanh(Op):
    def __init__(self, arg: Op):
        self.arg = arg

    def eval(self, perturbed_variable=None):
        arg = self.arg.eval(perturbed_variable)
        e = np.exp(2 * arg)
        return (e - 1) / (e + 1)

    def diff(self):
        return Sech2(self.arg) * self.arg.diff()

    def __repr__(self):
        return f"Tanh({self.arg})"


class Sech2(Op):
    def __init__(self, arg: Op):
        self.arg = arg

    def eval(self, perturbed_variable=None):
        arg = self.arg.eval(perturbed_variable)
        e_pos = np.exp(arg)
        e_neg = np.exp(-arg)
        return 4 / (e_pos + e_neg)

    def __repr__(self):
        return f"Sech2({self.arg})"

    # diff is purposedly undefined


class Exp(Op):
    def __init__(self, arg: Op):
        """
        Pointwise exponential
        """
        self.arg = arg

    def diff(self):
        return Mul(Exp(self.arg), self.arg.diff())

    def __repr__(self):
        return f"exp({self.arg})"

    def eval(self, perturbed_variable=None):
        return exp(self.arg.eval(perturbed_variable))


# class Div(Op):
#     def __init__(self, numerator, denominator):
#         self.numerator = numerator
#         self.denominator = denominator

#     def __repr__(self):
#         return f"({self.numerator})/({self.denominator})"

#     def eval(self, perturbed_variable=None):
#         return self.numerator.eval(perturbed_variable) / self.numerator.eval(perturbed_variable)

#     def diff(self, perturbed_variable):
#         return super().diff()


if __name__ == "__main__":
    x = Variable(name="x", value=np.random.randn(3, 3), derivable=True)
    y = Variable(name="y", value=np.random.randn(3, 3), derivable=True)
    z = Variable(name="z", value=np.random.randn(3, 3), derivable=True)

    expr = x
    for v in (y, z):
        expr = Tanh(expr) @ v

    print(f"Expression: {expr}")
    print(f"Evaluated: {expr.eval(perturbed_variable=x)}")
    print(f"Derivative: {expr.diff()}")

    expr_diff = expr.diff()

    variables_to_eval_derivative_on = [x, y, z]
    evaluated_derivatives_by_variable = {}
    for evaluated_variable in variables_to_eval_derivative_on:
        evaluated_derivatives_by_variable[evaluated_variable] = expr_diff.eval(
            perturbed_variable=evaluated_variable
        )

    print(evaluated_derivatives_by_variable)
