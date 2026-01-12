from collections import defaultdict
from math import exp


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


class Variable(Op):
    def __init__(self, name: str, value: float, derivable: bool):
        self.name = name
        self.value = value
        self.derivable = derivable

    def __repr__(self):
        if self.derivable:
            return self.name
        else:
            return str(self.value)

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


class Sub(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} - {self.right})"

    def diff(self):
        return Sub(self.left.diff(), self.right.diff())

    def eval(self, perturbed_variable=None):
        return self.left.eval(perturbed_variable) - self.right.eval(perturbed_variable)


class Exp(Op):
    def __init__(self, arg: Op):
        self.arg = arg

    def diff(self):
        return Mul(Exp(self.arg), self.arg.diff())

    def __repr__(self):
        return f"exp({self.arg})"

    def eval(self, perturbed_variable=None):
        return exp(self.arg.eval(perturbed_variable))


class Mul(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} * {self.right})"

    def eval(self, perturbed_variable=None):
        return self.left.eval(perturbed_variable) * self.right.eval(perturbed_variable)

    def diff(self):
        left_diff = self.left.diff()
        right_diff = self.right.diff()
        if is_zero(left_diff) and is_zero(right_diff):
            return Variable(name="zero", value=0.0, derivable=False)

        elif is_zero(left_diff):
            return Mul(self.left, right_diff)
        elif is_zero(right_diff):
            return Mul(left_diff, self.right)
        else:
            return Add(Mul(left_diff, self.right), Mul(self.left, right_diff))


if __name__ == "__main__":
    a = Variable("a", 3.0, False)
    b = Variable("b", 2.0, False)

    x = Variable(name="x", value=0.1, derivable=True)
    y = Variable(name="y", value=0.2, derivable=True)

    expr = a * x * x + b * y
    print(f"Expression: {expr}")
    print(f"Evaluated: {expr.eval()}")
    print(f"Derivative: {expr.diff()}")

    expr_diff = expr.diff()

    variables_to_eval_derivative_on = [x, y]
    evaluated_derivatives_by_variable = {}
    for evaluated_variable in variables_to_eval_derivative_on:
        evaluated_derivatives_by_variable[evaluated_variable] = expr_diff.eval(
            perturbed_variable=evaluated_variable
        )

    print(evaluated_derivatives_by_variable)
