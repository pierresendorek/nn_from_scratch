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
        return self.left.eval() - self.right.eval()


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


class Function(Op):
    def __init__(self, name: str, arg: Op):
        self.arg = arg
        self.name = name

    def __repr__(self):
        return f"{self.name}({str(self.arg)})"

    def diff(self):
        return Mul(Function(arg=self.arg, name=f"{self.name}'"), self.arg.diff())


# class MultiFunction(Op):
#     def __init__(self, name: str, args: list[Op]):
#         self.args = args
#         self.name = name

#     def __repr__(self):
#         return f"{self.name}({str(self.args)})"

#     def diff(self):
#         for variable in args:
#             return Mul(Function(arg=self.args, name=f"(d {self.name}/d {variable.name})({str(self.args)})", self.args.diff())


if __name__ == "__main__":
    x = Variable(name="x", value="0.123", derivable=True)

    f = MultiFunction(name="f", args=[x])
    g = MultiFunction(name="g", args=[f])

    print(g)
    print(g.diff())
