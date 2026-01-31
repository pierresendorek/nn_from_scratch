from __future__ import annotations


class Op:
    def eval(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def diff(self, wrt: Variable) -> Op:
        raise NotImplementedError("Subclasses should implement this method.")


class Constant(Op):
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"{self.value}"

    def eval(self):
        return self.value

    def diff(self) -> Constant:
        return Constant(name="zero", value=0.0)


class Variable(Op):
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Variable(name={self.name}, value={self.value})"

    def eval(self):
        return self.value

    def diff(self) -> Variable:
        return Variable(name="one", value=1.0)


class Add(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} + {self.right})"

    def diff(self):
        return Add(self.left.diff(), self.right.diff())

    def eval(self):
        return self.left.eval() + self.right.eval()


class Sub(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} - {self.right})"

    def diff(self):
        return Sub(self.left.diff(), self.right.diff())

    def eval(self):
        return self.left.eval() - self.right.eval()


class Mul(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} * {self.right})"

    def eval(self):
        return self.left.eval() * self.right.eval()

    def diff(self):
        return Add(Mul(self.left.diff(), self.right), Mul(self.left, self.right.diff()))


# Example usage:
if __name__ == "__main__":
    x = Variable(name="x", value=5.0)
    C = Constant(name="C", value=2.0)

    expr = Mul(Add(x, C), x)  # (x + C) * x

    print(f"Expression: {expr}")
    print(f"Evaluated: {expr.eval()}")
    print(f"Derivative: {expr.diff()}")  # should be 1 * x + (x + C) * 1 = 2x + C
    print(f"Derivative evaluated: {expr.diff().eval()}")  # should be 2 * 5 + 2 = 12.0
