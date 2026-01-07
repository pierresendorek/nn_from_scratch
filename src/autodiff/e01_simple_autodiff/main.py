class Op:
    def eval(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def diff(self):
        raise NotImplementedError("Subclasses should implement this method.")


class Variable(Op):
    def __init__(self, name: str, value: float, derivable: bool):
        self.name = name
        self.value = value
        self.derivable = derivable

    def __repr__(self):
        if self.derivable:
            return f"Variable(name={self.name}, value={self.value})"
        else:
            return str(self.value)

    def eval(self):
        return self.value

    def diff(self):
        if self.derivable:
            return Variable(name="constant", value=1.0, derivable=False)
        else:
            return Variable(name="", value=0.0, derivable=False)


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
    x = Variable(name="x", value=5.0, derivable=True)
    C = Variable(name="C", value=2.0, derivable=False)

    expr = Mul(Add(x, C), x)  # (x + C) * x
    print(f"Expression: {expr}")
    print(f"Evaluated: {expr.eval()}")
    print(f"Derivative: {expr.diff()}")  # should be 1 * x + (x + C) * 1 = 2x + C
    print(f"Derivative evaluated: {expr.diff().eval()}")  # should be 2 * 5 + 2 = 12.0
