from main import Variable, Add, Mul, Sub, Op
import numpy as np
import matplotlib.pyplot as plt


def square(x: Variable) -> Variable:
    return Mul(x, x)


def sum(elements: list[Op]) -> Op:
    accumulator = Variable(name="zero", value=0.0, derivable=False)
    for element in elements:
        accumulator = Add(accumulator, element)
    return accumulator


if __name__ == "__main__":
    # Define the variable x
    a = Variable(name="a", value=0.0, derivable=True)
    b = Variable(name="b", value=0.0, derivable=True)
    c = Variable(name="c", value=0.0, derivable=True)

    # define the model
    def model(x: float, a: Variable, b: Variable, c: Variable):
        x_var = Variable(name="x", value=x, derivable=False)
        return sum([Mul(a, square(x_var)), Mul(b, x_var), c])

    print(model(1.0, a, b, c))

    # generating data to fit
    x_true = np.arange(-5.0, 5.0, 1)
    y_true = 3 * x_true**2 - 5 * x_true + 10 + np.random.randn(*x_true.shape)

    # plt.plot(y_true)
    # plt.show()

    expr = Variable(name="zero", value=0.0, derivable=False)
    partial_losses = []
    for xt, yt in zip(x_true, y_true):
        Yt = Variable(name="yt", value=yt, derivable=False)

        partial_loss = square(Sub(Yt, model(xt, a, b, c)))
        partial_losses.append(partial_loss)

    loss = sum(partial_losses)

    print("======= LOSS ==========")
    print(loss)
    print(loss.eval())
    print("======= DERIVATIVE OF LOSS ==========")
    print(loss.diff())

    # expr = square()

    # # Evaluate the expression
    # value = expr.eval()
    # print(f"Value of the expression x^2 at x={x.value}: {value}")

    # # Compute the derivative of the expression
    # derivative_expr = expr.diff()
    # derivative_value = derivative_expr.eval()
    # print(f"Derivative of the expression x^2 at x={x.value}: {derivative_value}")
