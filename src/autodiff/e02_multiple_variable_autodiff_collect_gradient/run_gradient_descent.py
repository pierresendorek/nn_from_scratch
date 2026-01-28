import matplotlib.pyplot as plt
import numpy as np
from main import Add, Mul, Op, Sub, Variable


def square(x: Variable) -> Variable:
    return Mul(x, x)


def sum(elements: list[Op]) -> Op:
    accumulator = Variable(name="zero", value=0.0, derivable=False)
    for element in elements:
        accumulator = Add(accumulator, element)
    return accumulator


if __name__ == "__main__":
    # Define the variables to estimate
    a = Variable(name="a", value=0.0, derivable=True)
    b = Variable(name="b", value=0.0, derivable=True)
    c = Variable(name="c", value=0.0, derivable=True)

    # define the model
    def model(x: float, a: Variable, b: Variable, c: Variable):
        x_var = Variable(name="x", value=x, derivable=False)
        return sum([Mul(a, square(x_var)), Mul(b, x_var), c])

    print(model(1.0, a, b, c))

    # generating data to fit
    x_ = np.arange(-5.0, 5.0, 1)

    a_true = 3.0
    b_true = -5.0
    c_true = 10.0

    y_true = a_true * x_**2 + b_true * x_ + c_true
    y_true = y_true + np.random.randn(*x_.shape)

    # plt.plot(y_true)
    # plt.show()

    # expr = Variable(name="zero", value=0.0, derivable=False)
    learning_rate = 1e-5

    for iteration in range(10**6):
        partial_losses = []
        for xt, yt in zip(x_, y_true):
            Yt = Variable(name="yt", value=yt, derivable=False)

            partial_loss = square(Sub(Yt, model(xt, a, b, c)))
            partial_losses.append(partial_loss)

        loss = sum(partial_losses)

        # print("======= LOSS ==========")
        # print(loss)
        # print(loss.eval())
        # print("======= DERIVATIVE OF LOSS ==========")
        # print(loss.diff())

        diff = loss.diff()

        a.value -= learning_rate * diff.eval(perturbed_variable=a)
        b.value -= learning_rate * diff.eval(perturbed_variable=b)
        c.value -= learning_rate * diff.eval(perturbed_variable=c)

        if iteration % 100 == 0:
            print(
                "loss = ",
                loss.eval(),
                "\ta =",
                a.value,
                "\tb =",
                b.value,
                "\tc =",
                c.value,
            )

    # expr = square()

    # # Evaluate the expression
    # value = expr.eval()
    # print(f"Value of the expression x^2 at x={x.value}: {value}")

    # # Compute the derivative of the expression
    # derivative_expr = expr.diff()
    # derivative_value = derivative_expr.eval()
    # print(f"Derivative of the expression x^2 at x={x.value}: {derivative_value}")
