import matplotlib.pyplot as plt
import numpy as np
from main import Add, Mul, Op, Sub, Variable


def square(x: Op) -> Op:
    return Mul(x, x)


def sum(elements: list[Op]) -> Op:
    accumulator = Variable(name="zero", value=0.0)
    for element in elements:
        accumulator = Add(accumulator, element)
    return accumulator


if __name__ == "__main__":
    # Define the variables to estimate, and set their initial values to zero
    a = Variable(name="a", value=0.0)
    b = Variable(name="b", value=0.0)
    c = Variable(name="c", value=0.0)

    # define the model
    def model(x: float | np.float64, a: Variable, b: Variable, c: Variable):
        x_var = Variable(name="x", value=x)
        return sum([Mul(a, square(x_var)), Mul(b, x_var), c])

    print(model(1.0, a, b, c))

    # generating data to fit
    x_ = np.arange(-5.0, 5.0, 1)

    a_true = 3.0
    b_true = -5.0
    c_true = 10.0

    y_true = a_true * x_**2 + b_true * x_ + c_true
    y_true = y_true + np.random.randn(*x_.shape)

    learning_rate = 1e-5

    for iteration in range(10**6):
        partial_losses = []
        for xt, yt in zip(x_, y_true):
            Yt = Variable(name="yt", value=yt)

            partial_loss = square(Sub(Yt, model(float(xt), a, b, c)))
            partial_losses.append(partial_loss)

        loss = sum(partial_losses)

        print(
            "Gradient a:",
            loss.diff(wrt=a),
        )

        print(
            "Gradient b:",
            loss.diff(wrt=b),
        )

        exit(0)

        a.value -= learning_rate * loss.diff(wrt=a).eval()
        b.value -= learning_rate * loss.diff(wrt=b).eval()
        c.value -= learning_rate * loss.diff(wrt=c).eval()

        if iteration % 100 == 0:
            print(
                "iteration = ",
                iteration,
                "\tloss = ",
                loss.eval(),
                "\ta =",
                a.value,
                "\tb =",
                b.value,
                "\tc =",
                c.value,
            )
