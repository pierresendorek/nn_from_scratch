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


mu = -2.8  # parameter to estimate

# generating data to fit
observations = mu + np.random.randn(10) * 0.1

# Define the variable mu_est
mu_est = Variable(name="mu_est", value=0.0, derivable=True)

# Gradient descent
step = 1e-4

for iteration in range(10**5):
    # compute loss
    partial_losses = []
    for i, x in enumerate(observations):
        xi = Variable(name=f"x_{i}", value=x, derivable=False)
        partial_losses.append(square(Sub(xi, mu_est)))

    loss = sum(partial_losses)

    delta = loss.diff().eval()

    mu_est.value = mu_est.value - step * delta

    print("mu_est:", mu_est, " delta:", delta)
