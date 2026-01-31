import numpy as np

from .main import Add, Linear, Mul, Op, Variable, square

# generate data
nb_samples = 100

a_true = 3.0
b_true = -5.0
c_true = 10.0

X_obs = np.random.randn(nb_samples, 1)

# linear relation with X_obs plus some noise
Y_obs = (
    a_true * X_obs * X_obs
    + b_true * X_obs
    + c_true
    + 0.1 * np.random.randn(nb_samples, 1)
)


def model(
    a_est: Op,
    b_est: Op,
    c_est: Op,
    x_obs: Op,
) -> Op:
    y_pred = a_est * x_obs * x_obs + b_est * x_obs + c_est
    return y_pred


# init "constants"
x_obs = Variable(name="x_obs", value=X_obs)
y_obs = Variable(name="y_obs", value=Y_obs)

# init variables to optimize
a_est = Variable(name="a_est", value=np.array([0.0]))
b_est = Variable(name="b_est", value=np.array([0.0]))
c_est = Variable(name="c_est", value=np.array([0.0]))

for iteration in range(20):
    y_pred = model(a_est, b_est, c_est, x_obs)

    loss = Linear(
        arg=square((y_pred - y_obs)),
        repr="mean",
        linear_operation=lambda x: np.mean(x),
    )

    print(
        "Gradient a:",
        loss.diff(wrt=a_est, direction=Variable(name="one", value=np.array([1.0]))),
    )

    print(
        "Gradient b:",
        loss.diff(wrt=b_est, direction=Variable(name="one", value=np.array([1.0]))),
    )

    exit(0)

    delta_a = loss.diff(
        wrt=a_est, direction=Variable(name="one", value=np.array([1.0]))
    ).eval()
    delta_b = loss.diff(
        wrt=b_est, direction=Variable(name="one", value=np.array([1.0]))
    ).eval()
    learning_rate = 1e-1

    a_est.value -= learning_rate * delta_a
    b_est.value -= learning_rate * delta_b

    print("loss = ", loss.eval(), "\ta_est = ", a_est.value, "\tb_est = ", b_est.value)
