import numpy as np
from main import Linear, Op, Tanh, Variable, fully_connected, square
from unit_vectors import generate_unit_vectors

dimensions = [2, 10, 100, 100, 10, 3]

x_true = Variable(name="x", value=np.random.randn(5, dimensions[0]))
y_true = Variable(name="y", value=np.random.randn(5, dimensions[-1]))


def build_nn(input: Variable) -> tuple[Op, list[Variable]]:
    """
    Builds a fully connected neural network
    """
    x = input
    derivable_variables = []
    for i_layer, (dim_in, dim_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        weight = Variable(
            name=f"weight_{i_layer}",
            value=np.random.randn(dim_in, dim_out) * np.sqrt(2.0 / dim_in),
        )
        derivable_variables.append(weight)

        bias = Variable(name=f"bias_{i_layer}", value=np.zeros((1, dim_out)))
        derivable_variables.append(bias)

        x = fully_connected(input=x, weight=weight, bias=bias)
        x = Tanh(x)

    return x, derivable_variables


y_pred, derivable_variables = build_nn(x_true)

loss = Linear(
    arg=square((y_pred - y_true)),
    repr="mean",
    linear_operation=lambda x: np.mean(x),
)


learning_rate = 1e-4

for iteration in range(100000):
    print(f"Iteration {iteration}, loss = {loss.eval()}")

    gradient_by_variable = {}
    for variable in derivable_variables:
        unit_vectors_list = generate_unit_vectors(variable.value)

        gradient = np.zeros_like(variable.value)

        direction = Variable(name="unit_vector", value=unit_vectors_list[0])

        for direction_value in unit_vectors_list:
            direction.value = direction_value

            gradient += loss.diff(
                wrt=variable,
                direction=direction,
            ).eval()

        gradient_by_variable[variable] = gradient

    for variable, gradient in gradient_by_variable.items():
        variable.value -= learning_rate * gradient
