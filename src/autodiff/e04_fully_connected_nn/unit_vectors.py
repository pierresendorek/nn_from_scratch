from math import prod

import numpy as np


def generate_unit_vectors(tensor):
    """
    Generates unit vectors for a given tensor shape.
    Each unit vector has the same shape as the tensor, with one element set to 1 and the rest set to 0.
    """

    total_elements = prod(tensor.shape)
    unit_vectors_list = []

    for i in range(total_elements):
        unit_vector = np.zeros(total_elements)
        unit_vector[i] = 1.0
        unit_vector = unit_vector.reshape(tensor.shape)
        unit_vectors_list.append(unit_vector)

    return unit_vectors_list


if __name__ == "__main__":
    unit_vectors_list = generate_unit_vectors(np.zeros((3, 3)))

    for vec in unit_vectors_list:
        print(vec)
        print("---")
