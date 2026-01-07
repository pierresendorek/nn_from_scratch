import numba
import numpy as np


def conv2d(kernel: np.ndarray, input: np.ndarray, stride: int = 1):
    """_summary_

    Args:
        kernel (np.ndarray): shape : (kernel_size_0, kernel_size_1, nb_channels_in, nb_channels_out)
        input (np.ndarray): shape : (batch_size, nb_channels, image_size_0, image_size_1)
        stride (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    (kernel_size_0, kernel_size_1, nb_channels_in, nb_channels_out) = kernel.shape
    (batch_size, nb_channels, input_size_0, input_size_1) = input.shape

    if nb_channels_in != nb_channels:
        raise ValueError("nb channels should be the same")

    # output shape : (batch_size, nb_channels_out, output_size_0, output_size_1)

    output_size_0 = (input_size_0 - kernel_size_0) // stride + 1
    output_size_1 = (input_size_1 - kernel_size_1) // stride + 1
    output = np.zeros([batch_size, nb_channels_out, output_size_0, output_size_1])

    return conv2d_numba(kernel, input, stride, output)


@numba.njit()
def conv2d_numba(
    kernel: np.ndarray, input: np.ndarray, stride: int, output: np.ndarray
) -> np.ndarray:
    (kernel_size_0, kernel_size_1, nb_channels_in, nb_channels_out) = kernel.shape
    (batch_size, nb_channels, input_size_0, input_size_1) = input.shape
    (_, _, output_size_0, output_size_1) = output.shape

    for b in range(batch_size):
        for c_out in range(nb_channels_out):
            for i in range(output_size_0):
                for j in range(output_size_1):
                    sum = 0.0
                    for c_in in range(nb_channels_in):
                        for ki in range(kernel_size_0):
                            for kj in range(kernel_size_1):
                                sum += (
                                    input[b, c_in, i * stride + ki, j * stride + kj]
                                    * kernel[ki, kj, c_in, c_out]
                                )
                    output[b, c_out, i, j] = sum

    return output


def conv2d_transpose(
    kernel: np.ndarray, input: np.ndarray, stride: int = 1, output_dimensions=None
) -> np.ndarray:
    (kernel_size_0, kernel_size_1, nb_channels_in, nb_channels_out) = kernel.shape
    (batch_size, nb_channels, input_size_0, input_size_1) = input.shape

    if output_dimensions is None:
        output_size_0 = stride * (input_size_0 - 1) + kernel_size_0
        output_size_1 = stride * (input_size_1 - 1) + kernel_size_1
        
    


if __name__ == "__main__":
    kernel = np.random.rand(3, 3, 3, 5)
    input = np.random.rand(1, 3, 100, 100)

    res = conv2d(kernel, input, stride=3)

    print(res)
    print(res.shape)
