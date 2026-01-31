import numba
import numpy as np

# np.random.seed(0)


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

    output_size_0 = int(np.floor(input_size_0 - kernel_size_0 + 1) / stride)
    output_size_1 = int(np.floor(input_size_1 - kernel_size_1 + 1) / stride)
    output = np.zeros([batch_size, nb_channels_out, output_size_0, output_size_1])

    return conv2d_numba(kernel, input, stride, output)


@numba.njit()
def conv2d_numba(
    kernel: np.ndarray, input: np.ndarray, stride: int, output: np.ndarray
) -> np.ndarray:
    """
    Convolution of 'valid' kind
    """

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
    kernel: np.ndarray, small: np.ndarray, stride, big_dimensions=None
) -> np.ndarray:
    (kernel_size_0, kernel_size_1, nb_channels_big, nb_channels_small) = kernel.shape
    (batch_size, nb_channels, small_size_0, small_size_1) = small.shape

    if nb_channels != nb_channels_small:
        raise ValueError("nb channels should be the same")

    if big_dimensions is None:
        big_size_0 = stride * (small_size_0 - 1) + kernel_size_0
        big_size_1 = stride * (small_size_1 - 1) + kernel_size_1
    else:
        big_size_0, big_size_1 = big_dimensions

    big = np.zeros([batch_size, nb_channels_big, big_size_0, big_size_1])

    return conv2d_transpose_numba(kernel, small, big, stride)


@numba.njit()
def conv2d_transpose_numba(
    kernel: np.ndarray, small: np.ndarray, big: np.ndarray, stride
):
    (kernel_size_0, kernel_size_1, nb_channels_big, nb_channels_small) = kernel.shape
    (batch_size, nb_channels, small_size_0, small_size_1) = small.shape

    big_size_0, big_size_1 = big.shape[2], big.shape[3]

    for b in range(batch_size):
        for j0 in range(big_size_0):
            for j1 in range(big_size_1):
                i0_start = max(0, (j0 - kernel_size_0) // stride)
                i1_start = max(0, (j1 - kernel_size_1) // stride)
                i0_end = (
                    min(j0 // stride + 1, (big_size_0 - kernel_size_0 + 1) // stride)
                    + 1
                )
                i1_end = (
                    min(j1 // stride + 1, (big_size_1 - kernel_size_1 + 1) // stride)
                    + 1
                )

                for i0 in range(i0_start, i0_end):
                    for i1 in range(i1_start, i1_end):
                        if (
                            i0 * stride <= j0
                            and i1 * stride <= j1
                            and (j0 - kernel_size_0) < i0 * stride
                            and (j1 - kernel_size_1) < i1 * stride
                            and (i0 * stride < (big_size_0 - kernel_size_0 + 1))
                            and (i1 * stride < (big_size_1 - kernel_size_1 + 1))
                            and 0 <= i0 < small_size_0
                            and 0 <= i1 < small_size_1
                        ):
                            for c_small in range(nb_channels_small):
                                for c_big in range(nb_channels_big):
                                    big[b, c_big, j0, j1] += (
                                        small[b, c_small, i0, i1]
                                        * kernel[
                                            j0 - i0 * stride,
                                            j1 - i1 * stride,
                                            c_big,
                                            c_small,
                                        ]
                                    )

    return big


if __name__ == "__main__":
    kernel_size = 3
    big_dim = 99
    nb_channels_big = 3
    nb_channels_small = 3

    # kernel dims (kernel_size_0, kernel_size_1, nb_channels_in, nb_channels_out)

    kernel = (
        np.random.rand(kernel_size, kernel_size, nb_channels_big, nb_channels_small)
        ** 3
    )

    big_tensor = np.random.randn(1, 3, 99, 99)
    small_tensor = np.random.randn(1, 3, 48, 48)

    Cv = conv2d(kernel, big_tensor, stride=2)
    print("Cv shape:", Cv.shape)
    print("Expected shape:", small_tensor.shape)
    assert Cv.shape == small_tensor.shape

    u_Cv = np.sum(small_tensor * conv2d(kernel, big_tensor, stride=2))

    Ctut = conv2d_transpose(
        kernel,
        small_tensor,
        stride=2,
        big_dimensions=(big_tensor.shape[2], big_tensor.shape[3]),
    )
    print("Ctut shape:", Ctut.shape)

    print(Ctut.shape)

    Ctut_v = np.sum(
        conv2d_transpose(
            kernel,
            small_tensor,
            stride=2,
            big_dimensions=(big_tensor.shape[2], big_tensor.shape[3]),
        )
        * big_tensor
    )

    print("u_Cv:", u_Cv)
    print("Ctut_v:", Ctut_v)

    # img = np.random.rand(48, 48, 3)

    # res = conv2d_transpose(
    #     kernel,
    #     np.permute_dims(img, [2, 0, 1])[np.newaxis, ...],
    #     stride=2,
    #     big_dimensions=(99, 99),
    # )

    # img_conv = np.permute_dims(res[0], [1, 2, 0])

    # import matplotlib.pyplot as plt

    # # plt.imshow((img_conv != 0).astype(np.float32))  # [20:-20, 20:-20, :])

    # n_img = img_conv - np.min(img_conv)
    # n_img = n_img / np.max(n_img)

    # plt.imshow(n_img)

    # plt.show()
