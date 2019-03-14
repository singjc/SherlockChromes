import numpy as np

def conv1d_out_shape(l_in, kernel_size, padding=0, dilation=1, stride=1):
    return np.floor(((
        l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
