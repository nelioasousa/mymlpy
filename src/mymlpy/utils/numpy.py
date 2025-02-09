import numpy as np


def build_vector(x, dtype=None, copy=None, as_column_vector=True):
    x = np.asarray(x, dtype=dtype, copy=copy)
    num_dims = len(x.shape)
    if (num_dims > 2) or (num_dims == 2 and 1 not in x.shape):
        raise ValueError("Can't be treated as a vector.")
    if as_column_vector:
        return x.reshape((-1, 1))
    return x.reshape((1, -1))


def build_matrix(x, dtype=None, copy=None, flat_to_column=True):
    x = np.asarray(x, dtype=dtype, copy=copy)
    num_dims = len(x.shape)
    if num_dims == 1:
        if flat_to_column:
            return x.reshape((-1, 1))
        return x.reshape((1, -1))
    if num_dims != 2:
        raise ValueError("Can't be treated as a 2D-matrix.")
    return x
