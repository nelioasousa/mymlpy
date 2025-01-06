"""Non-linear activation functions."""

import numpy as np


def sigmoid(z):
    """Computes the sigmoid function.

    The sigmoid function is defined as 1 / (1 + exp(-x)).

    Parameters:
        `z` (`Union[numbers.Real, numpy.typing.ArrayLike]`) - Input value(s).

    Returns:
        `Union[numbers.Real, numpy.typing.ArrayLike]` - Sigmoid of the input value(s).

    Raises:

        No exception is directly raised.
    """
    z = np.asarray(z)
    return 1 / (1 + np.exp(-z))
