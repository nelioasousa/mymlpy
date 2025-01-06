"""Non-linear activation functions."""

import numpy as np


def sigmoid(z):
    z = np.asarray(z)
    return 1 / (1 + np.exp(-z))
