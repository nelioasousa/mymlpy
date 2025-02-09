import numpy as np

from mymlpy.utils.numpy import build_vector


def tss(y):
    """Total Sum of Squares (TSS).

    A measure of the total variance in the response vector `y`.
    """
    y = build_vector(y)
    if not y.shape[0]:
        raise ValueError("`y` can't be empty.")
    return np.square(y - y.mean()).sum()


def rss(y, y_pred):
    """Residual Sum of Squares (RSS).

    A measure of the variance of `y` that isn't explained by `y_pred`.
    """
    y = build_vector(y)
    if not y.shape[0]:
        raise ValueError("`y` can't be empty.")
    y_pred = build_vector(y_pred)
    return np.square(y - y_pred).sum()


def rse(y, y_pred, p):
    """Residual Standard Error (RSE).

    Estimator for the standard deviation of the error term (irreducible error).
    The square of the RSE estimates the variance of the error term.

    Arguments:

        `y` (`numpy.typing.ArrayLike`) - Response vector.

        `y_pred` (`numpy.typing.ArrayLike`) - Predicted response.

        `p` (`int`) - Number of predictors.
    """
    p = int(p)
    if p < 1:
        raise ValueError("`p` must be at least 1.")
    y = build_vector(y)
    if y.shape[0] <= (p + 1):
        raise ValueError("Number of observations must be greater than `p + 1`.")
    return np.sqrt(rss(y, y_pred) / (y.shape[0] - p - 1))


def r_squared(y, y_pred):
    """R squared metric.

    The proportion of variance explained: how much of the inherent variance of
    `y` is explained by `y_pred`. Values closer to 1 indicate a better fit.
    """
    residual = rss(y, y_pred)
    total = tss(y)
    return 1 - residual / total


def mse(y, y_pred):
    """Mean Squared Error (MSE).

    The mean of the squared errors."""
    y = build_vector(y)
    return rss(y, y_pred) / y.shape[0]
