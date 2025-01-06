"""Linear classification models."""

import numpy as np

from mymlpy.linear.regression import StochasticLinearRegression
from mymlpy.activations import sigmoid


class BinaryLogisticRegression(StochasticLinearRegression):
    """Binary logistic regression using Stochastic Gradient Descent (SGD).

    Attributes:

        `ridge_alpha` (`float`) - Ridge regression coefficient (weight decay).

        `intercept` (`Union[None, numpy.float64]`) - Line/hyperplane intercept.

        `coefficients` (`Union[None, numpy.ndarray[numpy.float64]]`) -
        Line/hyperplane coefficients.

        `parameters` (`Union[None, numpy.ndarray[numpy.float64]]`) - Model
        parameters as a column vector, starting with the intercept.

        `learn_step` (`float`) - Training learn step.

        `early_stopper` (`Union[None, collections.abc.Callable[[numpy.ndarray[numpy.float64]], bool]]`) -
        Callable used for early stopping.
    """

    def _predict(self, X):
        z = super()._predict(X)
        return sigmoid(z)

    def _loss(self, y, y_pred, sample_weights):
        element_losses = (y - 1) * np.log(1 - y_pred) - y * np.log(y_pred)
        if sample_weights is None:
            loss = element_losses.sum() / y.shape[0]
        else:
            loss = (
                sample_weights.flatten() / sample_weights.sum()
            ) @ element_losses.flatten()
        if self._ridge_alpha > 0.0:
            loss += self._ridge_alpha * (self._parameters[1:] ** 2).sum() / 2
        return loss

    def loss(self, y, y_pred, sample_weights=None):
        """Cross-entropy loss.

        Arguments:

            `y` (`numpy.typing.ArrayLike`) - Target response.

            `y_pred` (`numpy.typing.ArrayLike`) - Predicted response.

            `sample_weights` (`Union[None, numpy.typing.ArrayLike]`) -
            Entries weights.

        Returns:

            `numpy.float64` - Loss value.

        Raises:

            `RuntimeError` - When `self` isn't fitted to any data.

            `ValueError` - If `y` is empty, or isn't a flat array or column
            vector. If `y_pred` or `sample_weights` aren't compatible with `y`.
        """
        return super().loss(y, y_pred, sample_weights)
