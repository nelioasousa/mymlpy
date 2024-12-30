import numpy as np

from mymlpy.linear.regression import StochasticLinearRegression
from mymlpy.activations import sigmoid


class BinaryLogisticRegression(StochasticLinearRegression):
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
