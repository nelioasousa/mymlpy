import numpy as np


class LinearRegression:
    def __init__(self, ridge_alpha=0.0):
        # Start public
        self.alpha = ridge_alpha
        # End public
        self._coefficients = None
        self._intercept = None

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = float(value)

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        raise AttributeError("Read-only attribute.")

    @property
    def intercept(self):
        return self._intercept

    @intercept.setter
    def intercept(self, value):
        raise AttributeError("Read-only attribute.")

    def fit(self, X, y, sample_weights=None):
        return

    def predict(self, X):
        return


class StochasticLinearRegression(LinearRegression):
    def __init__(self, ridge_alpha=0.0, early_stopper=None):
        super().__init__(ridge_alpha)
        self.early_stopper = early_stopper

    @property
    def early_stopper(self):
        return self._early_stopper

    @early_stopper.setter
    def early_stopper(self, value):
        early_stopper = value
        loss_example = np.array([1.0, 0.5, 0.01, 0.0])
        try:
            result = early_stopper(loss_example)
        except (TypeError, ValueError):
            raise ValueError("Invalid early stopper.") from None
        else:
            if not isinstance(result, bool):
                raise ValueError("Early stopper must return a boolean.")
        self._early_stopper = early_stopper

    def reset_parameters(self):
        return

    def fit_step(self, X, y, sample_weights=None):
        return

    def fit(self, X, y, num_epochs, batch_size, sample_weights=None):
        return
