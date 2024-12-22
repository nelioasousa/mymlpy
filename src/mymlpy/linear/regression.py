import numpy as np


class LinearRegression:
    def __init__(self, ridge_alpha=0.0):
        # Start public
        self.ridge_alpha = ridge_alpha
        # End public
        self._intercept = None
        self._coefficients = None
        self._parameters = None

    @property
    def ridge_alpha(self):
        return self._ridge_alpha

    @ridge_alpha.setter
    def ridge_alpha(self, value):
        alpha = float(value)
        if alpha < 0.0:
            raise ValueError("`ridge_alpha` must be non-negative.")
        self._ridge_alpha = alpha

    @property
    def intercept(self):
        return self._intercept

    @intercept.setter
    def intercept(self, value):
        raise AttributeError("Read-only attribute.")

    @property
    def coefficients(self):
        if self._coefficients is None:
            return None
        return np.copy(self._coefficients)

    @coefficients.setter
    def coefficients(self, value):
        raise AttributeError("Read-only attribute.")

    @property
    def parameters(self):
        if self._parameters is None:
            return None
        return np.copy(self._parameters)

    @parameters.setter
    def parameters(self, value):
        raise AttributeError("Read-only attribute.")

    def _check_X(self, X, features_dim=None):
        if not X.shape[0]:
            raise ValueError("`X` is empty.")
        if len(X.shape) != 2:
            raise ValueError("`X` must be a 2D-array (a matrix).")
        if features_dim is not None and X.shape[1] != features_dim:
            raise ValueError(
                f"Expecting `X` to have observations of size {features_dim} (axis 1 size)."
            )
        return

    def _check_y(self, y, N):
        if len(y.shape) > 3:
            raise ValueError("`y` must be a flat array or a column vector.")
        if len(y.shape) == 2 and y.shape[1] != 1:
            raise ValueError("Multi-output regression not supported.")
        try:
            return y.reshape((N, 1))
        except ValueError:
            raise ValueError("`y` isn't consistent with `X`.")

    def _check_sample_weights(self, sample_weights, N):
        if len(sample_weights) > 3:
            raise ValueError("`sample_weights` must be a flat array or a column vector.")
        try:
            sample_weights = sample_weights.reshape((N, 1))
        except ValueError:
            raise ValueError("`sample_weights` isn't consistent with `X`.")
        sample_weights[:] = sample_weights / sample_weights.sum()
        return sample_weights

    def fit(self, X, y, sample_weights=None):
        X = np.asarray(X)
        self._check_X(X)
        N, P = X.shape
        y = self._check_y(np.asarray(y, dtype=X.dtype), N)
        X_ext = np.concatenate((np.ones((N, 1), dtype=X.dtype), X), axis=1)
        if sample_weights is None:
            X_t = X_ext.transpose()
        else:
            sample_weights = self._check_sample_weights(
                np.asarray(sample_weights, dtype=X.dtype), N
            )
            X_t = (sample_weights * X_ext).transpose()
        if self.ridge_alpha > 0.0:
            l2_reg = self.ridge_alpha * np.identity(P + 1, dtype=X.dtype)
            l2_reg[0, 0] = 0  # Do not regularize intercept
            inv = np.linalg.inv((X_t @ X) + l2_reg)
        else:
            inv = np.linalg.inv(X_t @ X)
        params = inv @ X_t @ y
        self._intercept = params[0, 0]
        self._coefficients = params[:, 0][1:]
        self._parameters = params
        return self.parameters

    def predict(self, X):
        coefficients = self._coefficients
        if coefficients is None:
            raise RuntimeError("Model not fitted do any data.")
        X = np.asarray(X, dtype=coefficients.dtype)
        if len(X.shape) == 1:
            try:
                return X * coefficients + self._intercept
            except ValueError:
                raise ValueError("`X` has an inconsistent dimension.")
        self._check_X(X, len(coefficients))
        return X @ coefficients + self._intercept


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
