import numpy as np

from mymlpy.datasets import ArrayDataset


_NO_SAMPLE_WEIGHTS = object()


class LinearRegression:
    def __init__(self, ridge_alpha=0.0):
        # Start public
        self.ridge_alpha = ridge_alpha
        # End public
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
        if self._parameters is None:
            return None
        return self._parameters[0, 0]

    @intercept.setter
    def intercept(self, value):
        raise AttributeError("Read-only attribute.")

    @property
    def coefficients(self):
        if self._parameters is None:
            return None
        return np.copy(self._parameters[1:, 0])

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

    def unset_parameters(self):
        self._parameters = None

    def _check_X(self, X, features_dim=None):
        if not X.shape[0]:
            raise ValueError("`X` is empty.")
        if len(X.shape) != 2:
            raise ValueError("`X` must be a 2D-array (a matrix).")
        if features_dim is not None and X.shape[1] != features_dim:
            raise ValueError(
                f"Expecting `X` to have observations of size {features_dim} (axis 1 size)."
            )

    def _check_y(self, y, N):
        if len(y.shape) > 2:
            raise ValueError("`y` must be a flat array or a column vector.")
        if len(y.shape) == 2 and y.shape[1] != 1:
            raise ValueError("Multi-output/Multi-class regression not supported.")
        try:
            y.resize((N, 1))
        except ValueError:
            raise ValueError("`y` isn't consistent with `X`.")

    def _check_sample_weights(self, sample_weights, N):
        if len(sample_weights.shape) > 2:
            raise ValueError("`sample_weights` must be a flat array or a column vector.")
        try:
            sample_weights.resize((N, 1))
        except ValueError:
            raise ValueError("`sample_weights` isn't consistent with `X`.")
        # sample_weights[:] = sample_weights / sample_weights.sum()

    def fit(self, X, y, sample_weights=None):
        X, y, sample_weights = self._check(X, y, sample_weights)
        N, P = X.shape
        X_ext = np.concatenate((np.ones((N, 1), dtype=X.dtype), X), axis=1)
        if sample_weights is None:
            X_t = X_ext.transpose()
        else:
            X_t = ((sample_weights / sample_weights.sum()) * X_ext).transpose()
        if self._ridge_alpha > 0.0:
            l2_reg = self._ridge_alpha * np.identity(P + 1, dtype=X.dtype)
            l2_reg[0, 0] = 0  # Do not regularize intercept
            inv = np.linalg.inv((X_t @ X_ext) + l2_reg)
        else:
            inv = np.linalg.inv(X_t @ X_ext)
        parameters = inv @ X_t @ y
        self._parameters = parameters
        return self.parameters

    def _check(self, X, y, sample_weights):
        X = np.asarray(X, dtype=np.float64)
        self._check_X(X)
        y = np.asarray(y, dtype=X.dtype)
        self._check_y(y, X.shape[0])
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=X.dtype)
            self._check_sample_weights(sample_weights, X.shape[0])
        return X, y, sample_weights

    def _check_against(self, X, y=None, sample_weights=_NO_SAMPLE_WEIGHTS, /):
        parameters = self._parameters
        if parameters is None:
            raise RuntimeError("Model not fitted to any data.")
        X = np.asarray(X, dtype=parameters.dtype)
        self._check_X(X, parameters.shape[0] - 1)
        if y is None:
            return X
        y = np.asarray(y, dtype=X.dtype)
        self._check_y(y, X.shape[0])
        if sample_weights is _NO_SAMPLE_WEIGHTS:
            return X, y
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, X.dtype)
            self._check_sample_weights(sample_weights, X.shape[0])
        return X, y, sample_weights

    def _predict(self, X):
        return X @ self._parameters[1:] + self._parameters[0]

    def predict(self, X):
        X = self._check_against(X)
        return self._predict(X)

    def _errors(self, X, y):
        y_pred = self._predict(X)
        return y - y_pred

    def errors(self, X, y):
        X, y = self._check_against(X, y)
        return self._errors(X, y)

    def _loss(self, X, y, sample_weights):
        errors = self._errors(X, y)
        if sample_weights is None:
            loss = (errors**2).sum() / errors.shape[0]
        else:
            loss = (sample_weights.flatten() / sample_weights.sum()) @ (
                errors.flatten() ** 2
            )
        if self._ridge_alpha > 0.0:
            loss += self._ridge_alpha * (self._parameters[1:] ** 2).sum() / 2
        return loss

    def loss(self, X, y, sample_weights=None):
        X, y, sample_weights = self._check_against(X, y, sample_weights)
        return self._loss(X, y, sample_weights)


class StochasticLinearRegression(LinearRegression):
    def __init__(self, learn_step, ridge_alpha=0.0, early_stopper=None):
        super().__init__(ridge_alpha)
        self.learn_step = learn_step
        self.early_stopper = early_stopper

    @property
    def learn_step(self):
        return self._learn_step

    @learn_step.setter
    def learn_step(self, value):
        step = float(value)
        if step <= 0.0:
            raise ValueError("`learn_step` must be a positive number.")
        self._learn_step = step

    @property
    def early_stopper(self):
        return self._early_stopper

    @early_stopper.setter
    def early_stopper(self, value):
        if value is None:
            self._early_stopper = None
            return
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

    def _fit_step(self, X, y, sample_weights):
        parameters = self._parameters
        errors = self._errors(X, y)
        if sample_weights is None:
            N = X.shape[0]
            intercept_step = errors.sum() / N
            coefficients_step = (X.transpose() @ errors) / N
        else:
            sample_weights = sample_weights / sample_weights.sum()
            intercept_step = sample_weights.flatten() @ errors.flatten()
            coefficients_step = X.transpose() @ (sample_weights * errors)
        if self._ridge_alpha > 0.0:
            coefficients_step[:] -= self._ridge_alpha * parameters[1:]
        parameters[0, 0] += self._learn_step * intercept_step
        parameters[1:] += self._learn_step * coefficients_step
        return self.parameters

    def fit_step(self, X, y, sample_weights=None):
        if self._parameters is None:
            X, y, sample_weights = self._check(X, y, sample_weights)
            parameters = np.zeros((X.shape[1] + 1, 1), dtype=X.dtype)
            self._parameters = parameters
        else:
            X, y, sample_weights = self._check_against(X, y, sample_weights)
        return self._fit_step(X, y, sample_weights)

    def fit(self, X, y, num_epochs, batch_size, sample_weights=None):
        if num_epochs < 1:
            raise ValueError("`num_epochs` must be at least 1.")
        X, y, sample_weights = self._check(X, y, sample_weights)
        if batch_size < 1:
            batch_size = X.shape[0]
        self.unset_parameters()
        indexes = ArrayDataset(np.arange(X.shape[0]))
        loss_history = np.empty(num_epochs, dtype=X.dtype)
        for i in range(num_epochs):
            for batch in indexes.batch_iter(batch_size):
                X_train = X[batch]
                y_train = y[batch]
                sample_weights_train = (
                    None if sample_weights is None else sample_weights[batch]
                )
                self.fit_step(X_train, y_train, sample_weights_train)
            parameters = self.parameters
            loss = self._loss(X, y, sample_weights)
            loss_history[i] = loss
            if self._early_stopper is not None:
                loss_history_seg = loss_history[: (i + 1)]
                if self._early_stopper(loss_history_seg):
                    return parameters, np.copy(loss_history_seg)
        return parameters, loss_history
