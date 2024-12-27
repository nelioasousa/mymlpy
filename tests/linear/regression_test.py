import pytest
import numpy as np

from mymlpy.linear.regression import LinearRegression, StochasticLinearRegression
from mymlpy.datasets import split_data


@pytest.fixture
def random_linear_dataset():
    size = np.random.randint(100, 301)
    num_features = np.random.randint(1, 5)
    coefficients = np.random.normal(scale=2, size=num_features)
    intercept = np.random.randn() * 10
    irreducible_error = 0.5
    columns = []
    for _ in range(num_features):
        mean = np.random.randint(0, 101)
        std = np.random.rand() * mean
        std = std if std > 0 else 1.0
        columns.append(np.random.normal(loc=mean, scale=std, size=size))
    X = np.stack(columns, axis=1)
    y = X @ coefficients.reshape((num_features, 1)) + intercept
    # Random error with mean 0 and std deviation equal to `irreducible_error`
    y[:] += np.random.normal(scale=irreducible_error, size=(X.shape[0], 1))
    return intercept, coefficients, X, y, irreducible_error


@pytest.mark.parametrize(
    "ridge_alpha,generate_weights",
    ((0.0, False), (0.0, True), (0.02, False), (0.02, True)),
)
def test_linear_regression(ridge_alpha, generate_weights, random_linear_dataset):
    intercept, coefficients, X, y, irreducible_error = random_linear_dataset
    proportions = (0.7, 0.3)
    X_train, X_test = split_data(X, proportions=proportions)
    sample_weights = None
    if generate_weights:
        sample_weights = np.random.randint(1, 5, X_train.shape[0])
        sample_weights = sample_weights / sample_weights.sum()
    y_train, _ = split_data(y, proportions=proportions)
    regressor = LinearRegression(ridge_alpha=ridge_alpha)
    regressor.fit(X_train, y_train, sample_weights=sample_weights)
    y_pred = regressor.predict(X_test)
    y_real = X_test @ coefficients.reshape((-1, 1)) + intercept
    error = ((y_pred - y_real) ** 2).sum() / X_test.shape[0]
    # TODO: what test to perform?
    assert error >= irreducible_error


@pytest.mark.parametrize(
    "learn_step,num_epochs,ridge_alpha,generate_weights",
    (
        (10e-5, 400, 0.0, False),
        (10e-6, 500, 0.0, True),
        (10e-7, 700, 0.02, False),
        (10e-8, 1000, 0.02, True),
    ),
)
def test_stochastic_linear_regression(
    learn_step, num_epochs, ridge_alpha, generate_weights, random_linear_dataset
):
    intercept, coefficients, X, y, irreducible_error = random_linear_dataset
    proportions = (0.7, 0.3)
    X_train, X_test = split_data(X, proportions=proportions)
    y_train, _ = split_data(y, proportions=proportions)
    sample_weights = None
    if generate_weights:
        sample_weights = np.random.randint(1, 5, X_train.shape[0])
        sample_weights = sample_weights / sample_weights.sum()
    regressor = StochasticLinearRegression(learn_step=learn_step, ridge_alpha=ridge_alpha)
    regressor.fit(
        X_train,
        y_train,
        num_epochs=num_epochs,
        batch_size=5,
        sample_weights=sample_weights,
    )
    y_pred = regressor.predict(X_test)
    y_real = X_test @ coefficients.reshape((-1, 1)) + intercept
    error = ((y_pred - y_real) ** 2).sum() / X_test.shape[0]
    # TODO: what test to perform?
    assert error >= irreducible_error
