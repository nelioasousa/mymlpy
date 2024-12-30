import pytest
import numpy as np

from mymlpy.linear.classification import BinaryLogisticRegression
from mymlpy.datasets import split_data
from mymlpy.datasets.normalizers import ZScoreNormalizer


@pytest.fixture
def random_binary_classification_dataset():
    num_features = np.random.randint(1, 5)
    categories_entries = []
    for _ in range(2):
        mean = np.random.randn(num_features)
        mean[:] *= np.random.randint(1, 11, size=num_features)
        # Independent features
        cov = np.eye(num_features) * np.random.rand(num_features) * 2
        data = np.random.multivariate_normal(mean, cov, size=np.random.randint(50, 151))
        categories_entries.append(data)
    X = np.concatenate(categories_entries, axis=0)
    y = np.zeros(shape=(X.shape[0], 1), dtype=X.dtype)
    y[categories_entries[0].shape[0] :] = 1.0
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    X[:] = X[indexes]
    y[:] = y[indexes]
    return X, y


@pytest.mark.parametrize(
    "learn_step,num_epochs,ridge_alpha,generate_weights",
    (
        (10e-2, 1000, 0.0, False),
        (10e-3, 2000, 0.0, True),
        (10e-2, 1000, 0.02, False),
        (10e-3, 2000, 0.02, False),
    ),
)
def test_stochastic_linear_regression(
    learn_step,
    num_epochs,
    ridge_alpha,
    generate_weights,
    random_binary_classification_dataset,
):
    X, y = random_binary_classification_dataset
    proportions = (0.7, 0.3)
    X_train, X_test = split_data(X, proportions=proportions)
    y_train, y_test = split_data(y, proportions=proportions)
    sample_weights = None
    if generate_weights:
        sample_weights = np.random.randint(1, 5, X_train.shape[0])
        sample_weights = sample_weights / sample_weights.sum()
    X_normalizer = ZScoreNormalizer(X_train)
    regressor = BinaryLogisticRegression(learn_step=learn_step, ridge_alpha=ridge_alpha)
    regressor.fit(
        X_normalizer(X_train),
        y_train,
        num_epochs=num_epochs,
        batch_size=50,
        sample_weights=sample_weights,
    )
    y_test_pred = regressor.predict(X_normalizer(X_test))
    loss = regressor.loss(y_test, y_test_pred)
    # 0.5 threshold
    misses = np.logical_xor(y_test_pred > 0.5, y_test > 0.0).sum()
    # TODO: what test to perform?
    assert misses >= 0
    assert loss >= 0.0
