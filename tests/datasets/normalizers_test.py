import numpy as np

from mymlpy.datasets.normalizers import ZScoreNormalizer


def test_zscore_normalizer():
    n_size = np.random.randint(5, 11)
    p_size = np.random.randint(1, 6)
    data = np.random.randint(0, 11, size=(n_size, p_size))
    means = data.mean(axis=0, dtype=np.float64)
    stds = data.std(axis=0, dtype=np.float64)
    normalizer = ZScoreNormalizer(data)
    assert np.array_equal(means, normalizer.means)
    assert np.array_equal(stds, normalizer.stds)
    assert data.shape[1:] == normalizer.match_shape
    norm_data = normalizer(data)
    norm_data_check = (data - means) / stds
    assert np.array_equal(norm_data, norm_data_check)
    unnormilized_data = normalizer.unnormalize(norm_data).astype(data.dtype)
    assert np.array_equal(data, unnormilized_data)
