import numpy as np


class ZScoreNormalizer:
    def __init__(self, data=None):
        if data is None:
            self._match_shape = None
            self._means = None
            self._stds = None
            return
        self.fit(data)

    @property
    def means(self):
        if self._means is None:
            return None
        return np.copy(self._means)

    @means.setter
    def means(self, value):
        raise AttributeError("Read-only attribute.")

    @property
    def stds(self):
        if self._stds is None:
            return None
        return np.copy(self._stds)

    @stds.setter
    def stds(self, value):
        raise AttributeError("Read-only attribute.")

    @property
    def match_shape(self):
        return self._match_shape

    @match_shape.setter
    def match_shape(self, value):
        raise AttributeError("Read-only attribute.")

    def fit(self, data):
        data = np.asarray(data)
        self._means = data.mean(axis=0, dtype=np.float64)
        self._stds = data.std(axis=0, dtype=np.float64)
        self._match_shape = self._means.shape
        return self

    def _check_data(self, data):
        match = self._match_shape
        if match is None:
            raise RuntimeError("Normalizer not fitted to any data.")
        data = np.array(data, dtype=np.float64)
        if data.shape[1:] != match:
            raise ValueError("`data` has inconsistent dimensions.")
        return data

    def __call__(self, data, unnormalize=False):
        data = self._check_data(data)
        if unnormalize:
            self._unnormalize(data)
            return data
        self._normalize(data)
        return data

    def _normalize(self, data):
        data[:] = (data - self._means) / self._stds

    def _unnormalize(self, data):
        data[:] = (data * self._stds) + self._means

    def normalize(self, data):
        data = self._check_data(data)
        self._normalize(data)
        return data

    def unnormalize(self, data):
        data = self._check_data(data)
        self._unnormalize(data)
        return data
