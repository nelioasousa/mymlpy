"""Normalizers for array like datasets."""

import numpy as np


class ZScoreNormalizer:
    """Implement '0 mean' and '1 standard deviation' normalization.

    Attributes:

        `means` (`numpy.ndarray`) - Normalization means.

        `stds` (`numpy.ndarray`) - Normalization standard deviations.

        `match_shape` (`tuple[int]`) - Shape that the data must matched during
        normalization.
    """

    def __init__(self, data=None):
        """Default initializer.

        Arguments:

            `data` (`numpy.typing.ArrayLike`) - Data to fit the normalizer. If
            not a numpy array, one is constructed based on `data`. Can't be
            empty.

        Return:

            `None` - `self` is initialized and nothing is returned.

        Raises:

            `ValueError` - When `data` is empty.
        """
        if data is None:
            self._match_shape = None
            self._means = None
            self._stds = None
            return
        self.fit(data)

    @property
    def means(self):
        """Normalization means."""
        if self._means is None:
            return None
        return np.copy(self._means)

    @means.setter
    def means(self, value):
        raise AttributeError("Read-only attribute.")

    @property
    def stds(self):
        """Normalization standard deviations."""
        if self._stds is None:
            return None
        return np.copy(self._stds)

    @stds.setter
    def stds(self, value):
        raise AttributeError("Read-only attribute.")

    @property
    def match_shape(self):
        """Shape that the data must matched during normalization."""
        return self._match_shape

    @match_shape.setter
    def match_shape(self, value):
        raise AttributeError("Read-only attribute.")

    def fit(self, data):
        """Fit normalizer to `data`.

        Arguments:

            `data` (`numpy.typing.ArrayLike`) - Data to fit the normalizer. If
            not a numpy array, one is constructed based on `data`. Can't be
            empty.

        Returns:

            Return `self`.

        Raises:

            `ValueError` - When `data` is empty.
        """
        # TODO: not allow empty arrays
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
        """Implement self(data, unnormalize).

        Arguments:

            `data` (`numpy.typing.ArrayLike`) - Data to be normalized or
            unnormalized.

            `unnormalize` (`bool`) - Whether to unnormalize `data`.

        Returns:

            `numpy.ndarray` - Normalized/unnormalized data.

        Raises:

            `RuntimeError` - Normalizer isn't fit to any data.

            `ValueError` - `data` has an inconsistent shape based on the data
            the normalizer was fitted to.
        """
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
        """Apply normalization to `data`.

        Arguments:

            `data` (`numpy.typing.ArrayLike`) - Data to be normalized.

        Returns:

            `numpy.ndarray` - Normalized data.

        Raises:

            `RuntimeError` - Normalizer isn't fit to any data.

            `ValueError` - `data` has an inconsistent shape based on the data
            the normalizer was fitted to.
        """
        data = self._check_data(data)
        self._normalize(data)
        return data

    def unnormalize(self, data):
        """Apply unnormalization to `data`.

        Arguments:

            `data` (`numpy.typing.ArrayLike`) - Data to be unnormalized.

        Returns:

            `numpy.ndarray` - Unnormalized data.

        Raises:

            `RuntimeError` - Normalizer isn't fit to any data.

            `ValueError` - `data` has an inconsistent shape based on the data
            the normalizer was fitted to.
        """
        data = self._check_data(data)
        self._unnormalize(data)
        return data
