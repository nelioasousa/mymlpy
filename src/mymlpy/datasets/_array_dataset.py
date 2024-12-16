import numpy as np


class ArrayBatchIterator:
    """Implement batch iteration over numpy arrays."""

    def __init__(self, data, batch_size, return_copies=False):
        """`ArrayBatchIterator` default initializer.

        Arguments:

        `data` - Data to be iterated over. If not a numpy array, one is constructed based
        on `data`.

        `batch_size` - Size of the batch returned.

        `return_copies` - Whether to return the batches as copies or views of the
        underlying numpy array.
        """
        try:
            data = np.asarray(data, copy=False)
        except ValueError:
            self._data = np.asarray(data)
            return_copies = False
        else:
            self._data = data.view()
        self._batch_size = int(batch_size)
        self._return_copies = bool(return_copies)
        self._start = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._start >= self._data.shape[0]:
            raise StopIteration()
        end = self._start + self._batch_size
        batch = self._data[self._start : end]
        self._start = end
        if self._return_copies:
            return np.copy(batch)
        return batch


class ArrayDataset:
    """Utility wrapper for numpy arrays representing tabular datasets."""

    def __init__(self, data, dtype=None, copy=None):
        """`ArrayDataset` default initializer.

        Arguments:

        `data` - Array dataset to be wrapped.

        `dtype` - Which numpy dtype to use. Default is `None`, meaning the dtype is
        infered by numpy.

        `copy` - Whether to wrap a copy of `data`. Default is `None`, meaning that `data`
        is only copied if necessary.
        """
        # Start public
        self.set_data_as(data=data, dtype=dtype, copy=copy)
        # End public

    def set_data_as(self, data, dtype=None, copy=None):
        """Set underlying/wrapped numpy array."""
        self._data = np.asarray(data, dtype=dtype, copy=copy)

    def get_data_as(self, dtype=None, copy=None):
        """Get underlying/wrapped numpy array."""
        return np.asarray(self._data, dtype=dtype, copy=copy)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.asarray(value)

    def __iter__(self):
        return iter(self._data)

    def iter(self, batch_size=None, shuffle=False, return_copies=False):
        """Return an iterator for the wrapped data array.

        Arguments:

        `batch_size` - Size of the batch to return. Default is `None`, meaning that
        `iter(data)` is returned if `return_copies` is `False`, or `iter(np.copy(data))`
        is returned if `return_copies` is `True`.

        `shuffle` - Whether to shuffle the underlying/wrapped numpy array before
        iteration.

        `return_copies` - Whether to return the iteration entries as copies or views of
        the underlying numpy array.
        """
        if shuffle:
            np.random.shuffle(self._data)
        if batch_size is None:
            if return_copies:
                return iter(np.copy(self._data))
            return iter(self._data)
        return ArrayBatchIterator(
            data=self._data, batch_size=batch_size, return_copies=return_copies
        )

    def batch_iter(self, batch_size, shuffle=True, return_copies=False):
        """Fall back to `self.iter(...)` with different defaults.

        Arguments:

        `batch_size` - Size of the batch to return. If set to `None`, `iter(data)` is
        returned if `return_copies` is `False`, or `iter(np.copy(data))` is returned if
        `return_copies` is `True`.

        `shuffle` - Whether to shuffle the underlying/wrapped numpy array before
        iteration.

        `return_copies` - Whether to return the iteration entries as copies or views of
        the underlying numpy array.
        """
        return self.iter(
            batch_size=batch_size, shuffle=shuffle, return_copies=return_copies
        )

    def shuffle(self):
        """Shuffle underlying/wrapped numpy array."""
        np.random.shuffle(self._data)
