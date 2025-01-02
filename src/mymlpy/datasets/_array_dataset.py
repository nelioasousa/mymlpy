import numpy as np


class ArrayBatchIterator:
    """Batch iteration functionality for numpy arrays.

    Attributes:

        No public attributes.
    """

    def __init__(self, data, batch_size, return_copies=False):
        """Default initializer.

        Arguments:

            `data` (`numpy.typing.ArrayLike`) - Data to be iterated over. If
            not a numpy array, one is constructed based on `data`.

            `batch_size` (`int`) - Size of the batch returned.

            `return_copies` (`bool`) - Whether to return the batches as copies
            or views of the underlying numpy array.

        Returns:

            `None` - `self` is initialized and nothing is returned.

        Raises:

            No exception is directly raised.
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
        """Implement iter(self)."""
        return self

    def __next__(self):
        """Implement next(self).

        Raises:

            `StopIteration` - Signal the end of the iterator.
        """
        if self._start >= self._data.shape[0]:
            raise StopIteration()
        end = self._start + self._batch_size
        batch = self._data[self._start : end]
        self._start = end
        if self._return_copies:
            return np.copy(batch)
        return batch


class ArrayDataset:
    """Utility wrapper for numpy arrays representing tabular datasets.

    Attributes:

        `data` (`numpy.ndarray`) - Underlying tabular dataset.
    """

    def __init__(self, data, dtype=None, copy=None):
        """`ArrayDataset` default initializer.

        Arguments:

            `data` (`numpy.typing.ArrayLike`) - Tabular dataset to be wrapped.

            `dtype` (`numpy.typing.DTypeLike`) - Which numpy dtype to use.
            Default is `None`, meaning the dtype is infered by numpy.

            `copy` (`bool`) - Whether to wrap a copy of `data`. Default is
            `None`, meaning that `data` is only copied if necessary.

        Returns:

            `None` - `self` is initialized and nothing is returned.

        Raises:

            No exception is directly raised.
        """
        # Start public
        self.set_data_as(data=data, dtype=dtype, copy=copy)
        # End public

    def set_data_as(self, data, dtype=None, copy=None):
        """Set underlying/wrapped tabular dataset as a numpy array.

        Arguments:

            See the initializer for the meaning of the arguments.

        Returns:

            `None` - Set `self.data` and return `None`.

        Raises:

            No exception is directly raised.
        """
        self._data = np.asarray(data, dtype=dtype, copy=copy)

    def get_data_as(self, dtype=None, copy=None):
        """Get underlying/wrapped tabular dataset as a numpy array.

        Arguments:

            See the initializer for the meaning of the arguments.

        Returns:

            `numpy.typing.ArrayLike` - Underlying/wrapped tabular dataset.

        Raises:

            No exception is directly raised.
        """
        return np.asarray(self._data, dtype=dtype, copy=copy)

    @property
    def data(self):
        """Underlying/wrapped tabular dataset."""
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.asarray(value)

    def __iter__(self):
        """Implement iter(self)."""
        return iter(self._data)

    def iter(self, batch_size=None, shuffle=False, return_copies=False):
        """Build and return an iterator over the wrapped tabular dataset.

        Arguments:

            `batch_size` (`int`) - Size of the iteration batch. Default is
            `None`, meaning that `iter(self)` is returned if `return_copies` is
            `False`, or `iter(np.copy(self))` is returned if `return_copies` is
            `True`.

            `shuffle` (`bool`) - Whether to shuffle the underlying/wrapped
            numpy array before iteration.

            `return_copies` (`bool`) - Whether to return the iteration entries
            as copies or views of the underlying numpy array.

        Returns:

            `typing.Iterator[numpy.ndarray]` - Iterator object.

        Raises:

            No exception is directly raised.
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

        See `self.iter()` for more informations.
        """
        return self.iter(
            batch_size=batch_size, shuffle=shuffle, return_copies=return_copies
        )

    def shuffle(self):
        """Shuffle underlying/wrapped numpy array."""
        np.random.shuffle(self._data)
