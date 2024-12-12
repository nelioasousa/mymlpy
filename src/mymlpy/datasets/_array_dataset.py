import numpy as np


class ArrayBatchIterator:
    """Implement batch iteration over numpy arrays."""

    def __init__(self, data, batch_size, return_copies=False):
        """`ArrayBatchIterator` default initializer.

        `data` - Data to be iterated over. If not a numpy array, one is constructed
        based on `data`.

        `batch_size` - Size of the batch returned.

        `return_copies` - Whether to return copies of the batch instead o views from
        the underlying/wrapped numpy array.
        """
        self._data = np.asarray(data)
        self._batch_size = batch_size
        self._return_copies = return_copies
        self._start = 0

    def __iter__(self):
        return self

    def __next__(self):
        end = self._start + self._batch_size
        batch = self._data[self._start : end]
        if not batch.shape[0]:
            raise StopIteration()
        self._start = end
        if self._return_copies:
            return np.copy(batch)
        return batch


class ArrayDataset:
    """Handle numpy arrays representing tabular datasets."""

    def __init__(self, data_array, dtype=None, copy=None):
        """`ArrayDataset` default initializer.

        `data_array` - Array dataset to be wrapped.

        `dtype` - Which numpy dtype to use. Default is `None`, meaning the dtype is
        infered by numpy.

        `copy` - Whether to wrap a copy of `data_array`. Default is `None`, meaning that
        `data_array` is only copied if necessary.
        """
        self.set_data_array(data_array=data_array, dtype=dtype, copy=copy)

    def __iter__(self):
        return iter(self._data)

    def iter(self, batch_size=None, shuffle=False, return_copies=False):
        """Return an iterator for the wrapped data array.

        `batch_size` - Size of the batch returned. Default is `None`, meaning that
        `iter(data_array)` is returned if `return_copies` is `False` or
        `iter(np.copy(data_array))` if `return_copies` is `True`.

        `shuffle` - Whether to shuffle the underlying/wrapped numpy array before
        iteration.

        `return_copies` - Whether to return copies of the batch instead o views from
        the underlying/wrapped numpy array.
        """
        data = self._data
        if shuffle:
            self.shuffle()
        if batch_size is None:
            if return_copies:
                return iter(np.copy(data))
            return iter(self)
        return ArrayBatchIterator(
            data=data, batch_size=batch_size, return_copies=return_copies
        )

    def batch_iter(self, batch_size, shuffle=True, return_copies=False):
        """Fall back to `self.iter(...)` with different defaults."""
        return self.iter(
            batch_size=batch_size, shuffle=shuffle, return_copies=return_copies
        )

    def shuffle(self):
        """Shuffle underlying/wrapped numpy array."""
        np.random.shuffle(self._data)

    def set_data_array(self, data_array, dtype=None, copy=None):
        """Set underlying/wrapped numpy array."""
        self._data = np.asarray(data_array, dtype=dtype, copy=copy)

    def get_data_array(self, dtype=None, copy=None):
        """Get underlying/wrapped numpy array."""
        return np.asarray(self._data, dtype=dtype, copy=copy)
