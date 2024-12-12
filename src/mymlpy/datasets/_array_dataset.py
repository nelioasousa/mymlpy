import numpy as np


class ArrayBatchIterator:
    def __init__(self, data, batch_size, return_copies=False):
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
    def __init__(self, data_array, dtype=None, copy=None):
        self.set_data_array(data_array=data_array, dtype=dtype, copy=copy)

    def __iter__(self):
        return iter(self._data)

    def iter(self, batch_size=None, shuffle=False, return_copies=False):
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
        return self.iter(
            batch_size=batch_size, shuffle=shuffle, return_copies=return_copies
        )

    def shuffle(self):
        np.random.shuffle(self._data)

    def set_data_array(self, data_array, dtype=None, copy=None):
        self._data = np.asarray(data_array, dtype=dtype, copy=copy)

    def get_data_array(self, dtype=None, copy=None):
        return np.asarray(self._data, dtype=dtype, copy=copy)
