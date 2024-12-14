import numpy as np


def _check_proportions(target_size, proportions):
    if target_size < 1:
        raise ValueError("`target_size` must be a positive number")
    if min(proportions) < 0.0:
        raise ValueError("`proportions` must contain only non-negative values")
    total = sum(proportions)
    # Floating-point imprecision is not considered
    # Can generate bugs but will be easy to undertand
    if total > 1.0:
        raise ValueError("`proportions` sum to more than 1.0")
    # Proportions that result in values close to 1 are unsafe
    if (1.0 - total) * target_size >= 1.0:
        return tuple(proportions) + (1.0 - total,)
    return tuple(proportions)


def _process_proportions(target_size, proportions):
    float_sizes = [(target_size * p) for p in proportions]
    sizes = [int(size) for size in float_sizes]
    distribute = target_size - sum(sizes)
    if not distribute:
        return tuple(sizes)
    indexed_fracs = [(i, fsize - sizes[i]) for i, fsize in enumerate(float_sizes)]
    indexed_fracs.sort(key=(lambda x: x[1]), reverse=(distribute > 0))
    diff = 1 if distribute > 0 else -1
    for i in range(abs(distribute)):
        sizes[indexed_fracs[i][0]] += diff
    return tuple(sizes)


def split_data(
    data_array, proportions, shuffle=False, categorizer=None, return_copies=False
):
    """Split array based on proportions.

    `data_array` - `numpy.ndarray` to split. Can't be an empty array.

    `proportions` - Proportions for each split set. If the proportions sum to less than
    1.0, the last proportion is inferred to make the total equal to 1.0. If the
    proportions exceed 1.0, a `ValueError` is raised.

    `shuffle` - Whether to shuffle `data_array` in-place before splitting.

    `categorizer` - A callable that returns a unique hashable object for each category in
    `data_array`. It receives each entry in `data_array` (one dimension lest than
    `data_array`) as an argument and must return a hashable value representing the
    entry's category.

    `return_copies` - Whether to return the splits as copies. If `shuffle` is True,
    `data_array` is shuffled in place regardless of the value of `return_copies`.
    """
    proportions = _check_proportions(data_array.shape[0], proportions)
    if shuffle:
        np.random.shuffle(data_array)
    if categorizer is None:
        sizes = _process_proportions(data_array.shape[0], proportions)
        splits = []
        start = 0
        for size in sizes:
            splits.append(data_array[start : start + size])
            start += size
        if return_copies:
            return tuple(np.copy(arr) for arr in splits)
        return tuple(splits)
    categories = {}
    for idx, entry in enumerate(data_array):
        categories.setdefault(categorizer(entry), list()).append(idx)
    splits_idxs = [list() for _ in proportions]
    for category_idxs in categories.values():
        sizes = _process_proportions(len(category_idxs), proportions)
        start = 0
        for i, size in enumerate(sizes):
            splits_idxs[i].extend(category_idxs[start : start + size])
            start += size
    splits = tuple(data_array[sorted(idxs)] for idxs in splits_idxs)
    if return_copies:
        return tuple(np.copy(split) for split in splits)
    return splits


def _train_test_separator(data, test_indexes):
    train_indexes = np.ones(data.shape[0], dtype=np.bool_)
    train_indexes[test_indexes] = False
    return data[train_indexes], data[test_indexes]


class _SplitsIterator:
    def __init__(self, data, splits, return_copies):
        try:
            data = np.asarray(data, copy=False)
        except ValueError:
            self._data = np.asarray(data)
            return_copies = False
        else:
            self._data = data.view()
        self._splits = splits
        self._return_copies = return_copies
        self._next = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            test_indexes = self._splits[self._next]
        except IndexError:
            raise StopIteration()
        self._next += 1
        train, test = _train_test_separator(self._data, test_indexes)
        if self._return_copies:
            return np.copy(train), np.copy(test)
        return train, test


class KFold:
    """K-fold cross validation for numpy arrays.

    Stratified folding is not reliable for very small datasets with categories that have
    fewer entries than the number of folds.
    """

    def __init__(self, data, k, shuffle=False, categorizer=None, return_copies=False):
        # Start public
        self.data = data
        self.k = k
        self.categorizer = categorizer
        self.return_copies = return_copies
        # End public
        self._shape = None
        self._folds = None
        self.prepare_folds(shuffle)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        data = np.asarray(value)
        if not data.shape[0]:
            raise ValueError("Empty arrays are not allowed.")
        self._data = data
        self._shape = None

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        k = int(value)
        if k < 2:
            raise ValueError("The minimum accepted value for `k` is 2.")
        try:
            old_k = self._k
        except AttributeError:
            old_k = None
        if k == old_k:
            return
        self._k = k
        self._shape = None

    @property
    def categorizer(self):
        return self._categorizer

    @categorizer.setter
    def categorizer(self, value):
        categorizer = value
        try:
            old_categorizer = self._categorizer
        except AttributeError:
            # `None` is a valid value
            old_categorizer = object()
        if categorizer is old_categorizer:
            return
        self._categorizer = categorizer
        self._shape = None

    @property
    def return_copies(self):
        return self._return_copies

    @return_copies.setter
    def return_copies(self, value):
        self._return_copies = bool(value)

    def _fold_common(self):
        min_size = self._data.shape[0] // self._k
        remainder = self._data.shape[0] - (self._k * min_size)
        folds = []
        start = 0
        for _ in range(self._k):
            end = start + min_size + (remainder > 0)
            folds.append(slice(start, end))
            start = end
            remainder -= 1
        self._folds = folds

    def _fold_stratified(self):
        categories = {}
        for idx, entry in enumerate(self._data):
            ctg = self._categorizer(entry)
            try:
                categories[ctg].append(idx)
            except KeyError:
                categories[ctg] = [idx]
        # List comprehension ensures that each list is a distinct object
        folds = [list() for _ in range(self._k)]
        for category_idxs in categories.values():
            min_size = len(category_idxs) // self._k
            remainder = len(category_idxs) - (self._k * min_size)
            start = 0
            for i in range(self._k):
                end = start + min_size + (remainder > 0)
                folds[i].extend(category_idxs[start:end])
                start = end
                remainder -= 1
        self._folds = folds

    def _check_folds(self):
        if self._shape != self._data.shape:
            self.prepare_folds()

    def __iter__(self):
        self._check_folds()
        return _SplitsIterator(
            data=self._data, splits=self._folds, return_copies=self._return_copies
        )

    def prepare_folds(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self._data)
        if self._categorizer is None:
            self._fold_common()
        else:
            self._fold_stratified()
        self._shape = self._data.shape

    def get_split(self, split_index):
        """Get (train, test) split at index `split_index`.

        Raise `IndexError` if `split_index` isn't a valid `list` index.
        """
        self._check_folds()
        test_indexes = self._folds[split_index]
        train, test = _train_test_separator(self._data, test_indexes)
        if self._return_copies:
            return np.copy(train), np.copy(test)
        return train, test

    def get_fold(self, fold_index):
        """Get fold (split's test array) at index `fold_index`.

        Raise `IndexError` if `fold_index` isn't a valid `list` index.
        """
        self._check_folds()
        fold_indexes = self._folds[fold_index]
        fold = self._data[fold_indexes]
        if self._return_copies:
            return np.copy(fold)
        return fold
