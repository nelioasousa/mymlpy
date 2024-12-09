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


def split_data(data_array, proportions, shuffle=True, categorizer=None, copy=False):
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

    `copy` - Whether to return the splits as copies. If `shuffle` is True, `data_array`
    is shuffled in place regardless of the value of `copy`.
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
        if copy:
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
    if copy:
        return tuple(np.copy(split) for split in splits)
    return splits


class KFold:
    def __init__(self, data_array, k, shuffle=False, categorizer=None, copy=False):
        if k < 2:
            raise ValueError("Minimum of k=2 folds.")
        self._k = k
        self._shuffle = shuffle
        self._categorizer = categorizer
        self._next = 0
        self._copy = copy
        self._data = None
        self._folds = None
        self.set_data_array(data_array)

    def _fold_common(self):
        min_size = self._data.shape[0] // self._k
        remainder = self._data.shape[0] - (self._k * min_size)
        folds = []
        start = 0
        for _ in range(self._k):
            rng = (start, start + min_size + (remainder > 0))
            folds.append(rng)
            remainder -= 1
            start = rng[1]
        self._folds = folds

    def _fold_stratified(self):
        categories = {}
        for idx, entry in enumerate(self._data):
            categories.setdefault(self._categorizer(entry), list()).append(idx)
        folds = [list() for _ in range(self._k)]
        for category_idxs in categories.values():
            min_size = len(category_idxs) // self._k
            remainder = len(category_idxs) - (self._k * min_size)
            start = 0
            for i in range(self._k):
                end = start + min_size + (remainder > 0)
                folds[i].extend(category_idxs[start:end])
                remainder -= 1
                start = end
        self._folds = folds

    def __iter__(self):
        self._next = 0
        return self

    def __next__(self):
        if self._next >= self._k:
            raise StopIteration("Exhausted iterator.")
        split = self.get_split(self._next)
        self._next += 1
        return split

    def get_split(self, split_index):
        test_idxs = self._folds[split_index]
        if self._categorizer is None:
            test_idxs = slice(*test_idxs)
        train_idxs = np.ones(self._data.shape[0], dtype=np.bool_)
        train_idxs[test_idxs] = 0
        test = self._data[test_idxs]
        train = self._data[train_idxs]
        if self._copy:
            return np.copy(train), np.copy(test)
        return train, test

    def get_fold(self, fold_index):
        fold_idxs = self._folds[fold_index]
        if self._categorizer is None:
            fold_idxs = slice(*fold_idxs)
        fold = self._data[fold_idxs]
        if self._copy:
            return np.copy(fold)
        return fold

    def set_data_array(self, data_array):
        if data_array.shape[0] < self._k:
            raise ValueError(f"`data_array` must have at least k={self._k} entries.")
        self._data = data_array
        self._next = 0
        if self._shuffle:
            np.random.shuffle(self._data)
        if self._categorizer is None:
            self._fold_common()
        else:
            self._fold_stratified()
