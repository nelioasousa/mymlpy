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
    indexed_fracs.sort(key=(lambda x: x[1]))
    if distribute < 0:
        for i in range(abs(distribute)):
            sizes[indexed_fracs[i][0]] -= 1
        return tuple(sizes)
    indexed_fracs.reverse()
    for i in range(distribute):
        sizes[indexed_fracs[i][0]] += 1
    return tuple(sizes)


def split_data(data_array, proportions, shuffle=True, categorizer=None, copy=False):
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
            splits_idxs[i].append(category_idxs[start : start + size])
            start += size
    splits = tuple(data_array[sorted(idxs)] for idxs in splits_idxs)
    if copy:
        return tuple(np.copy(split) for split in splits)
    return splits


class KFold:
    def __init__(self, data_array, k, shuffle=True, categorizer=None):
        return

    def __iter__(self):
        return self

    def __next__(self):
        return

    def get_fold(self, fold_index):
        return

    def refold(self, shuffle=True):
        return
