import pytest
import numpy as np

from mymlpy.datasets import split_data, KFold


@pytest.fixture
def random_array():
    return np.random.randn(np.random.randint(500, 1001), np.random.randint(2, 6))


@pytest.fixture
def random_stratified_array():
    num_cols = 4
    num_categories = np.random.randint(2, 5)
    sizes = tuple(np.random.randint(2, 2001) for _ in range(num_categories))
    data = np.empty((sum(sizes), num_cols), dtype=np.uint8)
    start = 0
    for i, size in enumerate(sizes):
        end = start + size
        data[start:end] = np.random.randint(
            i * 2, (i + 1) * 2, (size, num_cols), dtype=np.uint8
        )
        start = end
    categorizer = lambda x: x[0] // 2
    np.random.shuffle(data)
    return sizes, categorizer, data


def test_split_data_empty():
    """Test empty array spliting."""
    num_cols = 4
    proportions = (0.3, 0.7)
    with pytest.raises(ValueError):
        split_data(np.empty((0, num_cols)), proportions)


@pytest.mark.parametrize(
    "bad_proportions",
    (
        pytest.param((0.1, 0.3, 0.7), id="sum>1.0"),
        pytest.param((-0.1, 0.2, 0.9), id="negative&sum==1.0"),
        pytest.param((0.2, -0.3, 0.5), id="negative&sum<1.0"),
        pytest.param((0.3, 0.2, -0.1, 0.7), id="negative&sum>1.0"),
    ),
)
def test_split_data_bad_proportions(random_array, bad_proportions):
    """Test invalid proportions."""
    with pytest.raises(ValueError):
        split_data(random_array, bad_proportions)


def test_split_data_shuffle(random_array):
    """Verify shuffling."""
    proportions = (0.3, 0.7)
    arr_copy = np.copy(random_array)
    split_data(random_array, proportions, shuffle=True)
    assert not np.array_equal(random_array, arr_copy)


@pytest.mark.parametrize(
    "dim,sizes",
    (
        (150, (15, 35, 40, 60)),
        (500, (25, 75, 150, 250)),
        (1000, (100, 200, 700)),
        (10000, (2000, 3000, 5000)),
    ),
)
def test_split_data(dim, sizes):
    """Test array splitting with complete proportions."""
    num_cols = 4
    proportions = tuple(size / dim for size in sizes)
    splits = split_data(np.random.randn(dim, num_cols), proportions)
    for size, split in zip(sizes, splits):
        assert split.shape[0] == size


@pytest.mark.parametrize(
    "dim,sizes",
    (
        (130, (5, 15, 15, 70)),
        (235, (10, 25, 70)),
        (700, (35, 65, 200, 300)),
        (3575, (75, 200, 300, 1700)),
        (33000, (3000, 10000, 15000)),
    ),
)
def test_split_data_incomplete(dim, sizes):
    """Test array splitting with incomplete proportions."""
    num_cols = 5
    proportions = tuple(size / dim for size in sizes)
    splits = split_data(np.random.randn(dim, num_cols), proportions)
    sizes += (dim - sum(sizes),)
    assert len(splits) == len(sizes)
    for size, split in zip(sizes, splits):
        assert split.shape[0] == size


@pytest.mark.parametrize("shuffle", (False, True))
@pytest.mark.parametrize(
    "proportions", ((0.15, 0.35, 0.5), (0.13, 0.17, 0.70), (0.02, 0.03, 0.05, 0.9))
)
def test_split_data_categorizer(random_stratified_array, proportions, shuffle):
    """Test stratified splitting."""
    category_sizes, categorizer, data = random_stratified_array
    # Calculate the number of entries in each split for each category
    # Each category is represented as an index in the `categories_split_sizes` list
    categories_split_sizes = []
    for category_size in category_sizes:
        float_split_sizes = [category_size * p for p in proportions]
        split_sizes = [int(s) for s in float_split_sizes]
        distribute = category_size - sum(split_sizes)
        indexed_fracs = [(i, s - split_sizes[i]) for i, s in enumerate(float_split_sizes)]
        indexed_fracs.sort(key=(lambda x: x[1]), reverse=(distribute > 0))
        increment = 1 if distribute > 0 else -1
        for i in range(abs(distribute)):
            split_sizes[indexed_fracs[i][0]] += increment
        categories_split_sizes.append(split_sizes)
    # Stratified split of `data`
    splits = split_data(data, proportions, shuffle=shuffle, categorizer=categorizer)
    # Store the `data` entries by category
    data_categories = [
        np.empty((size, data.shape[1]), dtype=data.dtype) for size in category_sizes
    ]
    start_points = [0] * len(category_sizes)
    for i, split in enumerate(splits):
        # Segment each split based on the categories
        category_markers = np.array([categorizer(entry) for entry in split])
        ctg_unique, ctg_count = np.unique(category_markers, return_counts=True)
        # Check the number of entries in each category agains `categories_split_sizes`
        for ctg in range(len(category_sizes)):
            try:
                # np.where is ugly, see docs if you don't understand
                count = ctg_count[np.where(ctg_unique == ctg)[0][0]]
            except IndexError:
                count = 0
            assert (
                categories_split_sizes[ctg][i] == count
            ), f"Failed at split-{i} category-{ctg}"
            # Extract and store the entries from `ctg` category
            ctg_mask = category_markers == ctg
            start = start_points[ctg]
            end = start + count
            data_categories[ctg][start:end] = split[ctg_mask]
            start_points[ctg] = end
    # Remount `data` using `data_categories` and compare
    category_markers = np.array([categorizer(entry) for entry in data])
    data_organized = data[np.argsort(category_markers, stable=True)]
    assert np.array_equal(
        data_organized, np.concatenate(data_categories)
    ), "Failed to remount data from splits"


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_kfold_invalid_k(random_array, k):
    with pytest.raises(ValueError):
        KFold(random_array, k)


def test_kfold_not_enough_entries():
    data_size = 5
    offset = 1
    num_cols = 4
    data_array = np.random.randn(data_size, num_cols)
    with pytest.raises(ValueError):
        KFold(data_array, data_size + offset)


@pytest.mark.parametrize("k", tuple(range(2, 5)))
@pytest.mark.parametrize("shuffle", (False, True))
def test_kfold_common(random_array, k, shuffle):
    min_size = random_array.shape[0] // k
    diff = random_array.shape[0] - (k * min_size)
    folds = KFold(random_array, k, shuffle, return_copies=True)
    verify = np.empty((0, random_array.shape[1]), dtype=random_array.dtype)
    for train, test in folds:
        assert test.shape[0] == (min_size + (diff > 0))
        assert (test.shape[0] + train.shape[0]) == random_array.shape[0]
        diff -= 1
        verify = np.concatenate((verify, test))
    assert np.array_equal(random_array, verify)


@pytest.mark.parametrize("k", tuple(range(2, 5)))
@pytest.mark.parametrize("shuffle", (False, True))
def test_kfold_stratified(random_stratified_array, shuffle, k):
    category_sizes, categorizer, data = random_stratified_array
    # Calculate the number of entries in each fold for each category
    min_sizes = []
    diffs = []
    for size in category_sizes:
        min_size = size // k
        min_sizes.append(min_size)
        diffs.append(size - (k * min_size))
    # K-fold split
    folds = KFold(data, k, shuffle, categorizer, return_copies=True)
    # Store the `data` entries by category
    data_categories = tuple(
        np.empty((size, data.shape[1]), dtype=data.dtype) for size in category_sizes
    )
    start_points = [0] * len(category_sizes)
    for i, (train, test) in enumerate(folds):
        assert data.shape[0] == (
            train.shape[0] + test.shape[0]
        ), f"Inconsistency at split-{i}"
        # Test fold integrity
        category_markers = np.array([categorizer(entry) for entry in test])
        ctg_unique, ctg_count = np.unique(category_markers, return_counts=True)
        for ctg in range(len(category_sizes)):
            try:
                # np.where is ugly, see docs if you don't understand
                count = ctg_count[np.where(ctg_unique == ctg)[0][0]]
            except IndexError:
                count = 0
            assert count == (
                min_sizes[ctg] + (diffs[ctg] > 0)
            ), f"Failed at fold-{i} category-{ctg}"
            diffs[ctg] -= 1
            # Extract and store the entries from `ctg` category
            ctg_mask = category_markers == ctg
            start = start_points[ctg]
            end = start + count
            data_categories[ctg][start:end] = test[ctg_mask]
            start_points[ctg] = end
    # Remount `data` using `data_categories` and compare
    category_markers = np.array([categorizer(entry) for entry in data])
    data_organized = data[np.argsort(category_markers, stable=True)]
    assert np.array_equal(
        data_organized, np.concatenate(data_categories)
    ), "Failed to remount data from splits"
