import pytest
import numpy as np

from mymlpy.datasets import split_data, KFold


@pytest.fixture
def random_array():
    return np.random.randn(np.random.randint(500, 1001), np.random.randint(2, 6))


@pytest.fixture
def random_stratified_array():
    num_cols = 4
    size_ctg1 = np.random.randint(2000, 3001)
    size_ctg2 = np.random.randint(3000, 5001)
    rng_ctg1 = (0, 2)
    rng_ctg2 = (2, 4)
    categorizer = lambda x: bool(x.max() < 2)
    data_ctg1 = np.random.randint(*rng_ctg1, (size_ctg1, num_cols))
    data = np.empty((size_ctg1 + size_ctg2, num_cols), dtype=data_ctg1.dtype)
    data[:size_ctg1] = data_ctg1
    data[size_ctg1:] = np.random.randint(*rng_ctg2, (size_ctg2, num_cols))
    np.random.shuffle(data)
    return (size_ctg1, size_ctg2), categorizer, data


def test_split_data_empty():
    """Test empty array spliting."""
    with pytest.raises(ValueError):
        split_data(np.empty((0, 4)), (0.3, 0.7))


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
    arr_copy = np.copy(random_array)
    split_data(random_array, (0.3, 0.7))
    assert not np.array_equal(random_array, arr_copy)


@pytest.mark.parametrize(
    "dim,sizes",
    (
        pytest.param(150, (15, 35, 40, 60), id="3"),
        pytest.param(500, (25, 75, 150, 250), id="2"),
        pytest.param(1000, (100, 200, 700), id="0"),
        pytest.param(10000, (2000, 3000, 5000), id="1"),
    ),
)
def test_split_data(dim, sizes):
    """Test array splitting with complete proportions."""
    proportions = tuple(size / dim for size in sizes)
    splits = split_data(np.random.randn(dim, 4), proportions)
    for size, split in zip(sizes, splits):
        assert split.shape[0] == size


@pytest.mark.parametrize(
    "dim,sizes",
    (
        pytest.param(130, (5, 15, 15, 70), id="0"),
        pytest.param(235, (10, 25, 70), id="1"),
        pytest.param(700, (35, 65, 200, 300), id="2"),
        pytest.param(3575, (75, 200, 300, 1700), id="3"),
        pytest.param(33000, (3000, 10000, 15000), id="4"),
    ),
)
def test_split_data_incomplete(dim, sizes):
    """Test array splitting with incomplete proportions."""
    proportions = tuple(size / dim for size in sizes)
    splits = split_data(np.random.randn(dim, 5), proportions)
    sizes += (dim - sum(sizes),)
    assert len(splits) == len(sizes)
    for size, split in zip(sizes, splits):
        assert split.shape[0] == size


def test_split_data_categorizer():
    """Test stratified splitting."""
    num_cols = 4
    dim_ctg1 = 2000
    dim_ctg2 = 5000
    dim = dim_ctg1 + dim_ctg2
    data_ctg1 = np.random.randint(0, 2, (dim_ctg1, num_cols))
    data_ctg2 = np.random.randint(2, 4, (dim_ctg2, num_cols))
    data = np.empty((dim, num_cols), dtype=data_ctg1.dtype)
    np.concatenate((data_ctg1, data_ctg2), out=data)
    np.random.shuffle(data)
    categorizer = lambda x: bool(x.max() < 2)
    proportions = (0.1, 0.2, 0.3, 0.4)
    sizes_ctg1 = tuple(round(dim_ctg1 * p) for p in proportions)
    sizes_ctg2 = tuple(round(dim_ctg2 * p) for p in proportions)
    splits = split_data(data, proportions, categorizer=categorizer)
    for i, split in enumerate(splits):
        count_ctg1 = sum(categorizer(entry) for entry in split)
        count_ctg2 = split.shape[0] - count_ctg1
        assert count_ctg1 == sizes_ctg1[i]
        assert count_ctg2 == sizes_ctg2[i]


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_kfold_invalid_k(random_array, k):
    with pytest.raises(ValueError):
        _ = KFold(random_array, k)


def test_kfold_not_enough_entries():
    data_size = 5
    offset = 1
    num_cols = 4
    data_array = np.random.randn(data_size, num_cols)
    with pytest.raises(ValueError):
        _ = KFold(data_array, data_size + offset)


@pytest.mark.parametrize("k", tuple(range(2, 5)))
@pytest.mark.parametrize("shuffle", (False, True))
def test_kfold_common(random_array, k, shuffle):
    min_size = random_array.shape[0] // k
    diff = random_array.shape[0] - (k * min_size)
    folds = KFold(random_array, k, shuffle, copy=True)
    verify = np.empty((0, random_array.shape[1]), dtype=random_array.dtype)
    for train, test in folds:
        assert test.shape[0] == (min_size + (diff > 0))
        assert (test.shape[0] + train.shape[0]) == random_array.shape[0]
        diff -= 1
        verify = np.concatenate((verify, test))
    assert np.array_equal(random_array, verify)


@pytest.mark.parametrize("k", tuple(range(2, 5)))
@pytest.mark.parametrize("shuffle", (False, True))
def test_kfold_stratified(random_stratified_array, k, shuffle):
    sizes, categorizer, data = random_stratified_array
    min_size_ctg1 = sizes[0] // k
    min_size_ctg2 = sizes[1] // k
    diff_ctg1 = sizes[0] - (k * min_size_ctg1)
    diff_ctg2 = sizes[1] - (k * min_size_ctg2)
    verify_ctg1 = np.empty((0, data.shape[1]), dtype=data.dtype)
    verify_ctg2 = np.empty((0, data.shape[1]), dtype=data.dtype)
    folds = KFold(data, k, shuffle, categorizer, copy=True)
    for train, test in folds:
        test_ctg1_idxs = np.array([categorizer(entry) for entry in test], dtype=np.bool_)
        test_ctg1_size = test_ctg1_idxs.sum()
        test_ctg2_size = test.shape[0] - test_ctg1_size
        assert test_ctg1_size == (min_size_ctg1 + (diff_ctg1 > 0))
        assert test_ctg2_size == (min_size_ctg2 + (diff_ctg2 > 0))
        assert (test.shape[0] + train.shape[0]) == data.shape[0]
        diff_ctg1 -= 1
        diff_ctg2 -= 1
        verify_ctg1 = np.concatenate((verify_ctg1, test[test_ctg1_idxs]))
        verify_ctg2 = np.concatenate((verify_ctg2, test[~test_ctg1_idxs]))
    ctg1_idxs = np.array([categorizer(entry) for entry in data], dtype=np.bool_)
    assert np.array_equal(data[ctg1_idxs], verify_ctg1)
    assert np.array_equal(data[~ctg1_idxs], verify_ctg2)
