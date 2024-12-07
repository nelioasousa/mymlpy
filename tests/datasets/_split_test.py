import pytest
import numpy as np

from mymlpy.datasets import split_data


@pytest.fixture
def random_array():
    return np.random.randn(np.random.randint(500, 1001), np.random.randint(2, 6))


def test_split_data_empty():
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
    with pytest.raises(ValueError):
        split_data(random_array, bad_proportions)


def test_split_data_shuffle(random_array):
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
    proportions = tuple(size / dim for size in sizes)
    splits = split_data(np.random.randn(dim, 5), proportions)
    sizes += (dim - sum(sizes),)
    assert len(splits) == len(sizes)
    for size, split in zip(sizes, splits):
        assert split.shape[0] == size


def test_split_data_categorizer():
    dim_ctg1 = 10000
    dim_ctg2 = 50000
    dim = dim_ctg1 + dim_ctg2
    data_ctg1 = np.random.randint(0, 2, (dim_ctg1, 4), dtype=np.int64)
    data_ctg2 = np.random.randint(2, 4, (dim_ctg2, 4), dtype=np.int64)
    data = np.empty((dim, 4), dtype=np.int64)
    np.concatenate((data_ctg1, data_ctg2), axis=0, out=data)
    np.random.shuffle(data)
    categorizer = lambda x: bool(x.max() < 2)
    proportions = (0.1, 0.2, 0.3, 0.4)
    sizes_ctg1 = tuple(round(dim_ctg1 * p) for p in proportions)
    sizes_ctg2 = tuple(round(dim_ctg2 * p) for p in proportions)
    splits = split_data(data, proportions, categorizer=categorizer)
    for i, split in enumerate(splits):
        count_ctg1 = sum(categorizer(entry) for entry in split)
        count_ctg2 = split.shape[0] - count_ctg1
        assert (count_ctg1, count_ctg2) == (sizes_ctg1[i], sizes_ctg2[i])
