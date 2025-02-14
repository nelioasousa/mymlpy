import pytest
import numpy as np

from mymlpy.datasets import ArrayDataset


@pytest.fixture
def random_array():
    size = (np.random.randint(50, 201), np.random.randint(2, 5))
    info_int8 = np.iinfo(np.int8)
    return np.random.randint(info_int8.min, info_int8.max + 1, size=size, dtype=np.int8)


def test_array_dataset_no_copy(random_array):
    ds = ArrayDataset(random_array, copy=False)
    assert ds.get_data_as(copy=False) is random_array


def test_array_dataset_copy(random_array):
    ds = ArrayDataset(random_array, copy=True)
    ds_array = ds.get_data_as(copy=False)
    assert ds_array is not random_array
    assert np.array_equal(ds_array, random_array)
    ds_array[0, 0] = random_array[0, 0] + 1
    assert not np.array_equal(ds_array[0], random_array[0])


def test_array_dataset_batch_iter(random_array):
    ds = ArrayDataset(random_array)
    ds_array_copy = ds.get_data_as(copy=True)
    for idx, entry in enumerate(ds.batch_iter(1)):
        np.array_equal(entry[0], random_array[idx])
    assert not np.array_equal(ds_array_copy, random_array)
