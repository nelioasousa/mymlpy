import pytest

from mymlpy.datasets import TextDatasetBatchIterator

from importlib.resources import files as resource_files
from importlib.resources import as_file
from contextlib import ExitStack


@pytest.fixture
def tabular_01_path():
    resource = resource_files("tests.datasets").joinpath("resources/tabular_01.txt")
    with as_file(resource) as fpath:
        yield fpath


@pytest.fixture
def tabular_02_path():
    resource = resource_files("tests.datasets").joinpath("resources/tabular_02.txt")
    with as_file(resource) as fpath:
        yield fpath


@pytest.fixture
def tabular_03_path():
    resource = resource_files("tests.datasets").joinpath("resources/tabular_03.txt")
    with as_file(resource) as fpath:
        yield fpath


@pytest.fixture
def tabular_01_iterator(tabular_01_path):
    return TextDatasetBatchIterator(tabular_01_path, 2, (int, float, int), skip_lines=1)


def test_context(tabular_01_iterator):
    """Check if the file is closed properly after exiting context."""
    assert tabular_01_iterator._file is None
    with tabular_01_iterator:
        fstream = tabular_01_iterator._file
        assert fstream.readable()
    assert fstream.closed


def test_nested_contexts(tabular_01_iterator):
    """Check nested contexts restriction."""
    with ExitStack() as stack:
        stack.enter_context(pytest.raises(RuntimeError))
        stack.enter_context(tabular_01_iterator)
        stack.enter_context(tabular_01_iterator)


@pytest.mark.parametrize("batch_size", list(range(1, 9)))
def test_batch_size(tabular_01_path, batch_size):
    """Check batch size."""
    ds = TextDatasetBatchIterator(
        tabular_01_path, batch_size, (int, float, int), skip_lines=1
    )
    with ds:
        sizes = [len(batch) for batch in ds]
        for i, size in enumerate(sizes[:-1]):
            assert (
                size == batch_size
            ), f"Inconsistent size for batch #{i}: expecting {batch_size}, got {size}"
        assert (
            sizes[-1] <= batch_size
        ), f"Inconsistent size for batch #{len(sizes) - 1}: expecting <= {batch_size}, got {sizes[-1]}"


def test_expand_sequences(tabular_02_path):
    """Check sequence expansion."""
    with open(tabular_02_path, "r") as ds_file:
        lines = ds_file.read().split("\n")[1:]
    targets = [l.split(",")[2].split("-") for l in lines if l]
    parser = lambda x: x.split("-")
    ds = TextDatasetBatchIterator(
        tabular_02_path, 1, (int, float, parser), skip_lines=1, expand_sequences=True
    )
    with ds:
        counter = 0
        for batch in ds:
            target = targets[counter]
            result = batch[0][2:]
            assert (
                target == result
            ), f"Fail expansion at line #{counter}: expecting {target}, got {result}"
            counter += 1


def test_ignore_errors(tabular_03_path):
    """Test `ignore_errors` parameter."""
    ds = TextDatasetBatchIterator(
        tabular_03_path, 1, (int, str, float), ignore_errors=True
    )
    with ExitStack() as stack:
        stack.enter_context(ds)
        batches = list(ds)
        assert len(batches) == 8
        ds.iter(clear_batch_positions=True)
        ds._ignore_errors = False
        stack.enter_context(pytest.raises(ValueError))
        _ = [batch for batch in ds]
        stack.enter_context(pytest.raises(RuntimeError))
        _ = [batch for batch in ds]


def test_uninformed_advance(tabular_01_iterator):
    """Test `advance()` method without batch position information."""
    with tabular_01_iterator as ds:
        batches = list(ds)
        for i in range(len(batches)):
            ds.iter(clear_batch_positions=True)
            batch = ds.advance(i + 1)
            assert batch == batches[i]


def test_informed_advance(tabular_01_iterator):
    """Test `advance()` method with batch position information."""
    with tabular_01_iterator as ds:
        batches = list(ds)
        for i in range(len(batches)):
            ds.iter(clear_batch_positions=False)
            batch = ds.advance(i + 1)
            assert batch == batches[i]


def test_retreat(tabular_01_iterator):
    """Test `retreat()` method."""
    with tabular_01_iterator as ds:
        batches = list(ds)
        for n in range(1, len(batches) + 1):
            batch = ds.retreat(n)
            retreated_batches = [] if batch is None else [batch]
            retreated_batches.extend(ds)
            assert retreated_batches == batches[-(n + 1) :]


def test_uninformed_goto(tabular_01_iterator):
    """Test `goto()` method without batch position information"""
    with tabular_01_iterator as ds:
        batches = list(ds)
        for i in range(len(batches)):
            ds.iter(clear_batch_positions=True)
            assert ds.goto(i) == batches[i]
        # Reset iterator with `goto()`
        assert ds.goto(-1) is None
        assert list(ds) == batches


def test_informed_goto(tabular_01_iterator):
    """Test `goto()` method with batch position information."""
    with tabular_01_iterator as ds:
        batches = list(ds)
        for i in range(len(batches)):
            assert ds.goto(i) == batches[i]
        # Reset iterator with `goto()`
        assert ds.goto(-1) is None
        assert list(ds) == batches


@pytest.mark.parametrize("batch_size", list(range(1, 9)))
def test_iter_with_batch_size(tabular_01_iterator, batch_size):
    """Test `iter_with_batch_size()` iteration mode."""
    with tabular_01_iterator as ds:
        ds.iter_with_batch_size(batch_size)
        sizes = [len(batch) for batch in ds]
        for i, size in enumerate(sizes[:-1]):
            assert (
                size == batch_size
            ), f"Inconsistent size for batch #{i}: expecting {batch_size}, got {size}"
        assert (
            sizes[-1] <= batch_size
        ), f"Inconsistent size for batch #{len(sizes) - 1}: expecting <= {batch_size}, got {sizes[-1]}"
