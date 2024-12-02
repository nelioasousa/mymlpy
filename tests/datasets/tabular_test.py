import pytest

from mymlpy.datasets.tabular import TabularDatasetBatchIterator

from importlib.resources import files as resource_files
from importlib.resources import as_file
from contextlib import ExitStack


class TestTabularDatasetBatchIterator:
    @pytest.fixture
    def tabular_01_path(self):
        resource = resource_files('tests.datasets').joinpath('resources/tabular_01.txt')
        with as_file(resource) as fpath:
            yield fpath
    
    @pytest.fixture
    def tabular_02_path(self):
        resource = resource_files('tests.datasets').joinpath('resources/tabular_02.txt')
        with as_file(resource) as fpath:
            yield fpath
    
    @pytest.fixture
    def tabular_03_path(self):
        resource = resource_files('tests.datasets').joinpath('resources/tabular_03.txt')
        with as_file(resource) as fpath:
            yield fpath
    
    @pytest.fixture
    def tabular_01_iterator(self, tabular_01_path):
        return TabularDatasetBatchIterator(tabular_01_path, 2, (int, float, int), skip_lines=1)
    
    def test_context(self, tabular_01_iterator):
        """Check if the file is closed correcly after exiting context."""
        assert tabular_01_iterator._file is None
        with tabular_01_iterator:
            fstream = tabular_01_iterator._file
            assert fstream.readable()
        assert fstream.closed
    
    def test_nested_contexts(self, tabular_01_iterator):
        """Check nested contexts restriction."""
        with ExitStack() as stack:
            stack.enter_context(pytest.raises(RuntimeError))
            stack.enter_context(tabular_01_iterator)
            stack.enter_context(tabular_01_iterator)
    
    @pytest.mark.parametrize("batch_size", list(range(1, 9)))
    def test_batch_size(self, tabular_01_path, batch_size):
        """Check the size of the batches."""
        ds = TabularDatasetBatchIterator(tabular_01_path, batch_size, (int, float, int), skip_lines=1)
        with ds:
            sizes = [len(batch) for batch in ds]
            for i, size in enumerate(sizes[:-1]):
                assert size == batch_size, f"Inconsistent size for batch #{i}: expecting {batch_size}, got {size}"
            assert sizes[-1] <= batch_size, f"Inconsistent size for batch #{len(sizes) - 1}: expecting <= {batch_size}, got {sizes[-1]}"
    
    def test_expand_sequences(self, tabular_02_path):
        """Check if parsed sequences are expanded."""
        with open(tabular_02_path, 'r') as ds_file:
            lines = ds_file.read().split('\n')[1:]
        targets = [l.split(',')[2].split('-') for l in lines if l]
        parser = (lambda x: x.split('-'))
        ds = TabularDatasetBatchIterator(tabular_02_path, 1, (int, float, parser), skip_lines=1, expand_sequences=True)
        with ds:
            counter = 0
            for batch in ds:
                target = targets[counter]
                result = batch[0][2:]
                assert target == result, f"Fail expansion at line #{counter}: expecting {target}, got {result}"
                counter += 1
    
    def test_ignore_errors(self, tabular_03_path):
        """Test `ignore_errors` parameter."""
        ds = TabularDatasetBatchIterator(tabular_03_path, 1, (int, str, float), ignore_errors=True)
        with ExitStack() as stack:
            stack.enter_context(ds)
            batchs = [batch for batch in ds]
            assert len(batchs) == 8
            ds.iter(clear_batch_positions=True)
            ds.ignore_errors = False
            stack.enter_context(pytest.raises(ValueError))
            _ = [batch for batch in ds]
            stack.enter_context(pytest.raises(RuntimeError))
            _ = [batch for batch in ds]
    
    def test_uninformed_advance(self, tabular_01_iterator):
        """Test `.advance()` feature without position information."""
        with tabular_01_iterator as ds:
            batchs = [batch for batch in ds]
            for i in range(len(batchs)):
                ds.iter(clear_batch_positions=True)
                batch = ds.advance(i+1)
                assert batch == batchs[i]
    
    def test_informed_advance(self, tabular_01_iterator):
        """Test `.advance()` feature with position information."""
        with tabular_01_iterator as ds:
            batchs = [batch for batch in ds]
            for i in range(len(batchs)):
                ds.iter(clear_batch_positions=False)
                batch = ds.advance(i+1)
                assert batch == batchs[i]
    
    def test_retreat(self, tabular_01_iterator):
        """Test `.retreat()` feature. Retreat always have position information available."""
        with tabular_01_iterator as ds:
            batchs = [batch for batch in ds]
            for n in range(1, len(batchs) + 1):
                batch = ds.retreat(n)
                retreated_batchs = [] if batch is None else [batch]
                retreated_batchs.extend([batch for batch in ds])
                assert retreated_batchs == batchs[-(n+1):]
    
    def test_uninformed_goto(self, tabular_01_iterator):
        """Test `.goto()` feature without position information."""
        with tabular_01_iterator as ds:
            batchs = list(ds)
            for i in range(len(batchs)):
                ds.iter(clear_batch_positions=True)
                assert ds.goto(i) == batchs[i]
            # Reset iterator with `goto()`
            assert ds.goto(-1) is None
            assert list(ds) == batchs
    
    def test_informed_goto(self, tabular_01_iterator):
        """Test `.goto()` feature with position information."""
        with tabular_01_iterator as ds:
            batchs = list(ds)
            for i in range(len(batchs)):
                assert ds.goto(i) == batchs[i]
            # Reset iterator with `goto()`
            assert ds.goto(-1) is None
            assert list(ds) == batchs
    
    def test_iter_with_batch_size(self):
        """Test `.iter_with_batch_size()` iteration mode."""
        pass
