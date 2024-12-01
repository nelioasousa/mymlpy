import pytest

from mymlpy.datasets.tabular import TabularDatasetBatchIterator

from importlib.resources import files as resource_files
from importlib.resources import as_file
from contextlib import ExitStack


class TestTabularDatasetBatchIterator:
    @pytest.fixture
    def tabular_01_dataset(self):
        resource = resource_files('tests.datasets').joinpath('resources/tabular_01.txt')
        with as_file(resource) as fpath:
            yield fpath
    
    def test_context(self, tabular_01_dataset):
        """Check if the file is closed correcly after exiting context."""
        ds = TabularDatasetBatchIterator(tabular_01_dataset, 2, (int, int, float), skip_lines=1)
        assert ds._file is None
        with ds:
            fstream = ds._file
            assert fstream.readable()
        assert fstream.closed
    
    def test_nested_contexts(self, tabular_01_dataset):
        """Check nested contexts restriction."""
        ds = TabularDatasetBatchIterator(tabular_01_dataset, 4, (int, int, float), skip_lines=1)
        with ExitStack() as stack:
            stack.enter_context(pytest.raises(RuntimeError))
            stack.enter_context(ds)
            stack.enter_context(ds)
    
    def test_batch_size(self):
        """Check the size of the batches."""
        pass

    def test_expand_sequences(self):
        """Check if parsed sequences are expanded."""
        pass

    def test_ignore_errors(self):
        """Test `ignore_errors` parameter."""

    def test_uninformed_advance(self):
        """Test `.advance()` feature without position information."""
        pass

    def test_informed_advance(self):
        """Test `.advance()` feature with position information."""
        pass

    def test_uninformed_retreat(self):
        """Test `.retreat()` feature without position information."""
        pass

    def test_informed_retreat(self):
        """Test `.retreat()` feature with position information."""
        pass
    
    def test_uninformed_goto(self):
        """Test `.goto()` feature without position information."""
        pass
    
    def test_informed_goto(self):
        """Test `.goto()` feature with position information."""
        pass

    def test_iter_with_batch_size(self):
        """Test `.iter_with_batch_size()` iteration mode."""
        pass
