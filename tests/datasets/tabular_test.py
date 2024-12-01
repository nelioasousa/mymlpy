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
    def tabular_01_iterator(self, tabular_01_path):
        return TabularDatasetBatchIterator(tabular_01_path, 2, (int, int, float), skip_lines=1)
    
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
