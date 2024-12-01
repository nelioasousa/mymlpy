from mymlpy.datasets.tabular import TabularDatasetBatchIterator

class TestTabularDatasetBatchIterator:
    def test_context(self):
        """Check if the file is closed correcly after exiting context."""
        pass

    def test_nested_contexts(self):
        """Check nested contexts restriction."""
        pass

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
