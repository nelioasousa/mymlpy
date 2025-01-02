import numpy as np
from pathlib import Path
from collections.abc import Sequence


class TextDatasetBatchIterator:
    """Provides batch iteration functionality for tabular datasets.

    Designed for handling out-of-memory datasets and raw datasets that cannot
    be directly converted to homogeneous numpy arrays. Supports flexible
    iteration, including forward and backward traversal, and stores file
    positions of batches to enable faster access during subsequent iterations.

    Attributes:

        `batch_size` (`int`) - Batch size.

        `expand_sequences` (`bool`) - Whether to expand sequences returned by
        parsers.

        `ignore_errors` (`bool`) - Whether to ignore lines that raises errors.

        `parsers` (`tuple[collections.abc.Callable[[str], typing.Any]]`) -
        Registered parsers.
    """

    def __init__(
        self,
        file_path,
        batch_size,
        parsers,
        separator=",",
        skip_lines=0,
        expand_sequences=False,
        ignore_errors=False,
    ):
        """Default initializer.

        Arguments:

            `file_path` (`Union[str, pathlib.Path]`) - Path-like object giving
            the dataset location in the file system.

            `batch_size` (`int`) - Batch size.

            `parsers` (`typing.Sequence[collections.abc.Callable[[str], typing.Any]]`) -
            Sequence of parsers for each dataset column. A parser is simply a
            callable that accepts only one positional argument of type string.
            If the  argument can't be understood or isn't a string,
            `ValueError` must be raised.

            `separator` (`str`) - String that separates each column in a
            row/line of data.

            `skip_lines` (`int`) - Skip the first `skip_lines` lines of the
            dataset file.

            `expand_sequences` (`bool`) - If set to `True`, any sequence
            outputted by a parser (other than `str` and `bytes` objects) is
            expanded/unfolded. No dimension consistency check is performed
            after expansion. When `expand_sequences` is `True`, the size of
            each entry in a batch or between batches may vary depending on the
            parsers provided.

            `ignore_errors` (`bool`) - If set to `False`, errors encountered
            during iteration are raised. If set to `True`, the lines that
            raised those errors are skipped and no exception is raised. The
            errors that this argument applyes to are parsers `ValueError`s
            and `RuntimeError`s raised from inconsistent line entries (lines
            with more elements to parse than parsers available).

        Returns:

            `None` - `self` is initialized and nothing is returned.

        Raises:

            `ValueError` - If an invalid parser is encountered in `parsers`.
        """
        self._file_path = Path(file_path).resolve()
        self._separator = str(separator)
        self._skip_lines = int(skip_lines)
        # Start public
        self.batch_size = batch_size
        self.expand_sequences = expand_sequences
        self.ignore_errors = ignore_errors
        self.parsers = parsers
        # End public
        self._file = None
        self._next_batch_index = 0
        self._batch_positions = dict()
        self._batch = None

    @property
    def batch_size(self):
        """Batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        batch_size = int(value)
        try:
            old_batch_size = self._batch_size
        except AttributeError:
            self._batch_size = batch_size
            return
        if batch_size != old_batch_size:
            self._batch_positions.clear()
        self._batch_size = batch_size

    @property
    def ignore_errors(self):
        """Whether to ignore lines that raises errors."""
        return self._ignore_errors

    @ignore_errors.setter
    def ignore_errors(self, value):
        self._ignore_errors = bool(value)

    @property
    def expand_sequences(self):
        """Whether to expand sequences returned by parsers."""
        return self._expand_sequences

    @expand_sequences.setter
    def expand_sequences(self, value):
        self._expand_sequences = bool(value)

    @property
    def parsers(self):
        """Registered parsers."""
        return self._parsers

    @parsers.setter
    def parsers(self, value):
        parsers = tuple(value)
        try:
            old_parsers = self._parsers
        except AttributeError:
            old_parsers = None
        if old_parsers is not None and len(parsers) != len(old_parsers):
            raise ValueError("The number of parsers can't change.")
        for i, parser in enumerate(parsers):
            try:
                parser("value")
            except ValueError:
                pass
            except Exception:
                raise ValueError(f"Invalid parser at index {i}") from None
        self._parsers = parsers

    def __enter__(self):
        if self._file is not None:
            raise RuntimeError("Already inside the context. Nesting isn't allowed.")
        self._file = self._file_path.open("r")
        try:
            self._reset_iterator(clear_batch_positions=True)
        except:
            self._file.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
        self._file = None
        return False

    def __iter__(self):
        """Implement iter(self)."""
        return self

    def iter_with_batch_size(self, batch_size):
        """Reset the iterator with a different batch size.

        Position information is discarded independently of the `batch_size`
        value.

        Arguments:

            `batch_size` (`int`) - Batch size.

        Returns:

            Return `self`.

        Raises:

            `RuntimeError` - When no `with` context exists when the method is
            called.
        """
        self._assert_open_context()
        self._batch_size = batch_size
        self._reset_iterator(clear_batch_positions=True)
        return self

    def iter(self, clear_batch_positions=False):
        """Reset the iterator to the beginning of the dataset.

        Arguments:

            `clear_batch_positions` (`bool`) - Whether to clear position
            information before iteration.

        Returns:

            Return `self`.

        Raises:

            `RuntimeError` - When no `with` context exists when the method is
            called.
        """
        self._assert_open_context()
        self._reset_iterator(clear_batch_positions=clear_batch_positions)
        return self

    def __next__(self):
        """Implement next(self).

        Raises:

            `StopIteration` - Signal the end of the iterator.
        """
        self._assert_open_context()
        batch = []
        self._seek_batch_position(self._next_batch_index)
        while len(batch) < self._batch_size or self._batch_size < 1:
            entry_position = self._file.tell()
            nextline = self._file.readline()
            if not nextline:
                break
            elements = nextline.removesuffix("\n").split(self._separator)
            if len(elements) != len(self._parsers):
                if self._ignore_errors:
                    continue
                raise RuntimeError(
                    "Inconsistent entry found: expecting size %d, got %d"
                    % (len(self._parsers), len(elements))
                )
            entry = []
            for i, element in enumerate(elements):
                try:
                    element = self._parsers[i](element)
                except ValueError:
                    if self._ignore_errors:
                        entry = None
                        break
                    raise
                if (
                    isinstance(element, Sequence)
                    and not isinstance(element, (str, bytes))
                    and self._expand_sequences
                ):
                    entry.extend(element)
                else:
                    entry.append(element)
            if entry is None:
                continue
            batch.append(entry)
            if len(batch) == 1:
                batch_position = entry_position
        if not batch:
            raise StopIteration("EOF reached")
        self._store_batch_position(self._next_batch_index, batch_position)
        self._batch = batch
        self._next_batch_index += 1
        return batch

    def get_batch(self, dtype=None):
        """Return the last outputted batch.

        Arguments:

            `dtype` (`numpy.typing.DTypeLike`) - If `None` the raw batch (a
            python `list` object) is returned. Otherwise should be a valid
            numpy dtype so as the batch to be returned as a numpy array.

        Returns:

            `Union[None, list[list[typing.Any]], numpy.ndarray]` - The last
            outputted batch.

        Raises:

            No exception is directly raised.
        """
        if dtype is not None and self._batch is not None:
            return np.array(self._batch, dtype=dtype)
        return self._batch

    def advance(self, num_batches=1):
        """Advance `num_batches` from the actual batch.

        The batch landed at is returned.

        Arguments:

            `num_batches` (`int`) - How many batchs to advance over. The
            current batch is counted. Non-positive `num_batches` is a no-op
            and returns `None`.

        Returns:

            `Union[None, list[list[typing.Any]]]` - The raw batch landed at.

        Raises:

            `StopIteration` - Signal the end of the iterator.

            `RuntimeError` - When no `with` context exists when the method is
            called.
        """
        self._assert_open_context()
        if num_batches < 1:
            return None
        batch_index = self._next_batch_index - 1 + num_batches
        if self._seek_batch_position(batch_index):
            self._next_batch_index = batch_index
        else:
            for _ in range(num_batches - 1):
                next(self)
        return next(self)

    def retreat(self, num_batches=1):
        """Retreat `num_batches` from the actual batch.

        The batch landed at is returned. `self.retreat()` doesn't wrap around.
        If a large enough value is supplied by `num_batches`, the iteartor is
        reseted and `None` is returned.

        Arguments:

            `num_batches` (`int`) - How many batchs to go back over. The
            current batch is counted. Non-positive `num_batches` is a no-op
            and returns `None`.

        Returns:

            `Union[None, list[list[typing.Any]]]` - The raw batch landed at.

        Raises:

            `RuntimeError` - When no `with` context exists when the method is
            called.
        """
        self._assert_open_context()
        if num_batches < 1:
            return None
        return self.goto(self._next_batch_index - 1 - num_batches)

    def goto(self, batch_index):
        """Go to any batch based on it's index and return it.

        Arguments:

            `batch_index` (`int`) - 0-based index of the batch to go to.
            Negative valeus for `batch_index` resets the iterator at it's
            beginning, being equivalent to a call to `self.iter()`.

        Returns:

            `Union[None, list[list[typing.Any]]]` - The raw batch landed at.

        Raises:

            `StopIteration` - Signal the end of the iterator.

            `RuntimeError` - When no `with` context exists when the method is
            called.
        """
        self._assert_open_context()
        if batch_index < 0:
            self._reset_iterator()
            return None
        self._next_batch_index = batch_index
        if not self._seek_batch_position(batch_index):
            self._reset_iterator()
            for _ in range(batch_index):
                next(self)
        return next(self)

    def clear_batch_positions(self):
        """Clear stored batch positions.

        Returns:

            `None` - Nothing is returned.

        Raises:

            No exception is directly raised.
        """
        self._batch_positions.clear()

    def _reset_iterator(self, clear_batch_positions=False):
        self._next_batch_index = 0
        self._batch = None
        if clear_batch_positions:
            self._batch_positions.clear()
        if not self._seek_batch_position(self._next_batch_index):
            self._file.seek(0)
            for _ in range(self._skip_lines):
                self._file.readline()

    def _store_batch_position(self, batch_index, batch_position):
        self._batch_positions[batch_index] = batch_position

    def _seek_batch_position(self, batch_index):
        try:
            self._file.seek(self._batch_positions[batch_index])
            return True
        except KeyError:
            return False

    def _assert_open_context(self):
        if self._file is None:
            raise RuntimeError("A context is required for interacting with the object")
