import numpy as np
from pathlib import Path
from collections.abc import Sequence


class TextDatasetBatchIterator:
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
        """
        Provides batch iteration functionality for tabular datasets.

        Designed for handling out-of-memory datasets and raw datasets that cannot be
        directly converted to homogeneous NumPy arrays. Supports flexible iteration,
        including forward and backward traversal, and stores file positions of batches
        to enable faster access during subsequent iterations.

        `file_path` - Path-like object giving the dataset location in the file system.

        `batch_size` - Batch size.

        `parsers` - Sequence of parsers for each dataset column. A parser is simply a
        callable that accepts only one positional argument of type string. If the
        argument can't be understood or isn't a string, `ValueError` must be raised.

        `separator` - String that separates each column in a row/line of data.

        `skip_lines` - Skip the first n lines of the dataset file.

        `expand_sequences` - If set to True, any sequence outputted by a parser (other
        than a string or a bytes object) is expanded/unfolded. No dimension consistency
        check is performed after expansion. When `expand_sequences` is True, the size of
        each entry in a batch or between batches may vary depending on the parsers
        provided.

        `ignore_errors` - Skip lines that are inconsistent with the number of parsers or
        raises `ValueError` during parsing.
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
            self.clear_batch_positions()
        self._batch_size = batch_size

    @property
    def ignore_errors(self):
        return self._ignore_errors

    @ignore_errors.setter
    def ignore_errors(self, value):
        self._ignore_errors = bool(value)

    @property
    def expand_sequences(self):
        return self._expand_sequences

    @expand_sequences.setter
    def expand_sequences(self, value):
        self._expand_sequences = bool(value)

    @property
    def parsers(self):
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
        return self

    def iter_with_batch_size(self, batch_size):
        """Reset the iterator with a different batch size.

        Position information is discarded even when the batch_size remains unchanged.
        """
        self._assert_open_context()
        self._batch_size = batch_size
        self._reset_iterator(clear_batch_positions=True)
        return self

    def iter(self, clear_batch_positions=False):
        """Reset the iterator to the beginning of the dataset.

        Position information is discarded if `clear_batch_positions` is True.
        """
        self._assert_open_context()
        self._reset_iterator(clear_batch_positions=clear_batch_positions)
        return self

    def __next__(self):
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
        """Return the last outputted batch."""
        if dtype is not None and self._batch is not None:
            return np.array(self._batch, dtype=dtype)
        return self._batch

    def advance(self, num_batches=1):
        """Advance `num_batches` from the actual batch.

        The batch landed at is returned. Non-positive `num_batches` is a no-op and
        returns `None`.

        Raises `StopIteration`.
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
        """Advance `num_batches` from the actual batch.

        The batch landed at is returned. `retreat()` doesn't wrap around. If a large
        enough value is supplied by `num_batches`, the iteartor is reseted and `None`
        is returned. Non-positive `num_batches` is a no-op and returns `None`.
        """
        self._assert_open_context()
        if num_batches < 1:
            return None
        return self.goto(self._next_batch_index - 1 - num_batches)

    def goto(self, batch_index):
        """Go to any batch based on it's index and return it.

        Negative `batch_index` is equivalent to a call to `self.iter()`.

        Raises `StopIteration`.
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
        try:
            self._batch_positions.clear()
        except AttributeError:
            pass

    def _reset_iterator(self, clear_batch_positions=False):
        self._next_batch_index = 0
        self._batch = None
        if clear_batch_positions:
            self.clear_batch_positions()
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
