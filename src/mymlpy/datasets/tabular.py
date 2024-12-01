import numpy as np
from pathlib import Path
from collections.abc import Sequence


class TabularDatasetBatchIterator:
    
    def __init__(self, file_path, batch_size, parsers, separator=",",
            skip_lines=0, expand_sequences=False, ignore_errors=False):
        self.file_path = Path(file_path).resolve()
        self.batch_size = batch_size
        self.separator = separator
        self.skip_lines = skip_lines
        self.expand_sequences = expand_sequences
        self.ignore_errors = ignore_errors
        self.parsers = parsers
        self._file = None
        self._next_batch_index = 0
        self._batch_positions = dict()
        self._batch = None
    
    def __enter__(self):
        if self._file is not None:
            raise RuntimeError(
                "Already inside the context. Nesting isn't allowed"
            )
        self._file = self.file_path.open("r")
        try:
            self._reset_iterator(clear_batch_positions=True)
        except Exception:
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
        self._assert_open_context()
        self.batch_size = batch_size
        self._reset_iterator(clear_batch_positions=True)
        return self
    
    def __next__(self):
        self._assert_open_context()
        batch = []
        self._seek_batch_position(self._next_batch_index)
        while len(batch) < self.batch_size or self.batch_size < 1:
            position = self._file.tell()
            nextline = self._file.readline()
            if not nextline:
                break
            elements = nextline.removesuffix('\n').split(self.separator)
            if len(elements) != len(self.parsers):
                if self.ignore_errors:
                    continue
                raise RuntimeError(
                    "Inconsistent entry found: expecting size %d, got %d"
                    %(len(self.parsers), len(elements))
                )
            entry = []
            try:
                for i, element in enumerate(elements):
                    element = self.parsers[i](element)
                    if isinstance(element, Sequence) and not isinstance(element, (str, bytes)) and self.expand_sequences:
                        entry.extend(element)
                    else:
                        entry.append(element)
            except ValueError:
                if self.ignore_errors:
                    continue
                raise
            batch.append(entry)
            if len(batch) == 1:
                self._store_batch_position(self._next_batch_index, position)
        if not batch:
            raise StopIteration("EOF reached")
        self._batch = batch
        self._next_batch_index += 1
        return batch
    
    def get_batch(self):
        return self._batch
    
    def to_numpy(self, dtype):
        if self._batch is not None:
            return np.array(self._batch, dtype=dtype, copy=True)
        return None
    
    def advance(self, num_batchs=1):
        self._assert_open_context()
        if num_batchs < 1:
            return None
        batch_index = self._next_batch_index - 1 + num_batchs
        if self._seek_batch_position(batch_index):
            self._next_batch_index = batch_index
        else:
            for _ in range(num_batchs - 1):
                next(self)
        return next(self)
    
    def retreat(self, num_batchs=1):
        self._assert_open_context()
        if num_batchs < 1:
            return None
        return self.goto(self._next_batch_index - 1 - num_batchs)
    
    def goto(self, batch_index):
        self._assert_open_context()
        if batch_index < 0:
            return None
        self._next_batch_index = batch_index
        if not self._seek_batch_position(batch_index):
            self._reset_iterator()
            for _ in range(batch_index):
                next(self)
        return next(self)
    
    def _reset_iterator(self, clear_batch_positions=False):
        self._next_batch_index = 0
        self._batch = None
        if clear_batch_positions:
            self._batch_positions.clear()
        if not self._seek_batch_position(self._next_batch_index):
            self._file.seek(0)
            for _ in range(self.skip_lines):
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
            raise RuntimeError(
                "A context is required for interacting with the object"
            )
