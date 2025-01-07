"""Package for handling datasets stored in the local file system."""

from mymlpy.datasets import normalizers, parsers

from mymlpy.datasets._text_iterator import TextDatasetBatchIterator
from mymlpy.datasets._split import KFold, split_data
from mymlpy.datasets._array_dataset import ArrayDataset, ArrayBatchIterator

__all__ = [
    TextDatasetBatchIterator.__name__,
    KFold.__name__,
    split_data.__name__,
    ArrayDataset.__name__,
    ArrayBatchIterator.__name__,
]
