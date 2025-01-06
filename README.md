# `mymlpy`
Python library for Machine Learning.

```
mymlpy (package)
  |
  |-- datasets (package)
  |      |
  |      |-- split_data
  |      |-- TextDatasetBatchIterator
  |      |-- KFold
  |      |-- ArrayBatchIterator
  |      |-- ArrayDataset
  |      |
  |      |-- normalizers (module)
  |      |     |
  |      |     |-- ZScoreNormalizer
  |      |
  |      |-- parsers (module)
  |      |     |
  |      |     |-- missing_data
  |      |     |-- OneHotParser
  |      |     |-- IndexParser
  |
  |-- linear (package)
  |      |
  |      |-- regression (module)
  |      |     |
  |      |     |-- LinearRegression
  |      |     |-- StochasticLinearRegression
  |      |
  |      |-- classification (module)
  |      |     |
  |      |     |-- BinaryLogisticRegression
  |
  |-- activations (module)
  |      |
  |      |-- sigmoid
```

## `mymlpy.datasets`
Package for handling datasets stored in the local file system.

> `def split_data(data, proportions, shuffle=False, categorizer=None, return_copies=False)`
>
> Split data based on proportions.

> `class TextDatasetBatchIterator(file_path, batch_size, parsers, separator=',', skip_lines=0, expand_sequences=False, ignore_errors=False)`
>
> Provides batch iteration functionality for tabular datasets.
>
> Designed for handling out-of-memory datasets and raw datasets that cannot
be directly converted to homogeneous numpy arrays. Supports flexible
iteration, including forward and backward traversal, and stores file
positions of batches to enable faster access during subsequent iterations.

> `class KFold(data, k, shuffle=False, categorizer=None, return_copies=False)`
>
> K-fold cross validation.

> `class ArrayDataset(data, dtype=None, copy=None)`
>
> Utility wrapper for numpy arrays representing tabular datasets.

> `class ArrayBatchIterator(data, batch_size, return_copies=False)`
>
> Batch iteration functionality for numpy arrays.

### `mymlpy.datasets.parsers`
String parsing functionality.

> `def missing_data(missing_data_repr, missing_data_placeholder=None, case_sensitive=True, strip_values=False)`
>
> Decorator to enhance parsers with the ability to handle missing data.
>
> This decorator wraps a simple parser and extends its functionality to
recognize and appropriately process missing values.

> `class OneHotParser(categories, ignore_unknowns=False, case_sensitive=True, strip_values=False)`
>
> Parsers for one-hot encoding of categorical data.

> `class IndexParser(categories, unknowns_index=None, case_sensitive=True, strip_values=False)`
>
> Parsers for index encoding of categorical data.

### `mymlpy.datasets.normalizers`
Normalizers for array like datasets.

> `class ZScoreNormalizer(data=None)`
>
> Implement '0 mean' and '1 standard deviation' normalization.

## `mymlpy.linear`
Linear models.

### `mymlpy.linear.regression`
Linear regression models.

> `class LinearRegression(ridge_alpha=0.0)`
>
> Linear regression using Ordinary Least Squares (OLS).

> `class StochasticLinearRegression(learn_step, ridge_alpha=0.0, early_stopper=None)`
>
> Linear regression using Stochastic Gradient Descent (SGD).

### `mymlpy.linear.classification`
Linear classification models.

> `class BinaryLogisticRegression(learn_step, ridge_alpha=0.0, early_stopper=None)`
>
> Binary logistic regression using Stochastic Gradient Descent (SGD).

## `mymlpy.activations`
Non-linear activation functions.

> `def sigmoid(z)`
>
> Computes the sigmoid function.
