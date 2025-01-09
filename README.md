# Description
Python library for Machine Learning.

Feel free to contribute to the project! While there’s no strict contribution policy or formal guidelines,
adhering to the [pre-commit hooks](./.pre-commit-config.yaml) is mandatory (install `mymlpy[dev]` in a development environment).

# Installation
The library is not available on [PyPI](https://pypi.org/), so you can't install it using `pip install mymlpy`.
This is because `mymlpy` is not intended for production use; it’s a personal project created for learning and
experimentation. However, you can still install it using one of the following methods:

1. Installation directly from GitHub repository:

```
$ pip install git+https://github.com/nelioasousa/mymlpy.git@main
```

See [pip VCS support](https://pip.pypa.io/en/stable/topics/vcs-support/) for more information.

2. Installation from local copy (Linux):

```
$ cd <path>
$ git clone https://github.com/nelioasousa/mymlpy.git
$ cd mymlpy
$ python -m venv venv
or
$ virtualenv venv
$ . venv/bin/activate
(venv) $ pip install .
```


# `mymlpy` package

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

Use Python's built-in `help` function or the `pydoc` standard library to see the full documentaion of each package, module, class or function. E.g:

```
(venv) $ python -m pydoc mymlpy.datasets.split_data
```

Output:

```
Help on function split_data in mymlpy.datasets:

mymlpy.datasets.split_data = split_data(data, proportions, shuffle=False, categorizer=None, return_copies=False)
    Split data based on proportions.

    Arguments:

        `data` (`numpy.typing.ArrayLike`) - Data to split. If not a numpy
        array, one is constructed based on `data`. Can't be empty.

        `proportions` (`typing.Sequence[numbers.Real]`) - Proportions for each
        split set. If the proportions sum to less than 1.0, the last proportion
        is inferred to make the total equal to 1.0.

        `shuffle` (`bool`) - Whether to shuffle `data` in-place before
        splitting. If `data` isn't a numpy array, it will be copied into one
        and this copy that will be shuffled.

        `categorizer` (`collections.abc.Callable[[numpy.ndarray], typing.Hashable]`) -
        A callable that returns a unique hashable object for each entry present
        in `data`. `categorizer` will be called once for each entry in `data`
        and receive it as the first and only positional argument and
        must return a hashable value representing the entry's category.

        `return_copies` (`bool`) - Whether to return the splits as copies or
        views of `data`.

    Returns:

        `tuple[numpy.ndarray]` - The `data` splits.

    Raises:

        `ValueError` - Raised when `data` is empty, `proportions` contains
        negative values, or when `proportions` sum to a total greater than
        1.0.
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
