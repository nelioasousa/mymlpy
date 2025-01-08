"""String parsing functionality."""

from collections.abc import Sequence
from functools import wraps


def missing_data(
    missing_data_repr,
    missing_data_placeholder=None,
    case_sensitive=True,
    strip_values=False,
):
    """Decorator to enhance parsers with the ability to handle missing data.

    This decorator wraps a simple parser and extends its functionality to
    recognize and appropriately process missing values.

    Arguments:

        `missing_dara_repr` (`Union[str, typing.Sequence[str]]`) - Missing data
        representations. Either a `str` instance or a sequence of `str`s
        (`typing.Sequence[str]`).

        `missing_data_placeholder` (`typing.Any`) - Object to be the
        placeholder for missing data. Defaults to `None`.

        `case_sensitive` (`bool`) - Whether to consider characters case during
        comparison against `missing_data_repr`. Defaults to `True`.

        `strip_values` (`bool`) - Whether to strip leading and trailing
        whitespaces before comparison against `missing_data_repr`. Defaults to
        `False`.

    Returns:

        `collections.abc.Callable[[collections.abc.Callable[P, R]], collections.abc.Callable[P, Union[typing.Any, R]]]` -
        Decorator function.

    Raises:

        `ValueError` - Invalid value passed to `missing_data_repr`.
    """
    if isinstance(missing_data_repr, str):
        missing_data_repr = (missing_data_repr,)
    if isinstance(missing_data_repr, Sequence) and not isinstance(
        missing_data_repr, bytes
    ):
        missing_data_repr = (
            missing_data_repr
            if case_sensitive
            else (r.casefold() for r in missing_data_repr)
        )
        missing_data_repr = (
            (r.strip() for r in missing_data_repr) if strip_values else missing_data_repr
        )
        missing_data_repr = frozenset(missing_data_repr)
    else:
        raise ValueError(
            "`missing_data_repr` must be either a `str` object or an instance of `collections.abc.Sequence`"
        )

    def parser_decorator(parser):
        @wraps(parser)
        def wrapper(value):
            value_lookup = value if case_sensitive else value.casefold()
            value_lookup = value_lookup.strip() if strip_values else value_lookup
            if value_lookup in missing_data_repr:
                return missing_data_placeholder
            return parser(value)

        return wrapper

    return parser_decorator


def _unique_entries(data, sort_key):
    try:
        data = data.flatten()
    except AttributeError:
        pass
    categories = list(frozenset(data))
    categories.sort(key=sort_key)
    return categories


class OneHotParser:
    """Parsers for one-hot encoding of categorical data.

    Attributes:

        `ignore_unknowns` (`bool`) - Whether unknown categories doesn't raise
        `ValueError`.

        `case_sensitive` (`bool`) - Whether to consider case fold during string
        comparisons.

        `strip_values` (`bool`) - Whether to strip leading and trailing
        whitespaces before comparisons.

        `categories` (`typing.Sequence[str]`) - Target categories.
    """

    def __init__(
        self, categories, ignore_unknowns=False, case_sensitive=True, strip_values=False
    ):
        """Default initializer.

        Arguments:

            `categories` (`typing.Sequence[str]`) - A sequence of unique `str`s
            representing the targeted categories.

            `ignore_unknowns` (`bool`) - If ser to `True`, unknown categories
            don't raise `ValueError` and return a list with all category flags
            set to `False`.

            `case_sensitive` (`bool`) - If set to `False` all string
            comparisons are performed with case folded. See
            `help(str.casefold)` for more information.

            `strip_values` (`bool`) - If set to `False` all strings are
            stripped of leading and trailing whitespaces before comparison. See
            `help(str.strip)` for more information.

        Returns:

            `None` - `self` is initialized and nothing is returned.

        Raises:

            `ValueError` - If `categories` entries aren't unique.
        """
        # Start public
        self.ignore_unknowns = ignore_unknowns
        self.case_sensitive = case_sensitive
        self.strip_values = strip_values
        self.categories = categories
        # End public

    @property
    def ignore_unknowns(self):
        """Whether unknown categories doesn't raise `ValueError`."""
        return self._ignore_unknowns

    @ignore_unknowns.setter
    def ignore_unknowns(self, value):
        self._ignore_unknowns = bool(value)

    @property
    def case_sensitive(self):
        """Whether to consider case fold during string comparisons."""
        return self._case_sensitive

    @case_sensitive.setter
    def case_sensitive(self, value):
        case_sensitive = bool(value)
        try:
            old_case_sensitive = self._case_sensitive
        except AttributeError:
            self._case_sensitive = case_sensitive
            return
        self._case_sensitive = case_sensitive
        if old_case_sensitive and not case_sensitive:
            try:
                # Trigger a reprocessing of categories
                self.categories = self._categories
            except (ValueError, AttributeError):
                self._case_sensitive = old_case_sensitive
                raise ValueError(
                    f"Couldn't set `case_sensitive` from {old_case_sensitive} to {case_sensitive}"
                )

    @property
    def strip_values(self):
        """Whether to strip leading and trailing whitespaces before comparisons."""
        return self._strip_values

    @strip_values.setter
    def strip_values(self, value):
        strip_values = bool(value)
        try:
            old_strip_values = self._strip_values
        except AttributeError:
            self._strip_values = strip_values
            return
        self._strip_values = strip_values
        if not old_strip_values and strip_values:
            try:
                # Trigger a reprocessing of categories
                self.categories = self._categories
            except (ValueError, AttributeError):
                self._strip_values = old_strip_values
                raise ValueError(
                    f"Couldn't set `strip_values` from {old_strip_values} to {strip_values}"
                )

    @property
    def categories(self):
        """Target categories."""
        return self._categories

    @categories.setter
    def categories(self, value):
        ctgs = value if self._case_sensitive else (c.casefold() for c in value)
        ctgs = (c.strip() for c in ctgs) if self._strip_values else ctgs
        ctgs = tuple(ctgs)
        if len(ctgs) != len(frozenset(ctgs)):
            raise ValueError("`categories` must contain unique entries.")
        self._categories = ctgs

    def __call__(self, value):
        """Implement self(value).

        Arguments:

            `value` (`str`) - String to parse.

        Returns:

            `typing.Sequence[bool]` - Parsed one-hot (dummy) vector.

        Raises:

            `ValueError` - If `value` doesn't match any category in
            `self.categories` and `self.ignore_unknowns` is set to `False`.
        """
        value = value if self._case_sensitive else value.casefold()
        value = value.strip() if self._strip_values else value
        onehot = [value == category for category in self._categories]
        if not self._ignore_unknowns and not sum(onehot):
            raise ValueError(f"'{value}' doesn't match any category.")
        return onehot

    @classmethod
    def from_data(
        cls,
        data,
        sort_key=None,
        ignore_unknowns=False,
        case_sensitive=True,
        strip_values=False,
    ):
        """Build parser using categories stored in `data`.

        Arguments:

            `data` (`typing.Iterable[str]`) - An iterable holding the targeted
            categories. If `data` has a `flatten` method (similar to
            `numpy.ndarray`) then `data.flatten()` is called and the result is
            used instead.

            `sort_key` (`Union[None, collections.abc.Callable[[str], typing.Any]]`) -
            Function returning the key values for the mandatory sorting
            operation. The default is `None`, meaning that the intrinsic order
            of the elements is used.

            `ignore_unknowns` (`bool`) - If ser to `True`, unknown categories
            don't raise `ValueError` and return a list with all category flags
            set to `False`.

            `case_sensitive` (`bool`) - If set to `False` all string
            comparisons are performed with case folded. See
            `help(str.casefold)` for more information.

            `strip_values` (`bool`) - If set to `False` all strings are
            stripped of leading and trailing whitespaces before comparison. See
            `help(str.strip)` for more information.

        Returns:

            `OneHotParser` - Builded parser.

        Raises:

            No exception is directly raised.
        """
        # TODO: sort_key=False for skip sorting
        return cls(
            categories=_unique_entries(data, sort_key),
            ignore_unknowns=ignore_unknowns,
            case_sensitive=case_sensitive,
            strip_values=strip_values,
        )


class IndexParser(OneHotParser):
    """Parsers for index encoding of categorical data.

    Attributes:

        `ignore_unknowns` (`bool`) - Whether unknown categories doesn't raise
        `ValueError`.

        `case_sensitive` (`bool`) - Whether to consider case fold during string
        comparisons.

        `strip_values` (`bool`) - Whether to strip leading and trailing
        whitespaces before comparisons.

        `categories` (`typing.Sequence[str]`) - Target categories.

        `unknowns_index` (`Union[None, `int`]`) - Index returned when unknown
        categories are encountered. If `None`, unknown categories raise
        `ValueError`.
    """

    def __init__(
        self, categories, unknowns_index=None, case_sensitive=True, strip_values=False
    ):
        """Default initializer.

        Arguments:

            `categories` (`typing.Sequence[str]`) - A sequence of unique `str`s
            representing the targeted categories.

            `unknowns_index` (`Union[None, int]`) - Index returned when unknown
            categories are encountered. If `None`, unknown categories raise
            `ValueError`.

            `case_sensitive` (`bool`) - If set to `False` all string
            comparisons are performed with case folded. See
            `help(str.casefold)` for more information.

            `strip_values` (`bool`) - If set to `False` all strings are
            stripped of leading and trailing whitespaces before comparison. See
            `help(str.strip)` for more information.

        Returns:

            `None` - `self` is initialized and nothing is returned.

        Raises:

            `ValueError` - If `categories` entries aren't unique.
        """
        super().__init__(
            categories=categories,
            case_sensitive=case_sensitive,
            strip_values=strip_values,
        )
        self.unknowns_index = unknowns_index

    @property
    def ignore_unknowns(self):
        """Whether unknown categories doesn't raise `ValueError`."""
        return self._ignore_unknowns

    @ignore_unknowns.setter
    def ignore_unknowns(self, value):
        try:
            _ = self._ignore_unknowns
        except AttributeError:
            self._ignore_unknowns = bool(value)
        else:
            raise AttributeError("Readonly attribute. Set `self.unknowns_index` instead.")

    @property
    def unknowns_index(self):
        """Index returned when unknown categories are encountered.

        If `None`, unknown categories raise `ValueError`.
        """
        return self._unknowns_index

    @unknowns_index.setter
    def unknowns_index(self, value):
        if value is None:
            self._unknowns_index = None
            self._ignore_unknowns = False
            return
        try:
            self._unknowns_index = int(value)
        except ValueError:
            raise ValueError(
                "`unknowns_index` must be either `None` or an `int`."
            ) from None
        self._ignore_unknowns = True

    def __call__(self, value):
        """Implement self(value).

        Arguments:

            `value` (`str`) - String to parse.

        Returns:

            `int` - Parsed category index.

        Raises:

            `ValueError` - If `value` doesn't match any category in
            `self.categories` and `self.unknowns_index` is set to `None`.
        """
        onehot = super().__call__(value)
        if not sum(onehot):
            return self.unknowns_index
        return next(i for i, value in enumerate(onehot) if value)

    @classmethod
    def from_data(
        cls,
        data,
        sort_key=None,
        unknowns_index=None,
        case_sensitive=True,
        strip_values=False,
    ):
        """Build parser using categories stored in `data`.

        Arguments:

            `data` (`typing.Iterable[str]`) - An iterable holding the targeted
            categories. If `data` has a `flatten` method (similar to
            `numpy.ndarray`) then `data.flatten()` is called and the result is
            used instead.

            `sort_key` (`Union[None, collections.abc.Callable[[str], typing.Any]]`) -
            Function returning the key values for the mandatory sorting
            operation. The default is `None`, meaning that the intrinsic order
            of the elements is used.

            `unknowns_index` (`Union[None, int]`) - Index returned when unknown
            categories are encountered. If `None`, unknown categories raise
            `ValueError`.

            `case_sensitive` (`bool`) - If set to `False` all string
            comparisons are performed with case folded. See
            `help(str.casefold)` for more information.

            `strip_values` (`bool`) - If set to `False` all strings are
            stripped of leading and trailing whitespaces before comparison. See
            `help(str.strip)` for more information.

        Returns:

            `IndexParser` - Builded parser.

        Raises:

            No exception is directly raised.
        """
        # TODO: sort_key=False for skip sorting
        return cls(
            categories=_unique_entries(data, sort_key),
            unknowns_index=unknowns_index,
            case_sensitive=case_sensitive,
            strip_values=strip_values,
        )
