from collections.abc import Container
from functools import wraps


def missing_data(missing_data_repr, missing_data_placeholder=None, case_sensitive=True):
    """Decorator to enhance parsers with the ability to handle missing data.

    This decorator wraps a simple parser and extends its functionality to recognize and
    appropriately process missing values.
    """
    if isinstance(missing_data_repr, str):
        missing_data_repr = (missing_data_repr,)
    if isinstance(missing_data_repr, Container) and not isinstance(
        missing_data_repr, bytes
    ):
        try:
            # Faster lookup
            missing_data_repr = frozenset(missing_data_repr)
        except TypeError:
            pass
    else:
        raise ValueError(
            "`missing_data_repr` must be either a `str` object or an instance of `collections.abc.Container`"
        )

    def parser_decorator(parser):
        @wraps(parser)
        def wrapper(value):
            value_lookup = value if case_sensitive else value.casefold()
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

    The categories are meant to be `str` instances, but if `case_sensitive` is `True`
    and `strip_values` is `False`, they can be any hashable object.

    Methods implemented here:

    __call__(self, value)
        Implement self(value).

    from_data(cls, *args, **kwargs)
        Custom constructor to build instances from an iterable holding the categories.
    """

    def __init__(
        self, categories, ignore_unknowns=False, case_sensitive=True, strip_values=False
    ):
        categories = categories if case_sensitive else (c.casefold() for c in categories)
        categories = (c.strip() for c in categories) if strip_values else categories
        self.categories = tuple(categories)
        if len(self.categories) != len(frozenset(self.categories)):
            raise ValueError("`categories` must contain unique entries.")
        self.ignore_unknowns = ignore_unknowns
        self.case_sensitive = case_sensitive
        self.strip_values = strip_values

    def __call__(self, value):
        value = value if self.case_sensitive else value.casefold()
        value = value.strip() if self.strip_values else value
        onehot = [value == category for category in self.categories]
        if not self.ignore_unknowns and not sum(onehot):
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
        """Build parser using categories stored in the `data` iterator.

        The categories are meant to be `str` instances, but if `case_sensitive` is `True`
        and `strip_values` is `False`, they can be any hashable object.

        `data` - An iterator holding the target categories. If `data` has a `flatten`
        method (similar to `numpy.ndarray`) then `data.flatten()` is called and the
        result is assigned as the new `data`.

        `sort_key` - Function returning the key values for the sorting operation. The
        default is `None`, meaning that the intrinsic order of the elements is used.

        `ignore_unknowns` - If ser to `True`, unknown categories don't raise `ValueError`
        and return a list with all category flags set to `False`.

        `case_sensitive` - If set to `False` all string comparisons are performed with
        case folded. See `help(str.casefold)` for more information.

        `strip_values` - If set to `False` all strings are stripped of leading and
        trailing whitespaces. See `help(str.strip)` for more information.
        """
        return cls(
            categories=_unique_entries(data, sort_key),
            ignore_unknowns=ignore_unknowns,
            case_sensitive=case_sensitive,
            strip_values=strip_values,
        )


class IndexParser(OneHotParser):
    """Parsers for index encoding of categorical data.

    The categories are meant to be `str` instances, but if `case_sensitive` is `True`
    and `strip_values` is `False`, they can be any hashable object.

    Methods implemented here:

    __call__(self, value)
        Implement self(value).

    from_data(cls, *args, **kwargs)
        Custom constructor to build instances from an iterable holding the categories.
    """

    def __init__(
        self, categories, unknowns_index=None, case_sensitive=True, strip_values=False
    ):
        if unknowns_index is None:
            ignore_unknowns = False
        elif isinstance(unknowns_index, int):
            ignore_unknowns = True
            unknowns_index = int(unknowns_index)
        else:
            raise ValueError("`unknowns_index` must be either `None` or an `int`.")
        super().__init__(
            categories=categories,
            ignore_unknowns=ignore_unknowns,
            case_sensitive=case_sensitive,
            strip_values=strip_values,
        )
        self.unknowns_index = unknowns_index

    def __call__(self, value):
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
        """Build parser using categories stored in the `data` iterator.

        The categories are meant to be `str` instances, but if `case_sensitive` is `True`
        and `strip_values` is `False`, they can be any hashable object.

        `data` - An iterator holding the target categories. If `data` has a `flatten`
        method (similar to `numpy.ndarray`) then `data.flatten()` is called and the
        result is assigned as the new `data`.

        `sort_key` - Function returning the key values for the sorting operation. The
        default is `None`, meaning that the intrinsic order of the elements is used.

        `unknowns_index` - If `unknowns_index` is set to any `int` instance, unknown
        categories don't raise `ValueError` and return `unknowns_index` as the parsed
        index.

        `case_sensitive` - If set to `False` all string comparisons are performed with
        case folded. See `help(str.casefold)` for more information.

        `strip_values` - If set to `False` all strings are stripped of leading and
        trailing whitespaces. See `help(str.strip)` for more information.
        """
        return cls(
            categories=_unique_entries(data, sort_key),
            unknowns_index=unknowns_index,
            case_sensitive=case_sensitive,
            strip_values=strip_values,
        )
