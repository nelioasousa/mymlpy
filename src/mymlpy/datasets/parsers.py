from collections.abc import Container
from functools import wraps


def missing_data(missing_data_repr, missing_data_placeholder=None, case_sensitive=True):
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
        return cls(
            categories=_unique_entries(data, sort_key),
            ignore_unknowns=ignore_unknowns,
            case_sensitive=case_sensitive,
            strip_values=strip_values,
        )


class IndexParser(OneHotParser):
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
        return cls(
            categories=_unique_entries(data, sort_key),
            unknowns_index=unknowns_index,
            case_sensitive=case_sensitive,
            strip_values=strip_values,
        )
