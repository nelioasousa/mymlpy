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
