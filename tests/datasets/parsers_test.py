import pytest
import numpy as np

from mymlpy.datasets import parsers


@pytest.mark.parametrize(
    "missing_repr,case_sensitive,strip_values,missing_data_values,data_values",
    (
        pytest.param(
            "None ",
            True,
            True,
            ("None", " None"),
            (" none", "NONE", " "),
            id="None-CaseSensitive-Strip",
        ),
        pytest.param(
            "none",
            False,
            True,
            ("None", "NONE ", " none "),
            ("", "nan", "NaN"),
            id="None-CaseInsensitive-Strip",
        ),
        pytest.param(
            ("nan", "NaN"),
            True,
            False,
            ("nan", "NaN"),
            ("NAN", "naN"),
            id="(nan, NaN)-CaseSensitive",
        ),
        pytest.param("", True, False, ("",), (" ", "None", "nan"), id="EmptyString"),
    ),
)
def test_missing_data(
    missing_repr, case_sensitive, strip_values, missing_data_values, data_values
):
    @parsers.missing_data(
        missing_repr, case_sensitive=case_sensitive, strip_values=strip_values
    )
    def dummy_parser(value):
        return value

    for value in missing_data_values:
        assert dummy_parser(value) is None, "Failed to parse missing data '%s'" % value
    for value in data_values:
        assert dummy_parser(value) is value, "Failed to parse data '%s'" % value


def test_onehot_parser():
    categories = ("male", "female", "other")
    parser = parsers.OneHotParser(categories)
    for category in categories:
        target = [c == category for c in categories]
        assert parser(category) == target


def test_onehot_parser_ignore_unknowns():
    categories = ("awful", "bad", "average", "good", "perfect")
    parser = parsers.OneHotParser(categories, ignore_unknowns=True)
    target = [False] * len(categories)
    unknowns = ("horrible", "nice", "ordinary")
    for unknown in unknowns:
        assert parser(unknown) == target
    parser.ignore_unknowns = False
    for unknown in unknowns:
        with pytest.raises(ValueError):
            parser(unknown)


def test_onehot_parser_case_insensitive():
    categories = ("N", "S")
    parser = parsers.OneHotParser(categories, case_sensitive=False)
    for category in categories + tuple(c.casefold() for c in categories):
        folded_category = category.casefold()
        target = [folded_category == c.casefold() for c in categories]
        assert parser(category) == target


def test_onehot_parser_strip_values():
    categories = (" empty", " half ", " full")
    parser = parsers.OneHotParser(categories, strip_values=True)
    for category in ("empty", "empty ", "half", "full "):
        split_category = category.strip()
        target = [split_category == c.strip() for c in categories]
        assert parser(category) == target


def test_onehot_parser_case_strip():
    categories = (" eMpTy", " haLF ", " full")
    parser = parsers.OneHotParser(categories, case_sensitive=False, strip_values=True)
    processed_categories = tuple(c.strip().casefold() for c in categories)
    for category in ("empty", "EmPtY ", "  HAlf", "   fULL "):
        processed_category = category.strip().casefold()
        target = [processed_category == c for c in processed_categories]
        assert parser(category) == target


def test_onehot_parser_from_data():
    categories = ("missing", "found", "destroyed")
    repeat = 4
    data = np.concatenate(
        tuple(np.full(repeat, fill_value=i) for i in range(len(categories)))
    )
    np.random.shuffle(data)
    data = np.array([categories[idx] for idx in data])
    sort_key = lambda x: categories.index(x)
    parser = parsers.OneHotParser.from_data(data, sort_key=sort_key)
    for category in categories:
        target = [c == category for c in categories]
        assert parser(category) == target


def test_onehot_parser_not_str():
    categories = (10, tuple(), 0.5, (10, 10.0))
    parser = parsers.OneHotParser(categories)
    for category in categories:
        target = [c == category for c in categories]
        assert parser(category) == target


def test_onehot_parser_not_unique():
    categories = ("up", "left", "right", "down", "up")
    with pytest.raises(ValueError):
        parsers.OneHotParser(categories)


def test_index_parser():
    categories = ("S", "M", "L", "XL")
    parser = parsers.IndexParser(categories)
    for i, category in enumerate(categories):
        assert parser(category) == i


def test_index_parser_ignore_unknowns():
    categories = ("go", "slow", "stop")
    unknowns_index = -1
    parser = parsers.IndexParser(categories, unknowns_index=unknowns_index)
    unknowns = ("halt", "wait", "fast")
    for unknown in unknowns:
        assert parser(unknown) == unknowns_index
    parser.unknowns_index = None
    for unknown in unknowns:
        with pytest.raises(ValueError):
            parser(unknown)
