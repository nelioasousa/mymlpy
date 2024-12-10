import pytest

from mymlpy.datasets import parsers


@pytest.mark.parametrize(
    "missing_repr,case_sensitive,missing_data_values,data_values",
    (
        pytest.param(
            "None", True, ("None",), ("none", "NONE", ""), id="None-CaseSensitive"
        ),
        pytest.param(
            "none",
            False,
            ("None", "NONE", "none"),
            ("", "nan", "NaN"),
            id="None-CaseInsensitive",
        ),
        pytest.param(
            ("nan", "NaN"),
            True,
            ("nan", "NaN"),
            ("NAN", "naN"),
            id="(nan, NaN)-CaseSensitive",
        ),
        pytest.param("", True, ("",), (" ", "None", "nan"), id="EmptyString"),
    ),
)
def test_missing_data(missing_repr, case_sensitive, missing_data_values, data_values):
    @parsers.missing_data(missing_repr, case_sensitive=case_sensitive)
    def dummy_parser(value):
        return value

    for value in missing_data_values:
        assert dummy_parser(value) is None, "Failed to parse missing data '%s'" % value
    for value in data_values:
        assert dummy_parser(value) is value, "Failed to parse data '%s'" % value
