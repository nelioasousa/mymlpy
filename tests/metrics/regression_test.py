import pytest

import mymlpy.metrics.regression as metrics


def test_tss():
    mean = 5.0
    diff = 5.0
    y = (mean - diff, mean, mean + diff)
    tss = metrics.tss(y)
    assert tss == pytest.approx(2 * diff**2)


def test_rss():
    y = (1, 2, 3)
    diffs = (1.0, 0.5, -2.0)
    y_pred = tuple(y[i] + diffs[i] for i in range(len(y)))
    rss = metrics.rss(y, y_pred)
    assert rss == pytest.approx(sum(diff**2 for diff in diffs))


def test_rse():
    y = (1, 2, 3)
    diffs = (1.0, 0.5, -2.0)
    y_pred = tuple(y[i] + diffs[i] for i in range(len(y)))
    rse = metrics.rse(y, y_pred, 1)
    rss = metrics.rss(y, y_pred)
    assert rse == pytest.approx(rss ** (1 / 2))


def test_mse():
    y = (4.0, 2.5, 1.0, 5.0)
    diffs = (1.0, 0.5, -2.0, 1.0)
    y_pred = tuple(y[i] + diffs[i] for i in range(len(y)))
    mse = sum(diff**2 for diff in diffs) / len(diffs)
    assert metrics.mse(y, y_pred) == pytest.approx(mse)
