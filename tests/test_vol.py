"""Unit tests for options_screener.vol — no network access."""

import math

import pytest

from options_screener.vol import (annualized_yield, bs_put_delta, hv_range,
                                  iv_vs_hv_rank, norm_cdf, realized_vol)


def test_norm_cdf_basics():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(3.0) == pytest.approx(0.99865, abs=1e-4)
    assert norm_cdf(-3.0) == pytest.approx(0.00135, abs=1e-4)


def test_put_delta_is_negative_and_sane():
    # ATM-ish put, 1y out, 20% vol: delta should be roughly -0.4
    d = bs_put_delta(100.0, 100.0, 1.0, 0.04, 0.20)
    assert -0.55 < d < -0.30


def test_put_delta_deep_itm_near_minus_one():
    d = bs_put_delta(50.0, 100.0, 1.0, 0.04, 0.20)
    assert d < -0.90


def test_put_delta_far_otm_near_zero():
    d = bs_put_delta(200.0, 100.0, 1.0, 0.04, 0.20)
    assert -0.05 < d < 0.0


def test_put_delta_monotonic_in_strike():
    deltas = [bs_put_delta(100.0, k, 30 / 365, 0.04, 0.25)
              for k in (80, 90, 100, 110)]
    # higher strike -> more negative delta
    assert deltas == sorted(deltas, reverse=True)


def test_put_delta_rejects_bad_inputs():
    with pytest.raises(ValueError):
        bs_put_delta(100.0, 100.0, 0.0, 0.04, 0.20)
    with pytest.raises(ValueError):
        bs_put_delta(100.0, 100.0, 1.0, 0.04, -0.2)


def _prices(n=300, drift=0.0005, wiggle=0.01):
    # deterministic pseudo price series
    p, out = 100.0, []
    for i in range(n):
        p *= math.exp(drift + wiggle * math.sin(i / 7.0))
        out.append(p)
    return out


def test_realized_vol_positive_and_scaled():
    v = realized_vol(_prices(), 20)
    assert v is not None and v > 0
    # wiggle=0.01 daily amplitude -> annualized vol in a sane band
    assert 0.02 < v < 0.60


def test_realized_vol_needs_enough_history():
    assert realized_vol(_prices(n=10), 20) is None


def test_hv_range_ordering():
    lo, hi, now = hv_range(_prices())
    assert lo <= now <= hi
    assert lo > 0


def test_hv_range_needs_history():
    assert hv_range(_prices(n=100)) is None


def test_iv_vs_hv_rank_midpoint():
    assert iv_vs_hv_rank(0.30, 0.20, 0.40) == pytest.approx(0.5)


def test_iv_vs_hv_rank_clips():
    assert iv_vs_hv_rank(0.99, 0.20, 0.40) == 1.0
    assert iv_vs_hv_rank(0.01, 0.20, 0.40) == 0.0


def test_iv_vs_hv_rank_degenerate_range():
    assert iv_vs_hv_rank(0.30, 0.25, 0.25) is None


def test_annualized_yield():
    assert annualized_yield(0.01, 36) == pytest.approx(0.01 * 365 / 36)
    assert annualized_yield(0.01, 0) is None
