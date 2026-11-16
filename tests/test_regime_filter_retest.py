"""Tests for the v17 regime-filter re-test window (research only).

`slice_window` and the payload/report window handling are pure and
tested here with synthetic business-day series — no network. The
thin `get_closes_window` wrapper (yfinance fetch + slice) is covered
by the live smoke run, not by unit tests.
"""

import json
import os
from datetime import date, timedelta

import yaml

from options_screener.backtest import (DEFAULT_REGIME_FILTER,
                                       build_regime_filter_payload,
                                       render_regime_filter_markdown)
from options_screener.data import DataError, slice_window

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
TODAY = date(2026, 11, 16)


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def business_days(n, start=date(2020, 1, 6)):
    """`n` weekday dates starting on a Monday, oldest first."""
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def series(n=1800):
    """Distinct closes over business days, so window contents are checkable."""
    dates = business_days(n)
    closes = [100.0 + 0.01 * i for i in range(n)]
    return closes, dates


def _plain_filtered(plain_pnls, filt_pnls, skipped=3):
    from options_screener.backtest import summarize

    def trades(pnls):
        return [{"pnl_per_contract": p, "return_on_risk": p / 10000.0,
                 "days_held": 10,
                 "exit_reason": "profit" if p > 0 else "loss"}
                for p in pnls]
    return {
        "plain": {"trades": trades(plain_pnls),
                  "summary": summarize(trades(plain_pnls))},
        "filtered": {"trades": trades(filt_pnls),
                     "summary": summarize(trades(filt_pnls)),
                     "skipped": [{}] * skipped},
    }


# ------------------------------------------------------------- slice_window

def test_slice_window_picks_the_calendar_span_before_cutoff():
    closes, dates = series()
    window, start, end = slice_window(closes, dates, 2, 2, TODAY)
    cutoff = TODAY - timedelta(days=365 * 2)
    span_start = cutoff - timedelta(days=365 * 2)
    # every close in the span, no close outside it
    want = [c for c, d in zip(closes, dates) if span_start < d <= cutoff]
    want_dates = [d for d in dates if span_start < d <= cutoff]
    assert window == want
    assert start == want_dates[0] and end == want_dates[-1]
    assert span_start < start <= cutoff
    assert span_start < end <= cutoff
    # a real 2-year span holds ~500-506 trading days, not exactly 504:
    # the synthetic series has no market holidays, so it is at the top
    # of that range — the point is the count comes from the calendar,
    # not from a fixed 252/year assumption
    assert 500 <= len(window) <= 530


def test_slice_window_offset_zero_is_the_recent_window():
    closes, dates = series()
    window, start, end = slice_window(closes, dates, 2, 0, TODAY)
    span_start = TODAY - timedelta(days=365 * 2)
    want = [c for c, d in zip(closes, dates) if span_start < d <= TODAY]
    assert window == want
    assert end <= TODAY
    assert start > span_start


def test_slice_window_rejects_too_short_history():
    closes, dates = series(n=400)
    try:
        slice_window(closes, dates, 2, 2, TODAY)
    except DataError as exc:
        assert "need at least 480" in str(exc)
    else:
        raise AssertionError("too-short history should raise DataError")


def test_slice_window_rejects_bad_args():
    closes, dates = series()
    for years, offset in ((0, 0), (2, -1)):
        try:
            slice_window(closes, dates, years, offset, TODAY)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"years={years} offset={offset} should raise ValueError")
    try:
        slice_window(closes, dates[:-1], 2, 0, TODAY)
    except DataError:
        pass
    else:
        raise AssertionError("mismatched closes/dates should raise DataError")


def test_slice_window_same_shape_as_plain_run():
    # The re-test window has about the same length as the plain
    # --years run it is compared against (real years differ by a few
    # trading days, so the lengths match within days, not exactly),
    # and the two windows never share a close: the re-test is a
    # different period, not a re-read of the same days.
    closes, dates = series()
    recent, _, _ = slice_window(closes, dates, 2, 0, TODAY)
    earlier, _, _ = slice_window(closes, dates, 2, 2, TODAY)
    assert abs(len(recent) - len(earlier)) <= 5
    assert not set(earlier) & set(recent)


# ------------------------------------------------------- payload + markdown

def test_payload_with_window_records_it_and_says_retest():
    cfg = load_cfg()
    results = {"AAA": _plain_filtered([100] * 12, [200] * 6)}
    window = {"start": "2022-11-15", "end": "2024-11-15",
              "years": 2, "offset_years": 2}
    payload = build_regime_filter_payload(
        results, cfg, DEFAULT_REGIME_FILTER["lookback_days"],
        window=window)
    assert payload["window"] == window
    json.dumps(payload)  # must not raise
    md = render_regime_filter_markdown(payload)
    assert "2022-11-15" in md and "2024-11-15" in md
    assert "re-test" in md
    assert "This run IS that re-test" in " ".join(payload["assumptions"])


def test_payload_without_window_is_unchanged():
    cfg = load_cfg()
    results = {"AAA": _plain_filtered([100] * 12, [200] * 6)}
    payload = build_regime_filter_payload(
        results, cfg, DEFAULT_REGIME_FILTER["lookback_days"])
    assert "window" not in payload
    md = render_regime_filter_markdown(payload)
    assert "**Window:**" not in md
    assert "Re-test on a different period" in " ".join(
        payload["assumptions"])
