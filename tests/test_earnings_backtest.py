"""Tests for the earnings-aware backtest filter (v10).

All tests use synthetic price series and synthetic calendars — no
network. They verify the gate mechanics (same inclusive window rule as
the live v9 gate), the entry-cadence shift after a skip, the config
gating, and the report rendering — not any market outcome.
"""

import json
from datetime import date, timedelta

import pytest

from options_screener.backtest import (build_earnings_payload,
                                       load_earnings_calendar,
                                       make_earnings_gate,
                                       render_earnings_markdown,
                                       simulate,
                                       simulate_earnings_aware)
from tests.test_backtest import make_cfg


def cal_dates(n, start=date(2025, 1, 1)):
    """One calendar date per price index (test calendars, not trading days)."""
    return [start + timedelta(days=i) for i in range(n)]


def test_gate_skips_entry_with_inside_earnings():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    # First plain entry: day 20 -> 2025-01-21, window to 2025-02-25.
    # Earnings 2025-01-22 lands inside only that window.
    res = simulate_earnings_aware(prices, cfg, dates, [date(2025, 1, 22)])
    assert res["gate_active"] is True
    assert len(res["skipped"]) == 1
    assert res["skipped"][0]["entry_day"] == 20
    assert res["skipped"][0]["earnings_date"] == "2025-01-22"
    # No position opened, so the next attempt is day 20 + interval (25).
    assert res["trades"][0]["entry_day"] == 25


def test_gate_window_is_inclusive_on_both_ends():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    # Entry day 20: window 2025-01-21 .. 2025-02-25 (35 calendar days).
    for earn in (date(2025, 1, 21), date(2025, 2, 25)):
        res = simulate_earnings_aware(prices, cfg, dates, [earn])
        assert any(s["entry_day"] == 20 for s in res["skipped"]), earn
    # One day past either end: no skip.
    for earn in (date(2025, 1, 20), date(2025, 2, 26)):
        res = simulate_earnings_aware(prices, cfg, dates, [earn])
        assert not any(s["entry_day"] == 20 for s in res["skipped"]), earn


def test_no_earnings_dates_matches_plain_run():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    filtered = simulate_earnings_aware(prices, cfg, dates, [])
    assert filtered["skipped"] == []
    assert filtered["trades"] == simulate(prices, cfg)


def test_gate_inactive_when_exclude_or_enabled_false():
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    earn = [date(2025, 1, 22)]
    for ecfg in ({"enabled": False, "exclude": True},
                 {"enabled": True, "exclude": False}):
        cfg = make_cfg()
        cfg["earnings"] = ecfg
        res = simulate_earnings_aware(prices, cfg, dates, earn)
        assert res["gate_active"] is False
        assert res["skipped"] == []
        assert res["trades"] == simulate(prices, cfg)


def test_misaligned_entry_dates_raise():
    cfg = make_cfg()
    prices = [100.0] * 300
    with pytest.raises(ValueError):
        simulate_earnings_aware(prices, cfg, cal_dates(100), [])
    with pytest.raises(ValueError):
        make_earnings_gate([], [])


def test_multiple_skips_all_recorded():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    # Earnings roughly quarterly. A 35-day window with 5-day entry
    # spacing means ONE earnings date blocks several successive attempts
    # (the window slides 5 days at a time past it) — so the count is
    # larger than the number of earnings dates, and every earnings date
    # must appear among the skips.
    earns = [date(2025, 1, 22), date(2025, 4, 15), date(2025, 7, 10)]
    res = simulate_earnings_aware(prices, cfg, dates, earns)
    assert set(s["earnings_date"] for s in res["skipped"]) == {
        "2025-01-22", "2025-04-15", "2025-07-10"}
    assert len(res["skipped"]) > len(earns)
    # And no surviving trade's window contains an earnings date.
    for t in res["trades"]:
        start = dates[t["entry_day"]]
        end = start + timedelta(days=t["dte"])
        assert not any(start <= e <= end for e in earns)
    assert res["summary"]["trades"] < len(simulate(prices, cfg))


def test_load_earnings_calendar_drops_garbage():
    raw = {"AAPL": ["2025-02-20", "not-a-date", None, 42],
           "SPY": []}
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as f:
        json.dump(raw, f)
        path = f.name
    try:
        cal = load_earnings_calendar(path)
    finally:
        os.unlink(path)
    assert cal["AAPL"] == [date(2025, 2, 20)]
    assert cal["SPY"] == []
    with pytest.raises(ValueError):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump(["not", "an", "object"], f)
            bad = f.name
        try:
            load_earnings_calendar(bad)
        finally:
            os.unlink(bad)


def test_earnings_payload_is_json_serializable_and_renders():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    plain = {"trades": simulate(prices, cfg),
             "summary": None}
    from options_screener.backtest import summarize
    plain["summary"] = summarize(plain["trades"])
    filt = simulate_earnings_aware(prices, cfg, dates, [date(2025, 1, 22)])
    results = {"AAPL": {"plain": plain, "filtered": filt,
                        "calendar_dates": 1}}
    payload = build_earnings_payload(results, cfg)
    json.dumps(payload)  # must not raise
    md = render_earnings_markdown(payload)
    assert "2025-01-22" in md
    assert "Skipped entries" in md or "skipped entries" in md
    assert "Earnings-Aware Backtest" in md


def test_payload_flags_ticker_with_no_calendar():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    filt = simulate_earnings_aware(prices, cfg, dates, [])
    from options_screener.backtest import summarize
    results = {"SPY": {"plain": {"trades": simulate(prices, cfg),
                                 "summary": summarize(simulate(prices, cfg))},
                       "filtered": filt, "calendar_dates": 0}}
    md = render_earnings_markdown(build_earnings_payload(results, cfg))
    assert "no calendar" in md
