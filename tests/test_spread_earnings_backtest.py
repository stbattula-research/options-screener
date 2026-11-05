"""Tests for the earnings-aware spread backtest (v11).

All tests use synthetic price series and synthetic calendars — no
network. They verify the gate mechanics (same inclusive window rule as
the live v9 gate, applied to spread entries), the entry-cadence shift
after a skip, the config gating, and the report rendering — not any
market outcome.
"""

import json
from datetime import date, timedelta

import pytest

from options_screener.backtest import (build_spread_earnings_payload,
                                       render_spread_earnings_markdown,
                                       simulate_spreads,
                                       simulate_spreads_earnings_aware,
                                       summarize_spreads)
from tests.test_backtest import make_cfg as base_cfg


def make_cfg(**overrides):
    cfg = base_cfg()
    cfg["spread"] = {"min_width_pct": 0.02, "max_width_pct": 0.06,
                     "min_credit_fraction": 0.33}
    cfg["earnings"] = {"enabled": True, "exclude": True}
    for section, vals in overrides.items():
        cfg.setdefault(section, {}).update(vals)
    return cfg


def cal_dates(n, start=date(2025, 1, 1)):
    """One calendar date per price index (test calendars, not trading days)."""
    return [start + timedelta(days=i) for i in range(n)]


def test_gate_skips_first_spread_entry():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    # First plain spread entry: day 20 -> 2025-01-21, window to 2025-02-25.
    # Earnings 2025-01-22 lands inside only that window.
    res = simulate_spreads_earnings_aware(prices, cfg, 2.0, 0.02, dates,
                                           [date(2025, 1, 22)])
    assert res["gate_active"] is True
    assert len(res["skipped"]) == 1
    assert res["skipped"][0]["entry_day"] == 20
    assert res["skipped"][0]["earnings_date"] == "2025-01-22"
    # No position opened, so the next attempt is day 20 + interval (25).
    assert res["trades"][0]["entry_day"] == 25
    # Filtered summary is a spread summary, not a plain one.
    assert "avg_credit_fraction" in res["summary"]
    assert "credit_rule_pass_rate" in res["summary"]


def test_gate_window_is_inclusive_on_both_ends():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    # Entry day 20: window 2025-01-21 .. 2025-02-25 (35 calendar days).
    for earn in (date(2025, 1, 21), date(2025, 2, 25)):
        res = simulate_spreads_earnings_aware(prices, cfg, 2.0, 0.02,
                                               dates, [earn])
        assert any(s["entry_day"] == 20 for s in res["skipped"]), earn
    # One day past either end: no skip.
    for earn in (date(2025, 1, 20), date(2025, 2, 26)):
        res = simulate_spreads_earnings_aware(prices, cfg, 2.0, 0.02,
                                               dates, [earn])
        assert not any(s["entry_day"] == 20 for s in res["skipped"]), earn


def test_no_earnings_dates_matches_plain_spread_run():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    plain = simulate_spreads(prices, cfg, 2.0, 0.02)
    filt = simulate_spreads_earnings_aware(prices, cfg, 2.0, 0.02, dates, [])
    assert filt["skipped"] == []
    assert [t["entry_day"] for t in filt["trades"]] == [
        t["entry_day"] for t in plain]
    assert filt["summary"] == summarize_spreads(plain)


def test_gate_inactive_matches_plain_spread_run():
    cfg = make_cfg(earnings={"enabled": False, "exclude": True})
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    plain = simulate_spreads(prices, cfg, 2.0, 0.02)
    filt = simulate_spreads_earnings_aware(prices, cfg, 2.0, 0.02, dates,
                                            [date(2025, 1, 22)])
    assert filt["gate_active"] is False
    assert filt["skipped"] == []
    assert [t["entry_day"] for t in filt["trades"]] == [
        t["entry_day"] for t in plain]


def test_misaligned_entry_dates_raise():
    cfg = make_cfg()
    prices = [100.0] * 300
    with pytest.raises(ValueError):
        simulate_spreads_earnings_aware(prices, cfg, 2.0, 0.02,
                                         cal_dates(10), [])


def test_empty_prices_returns_empty_summary():
    cfg = make_cfg()
    res = simulate_spreads_earnings_aware([], cfg, 2.0, 0.02, [], [])
    assert res["trades"] == [] and res["skipped"] == []
    assert res["summary"]["trades"] == 0
    assert res["summary"]["avg_credit_fraction"] is None


def test_payload_and_markdown_render():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    skews = [0.0, 2.0]
    per_skew = {}
    for skew in skews:
        plain = simulate_spreads(prices, cfg, skew, 0.02)
        filt = simulate_spreads_earnings_aware(prices, cfg, skew, 0.02,
                                                dates, [date(2025, 1, 22)])
        per_skew[skew] = {"plain": {"trades": plain,
                                    "summary": summarize_spreads(plain)},
                          "filtered": filt}
    results = {"TEST": {"skews": per_skew, "calendar_dates": 1}}
    payload = build_spread_earnings_payload(results, cfg, skews, 0.02)
    json.dumps(payload)  # must be JSON-serializable
    md = render_spread_earnings_markdown(payload)
    assert "Earnings-Aware Spread Backtest" in md
    assert "2025-01-22" in md  # skipped-entry list names the date
    assert "active" in md
    # Filtered run at skew 2.0 skipped the first entry, so it has fewer
    # trades than the plain run on this flat series.
    assert payload["tickers"]["TEST"]["skews"][2.0]["filtered"]["summary"]["trades"] < \
        payload["tickers"]["TEST"]["skews"][2.0]["plain"]["summary"]["trades"]


def test_payload_flags_no_calendar_ticker():
    cfg = make_cfg()
    prices = [100.0] * 300
    dates = cal_dates(len(prices))
    plain = simulate_spreads(prices, cfg, 2.0, 0.02)
    filt = simulate_spreads_earnings_aware(prices, cfg, 2.0, 0.02, dates, [])
    results = {"TEST": {"skews": {2.0: {"plain": {"trades": plain,
                                                  "summary": summarize_spreads(plain)},
                                        "filtered": filt}},
                        "calendar_dates": 0}}
    md = render_spread_earnings_markdown(
        build_spread_earnings_payload(results, cfg, [2.0], 0.02))
    assert "(no calendar)" in md
