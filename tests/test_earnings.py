"""Tests for the v9 earnings-aware filter."""

from datetime import date, datetime

import pytest

from options_screener.earnings import (apply_earnings_gate,
                                      earnings_status, parse_earnings_date)
from options_screener.report import _earnings_cell, render_markdown

AS_OF = date(2026, 10, 31)
EXPIRY = "2026-12-04"  # 34 DTE from AS_OF


class FakeTimestamp:
    """Pandas Timestamp stand-in: only .date() matters to the parser."""

    def __init__(self, d):
        self._d = d

    def date(self):
        return self._d


def sig(**over):
    base = {"ticker": "AAPL", "expiry": EXPIRY}
    base.update(over)
    return base


CFG = {"earnings": {"enabled": True, "exclude": True}}


# --- parse_earnings_date ---

def test_parse_date_passthrough():
    assert parse_earnings_date(date(2026, 11, 5)) == date(2026, 11, 5)


def test_parse_datetime_truncates():
    assert parse_earnings_date(datetime(2026, 11, 5, 21, 30)) == date(2026, 11, 5)


def test_parse_timestamp_like():
    assert parse_earnings_date(FakeTimestamp(date(2026, 11, 5))) == date(2026, 11, 5)


def test_parse_string():
    assert parse_earnings_date("2026-11-05") == date(2026, 11, 5)


def test_parse_string_with_time_part():
    assert parse_earnings_date("2026-11-05 21:30:00") == date(2026, 11, 5)


def test_parse_garbage_returns_none():
    assert parse_earnings_date(None) is None
    assert parse_earnings_date("not-a-date") is None
    assert parse_earnings_date(["2026-11-05"]) is None
    assert parse_earnings_date(20261105) is None


# --- earnings_status window boundaries (inclusive both ends) ---

def test_inside_window():
    assert earnings_status(date(2026, 11, 20), AS_OF, date(2026, 12, 4))[0] == "inside"


def test_on_run_date_is_inside():
    assert earnings_status(AS_OF, AS_OF, date(2026, 12, 4))[0] == "inside"


def test_on_expiry_is_inside():
    assert earnings_status(date(2026, 12, 4), AS_OF, date(2026, 12, 4))[0] == "inside"


def test_before_window_is_outside():
    assert earnings_status(date(2026, 10, 30), AS_OF, date(2026, 12, 4))[0] == "outside"


def test_after_expiry_is_outside():
    assert earnings_status(date(2026, 12, 5), AS_OF, date(2026, 12, 4))[0] == "outside"


def test_unknown_when_no_date():
    status, d = earnings_status(None, AS_OF, date(2026, 12, 4))
    assert status == "unknown" and d is None


# --- apply_earnings_gate ---

def test_inside_excluded_with_reason():
    s = sig()
    gate = apply_earnings_gate(s, date(2026, 11, 20), AS_OF, CFG)
    assert gate["excluded"] is True
    assert "2026-11-20" in gate["reason"]
    assert s["earnings"] == {"status": "inside", "date": "2026-11-20"}


def test_inside_kept_when_exclude_off():
    s = sig()
    cfg = {"earnings": {"enabled": True, "exclude": False}}
    gate = apply_earnings_gate(s, date(2026, 11, 20), AS_OF, cfg)
    assert gate["excluded"] is False
    assert s["earnings"]["status"] == "inside"


def test_outside_never_excluded():
    s = sig()
    gate = apply_earnings_gate(s, date(2026, 12, 20), AS_OF, CFG)
    assert gate["excluded"] is False
    assert s["earnings"] == {"status": "outside", "date": "2026-12-20"}


def test_unknown_never_excluded_but_flagged():
    s = sig()
    gate = apply_earnings_gate(s, None, AS_OF, CFG)
    assert gate["excluded"] is False
    assert s["earnings"] == {"status": "unknown", "date": None}


def test_disabled_feature_leaves_no_flag():
    s = sig()
    gate = apply_earnings_gate(s, date(2026, 11, 20), AS_OF,
                               {"earnings": {"enabled": False}})
    assert gate["excluded"] is False
    assert s["earnings"]["status"] == "disabled"


def test_unparseable_expiry_is_unknown_not_crash():
    s = sig(expiry="garbage")
    gate = apply_earnings_gate(s, date(2026, 11, 20), AS_OF, CFG)
    assert gate["excluded"] is False
    assert s["earnings"]["status"] == "unknown"


# --- report rendering ---

def test_earnings_cells():
    assert _earnings_cell({"earnings": {"status": "unknown", "date": None}}) == "unknown"
    assert _earnings_cell({"earnings": {"status": "outside", "date": "2026-12-20"}}) == "2026-12-20"
    assert _earnings_cell({"earnings": {"status": "inside", "date": "2026-11-20"}}) == "2026-11-20 ⚠"
    assert _earnings_cell({"earnings": {"status": "disabled", "date": None}}) == "—"
    assert _earnings_cell({}) == "unknown"


def test_signals_table_has_earnings_column():
    payload = {
        "as_of": "2026-10-31",
        "strategy": "test",
        "data_source": "test",
        "disclaimer": "test",
        "signals": [{
            "ticker": "AAPL", "underlying": 250.0, "strike": 240.0,
            "expiry": EXPIRY, "dte": 34, "delta": -0.30, "iv": 0.25,
            "premium_pct": 0.012, "annualized_yield": 0.15,
            "bid": 2.80, "ask": 3.00, "open_interest": 1200,
            "breakeven": 237.10,
            "earnings": {"status": "unknown", "date": None},
            "exit_plan": {"profit_target": "x", "time_exit": "y",
                          "loss_manage": "z"},
        }],
        "spreads": [], "paper": None, "skipped": [], "errors": [],
    }
    md = render_markdown(payload)
    assert "| Earnings |" in md
    assert "| unknown |" in md
