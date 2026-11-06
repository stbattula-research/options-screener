"""Tests for the v12 dividend / ex-date awareness."""

from datetime import date, datetime, timezone

import pytest

from options_screener.dividends import (apply_dividend_gate,
                                        dividend_status, parse_dividend_date)
from options_screener.report import _dividend_cell, render_markdown

AS_OF = date(2026, 11, 6)
EXPIRY = "2026-12-11"  # 35 DTE from AS_OF


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


# Default config: flag, don't exclude — the v12 design choice.
CFG = {"dividends": {"enabled": True, "exclude": False}}
CFG_EXCLUDE = {"dividends": {"enabled": True, "exclude": True}}


def epoch_for(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


# --- parse_dividend_date ---

def test_parse_date_passthrough():
    assert parse_dividend_date(date(2026, 11, 20)) == date(2026, 11, 20)


def test_parse_datetime_truncates():
    assert parse_dividend_date(datetime(2026, 11, 20, 9, 30)) == date(2026, 11, 20)


def test_parse_timestamp_like():
    assert parse_dividend_date(FakeTimestamp(date(2026, 11, 20))) == date(2026, 11, 20)


def test_parse_string():
    assert parse_dividend_date("2026-11-20") == date(2026, 11, 20)


def test_parse_string_with_time_part():
    assert parse_dividend_date("2026-11-20 09:30:00") == date(2026, 11, 20)


def test_parse_epoch_seconds():
    assert parse_dividend_date(epoch_for(date(2026, 11, 20))) == date(2026, 11, 20)
    assert parse_dividend_date(float(epoch_for(date(2026, 11, 20)))) == date(2026, 11, 20)


def test_parse_garbage_returns_none():
    assert parse_dividend_date(None) is None
    assert parse_dividend_date("not-a-date") is None
    assert parse_dividend_date(["2026-11-20"]) is None
    assert parse_dividend_date(42) is None  # too small to be an epoch
    assert parse_dividend_date(True) is None  # bool is not an epoch
    assert parse_dividend_date(-1) is None


# --- dividend_status window boundaries (inclusive both ends) ---

def test_inside_window():
    assert dividend_status(date(2026, 11, 20), AS_OF, date(2026, 12, 11))[0] == "inside"


def test_on_run_date_is_inside():
    assert dividend_status(AS_OF, AS_OF, date(2026, 12, 11))[0] == "inside"


def test_on_expiry_is_inside():
    assert dividend_status(date(2026, 12, 11), AS_OF, date(2026, 12, 11))[0] == "inside"


def test_before_window_is_outside():
    assert dividend_status(date(2026, 11, 5), AS_OF, date(2026, 12, 11))[0] == "outside"


def test_after_expiry_is_outside():
    assert dividend_status(date(2026, 12, 12), AS_OF, date(2026, 12, 11))[0] == "outside"


def test_unknown_when_no_date():
    status, d = dividend_status(None, AS_OF, date(2026, 12, 11))
    assert status == "unknown" and d is None


# --- apply_dividend_gate ---

def test_inside_flagged_not_excluded_by_default():
    s = sig()
    gate = apply_dividend_gate(s, date(2026, 11, 20), AS_OF, CFG)
    assert gate["excluded"] is False
    assert gate["reason"] is None
    assert s["dividends"] == {"status": "inside", "date": "2026-11-20"}


def test_inside_excluded_when_exclude_on():
    s = sig()
    gate = apply_dividend_gate(s, date(2026, 11, 20), AS_OF, CFG_EXCLUDE)
    assert gate["excluded"] is True
    assert "2026-11-20" in gate["reason"]
    assert s["dividends"] == {"status": "inside", "date": "2026-11-20"}


def test_outside_never_excluded_even_with_exclude_on():
    s = sig()
    gate = apply_dividend_gate(s, date(2026, 12, 20), AS_OF, CFG_EXCLUDE)
    assert gate["excluded"] is False
    assert s["dividends"] == {"status": "outside", "date": "2026-12-20"}


def test_unknown_never_excluded_but_flagged():
    s = sig()
    gate = apply_dividend_gate(s, None, AS_OF, CFG_EXCLUDE)
    assert gate["excluded"] is False
    assert s["dividends"] == {"status": "unknown", "date": None}


def test_disabled_feature_leaves_no_flag():
    s = sig()
    gate = apply_dividend_gate(s, date(2026, 11, 20), AS_OF,
                               {"dividends": {"enabled": False}})
    assert gate["excluded"] is False
    assert s["dividends"]["status"] == "disabled"


def test_unparseable_expiry_is_unknown_not_crash():
    s = sig(expiry="garbage")
    gate = apply_dividend_gate(s, date(2026, 11, 20), AS_OF, CFG_EXCLUDE)
    assert gate["excluded"] is False
    assert s["dividends"]["status"] == "unknown"


# --- report rendering ---

def test_dividend_cells():
    assert _dividend_cell({"dividends": {"status": "unknown", "date": None}}) == "unknown"
    assert _dividend_cell({"dividends": {"status": "outside", "date": "2026-12-20"}}) == "2026-12-20"
    assert _dividend_cell({"dividends": {"status": "inside", "date": "2026-11-20"}}) == "2026-11-20 ⚠"
    assert _dividend_cell({"dividends": {"status": "disabled", "date": None}}) == "—"
    assert _dividend_cell({}) == "unknown"


def test_signals_table_has_dividend_column():
    payload = {
        "as_of": "2026-11-06",
        "strategy": "test",
        "data_source": "test",
        "disclaimer": "test",
        "signals": [{
            "ticker": "AAPL", "underlying": 250.0, "strike": 240.0,
            "expiry": EXPIRY, "dte": 35, "delta": -0.30, "iv": 0.25,
            "premium_pct": 0.012, "annualized_yield": 0.15,
            "bid": 2.80, "ask": 3.00, "open_interest": 1200,
            "breakeven": 237.10,
            "earnings": {"status": "unknown", "date": None},
            "dividends": {"status": "inside", "date": "2026-11-20"},
            "exit_plan": {"profit_target": "x", "time_exit": "y",
                          "loss_manage": "z"},
        }],
        "spreads": [], "paper": None, "skipped": [], "errors": [],
    }
    md = render_markdown(payload)
    assert "| Dividend |" in md
    assert "2026-11-20 ⚠" in md
    assert "ex-dividend date" in md.lower() or "Dividend flag" in md
