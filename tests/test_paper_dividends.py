"""Tests for dividend-aware paper-position marking (v21).

The live v12 flag only checks the ex-date at signal time. A paper
position lives 30-45 days, and an ex-date can be announced after it
opens (or was unknown at entry) — so every mark re-checks the ticker's
next ex-dividend date against the position's REMAINING window [today,
expiry]. The flag is informational only: it never changes the
mechanical exits. The dividend twin of v13's earnings check.
Network-independent.
"""

import os
from datetime import date, timedelta

import yaml

from options_screener.exits import (
    attach_dividend_exposure,
    open_from_signal,
    open_from_spread,
    update_positions,
)
from options_screener.report import render_paper_section

EXITS = {"profit_target_pct": 0.50, "manage_dte": 21,
         "loss_manage_multiple": 2.0}

TODAY = date.today()


def _expiry(days=30):
    return (TODAY + timedelta(days=days)).isoformat()


def _as_of(days=0):
    return (TODAY + timedelta(days=days)).isoformat()


def csp_signal(**kw):
    sig = {"ticker": "AAPL", "strike": 280.0, "expiry": _expiry(),
           "dte": 30, "delta": -0.29, "mid": 4.00, "underlying": 295.0,
           "breakeven": 276.0}
    sig.update(kw)
    return sig


def bull_spread(**kw):
    sp = {"ticker": "AAPL", "short_strike": 280.0, "long_strike": 270.0,
          "expiry": _expiry(), "dte": 30, "net_credit": 3.00,
          "max_loss": 7.00, "underlying": 295.0, "breakeven": 277.0}
    sp.update(kw)
    return sp


def _snap(mark, dte=30, underlying=295.0):
    return {"date": _as_of(), "underlying": underlying,
            "mark": mark, "dte": dte}


# --- attach_dividend_exposure ---------------------------------------------

def test_inside_window_flags_exposed():
    p = open_from_signal(csp_signal())
    got = attach_dividend_exposure(p, TODAY + timedelta(days=10), _as_of())
    assert got == {"status": "inside",
                   "date": (TODAY + timedelta(days=10)).isoformat()}
    assert p["dividends"] == got


def test_outside_window_stays_outside():
    p = open_from_signal(csp_signal())
    got = attach_dividend_exposure(p, TODAY + timedelta(days=40), _as_of())
    assert got["status"] == "outside"
    assert got["date"] == (TODAY + timedelta(days=40)).isoformat()


def test_unknown_ex_date_stays_unknown():
    p = open_from_signal(csp_signal())
    got = attach_dividend_exposure(p, None, _as_of())
    assert got == {"status": "unknown", "date": None}


def test_window_endpoints_are_inclusive():
    p = open_from_signal(csp_signal())
    # Ex-date today counts; ex-date on expiry day counts (v12 rule).
    assert attach_dividend_exposure(p, TODAY, _as_of())["status"] == "inside"
    assert attach_dividend_exposure(
        p, TODAY + timedelta(days=30), _as_of())["status"] == "inside"


def test_bad_expiry_or_date_never_raises():
    p = {"ticker": "AAPL", "expiry": "not-a-date", "key": "x"}
    got = attach_dividend_exposure(p, TODAY, _as_of())
    assert got == {"status": "unknown", "date": None}
    p2 = open_from_signal(csp_signal())
    got2 = attach_dividend_exposure(p2, TODAY + timedelta(days=5),
                                    "garbage-date")
    assert got2 == {"status": "unknown", "date": None}


def test_as_of_accepts_date_object():
    p = open_from_signal(csp_signal())
    got = attach_dividend_exposure(p, TODAY + timedelta(days=5), TODAY)
    assert got["status"] == "inside"


# --- positions carry entry-time dividend info ------------------------------

def test_open_from_signal_carries_entry_dividends():
    sig = csp_signal(dividends={"status": "outside",
                                "date": (TODAY + timedelta(days=40)).isoformat()})
    p = open_from_signal(sig, opened=_as_of())
    assert p["dividends"] == sig["dividends"]
    # And a copy, not a shared reference.
    p["dividends"]["status"] = "inside"
    assert sig["dividends"]["status"] == "outside"


def test_open_from_signal_without_dividends_has_no_key():
    assert "dividends" not in open_from_signal(csp_signal())


def test_open_from_spread_carries_entry_dividends():
    sp = bull_spread(dividends={"status": "inside",
                                "date": (TODAY + timedelta(days=5)).isoformat()})
    p = open_from_spread(sp, opened=_as_of())
    assert p["dividends"]["status"] == "inside"


# --- update_positions refreshes exposure at each mark ----------------------

def _store_with_open(pos):
    return {"open": [pos], "closed": []}


def test_mark_refreshes_exposure_when_date_appears():
    # Opened when the ex-date was unknown; the mark discovers a date
    # inside the remaining window.
    pos = open_from_signal(csp_signal(), opened=_as_of())
    pos["dividends"] = {"status": "unknown", "date": None}
    store = _store_with_open(pos)
    result = update_positions(
        store, {pos["key"]: _snap(mark=3.50)}, EXITS, as_of=_as_of(),
        dividends_map={"AAPL": TODAY + timedelta(days=12)})
    assert store["open"][0]["dividends"]["status"] == "inside"
    assert result["updated"][0]["dividends"]["status"] == "inside"
    # Informational only: the position still holds.
    assert result["updated"][0]["action"] == "hold"
    assert result["closed_today"] == []


def test_mark_clears_exposure_when_date_moves_outside():
    # The ex-date was inside at entry; the date has now passed (or the
    # calendar moved it) — the refresh must say so, not keep stale info.
    pos = open_from_signal(csp_signal(), opened=_as_of())
    pos["dividends"] = {"status": "inside",
                        "date": (TODAY + timedelta(days=5)).isoformat()}
    store = _store_with_open(pos)
    update_positions(store, {pos["key"]: _snap(mark=3.50)}, EXITS,
                     as_of=_as_of(days=10),
                     dividends_map={"AAPL": TODAY + timedelta(days=40)})
    assert store["open"][0]["dividends"]["status"] == "outside"


def test_no_map_preserves_existing_exposure():
    pos = open_from_signal(csp_signal(), opened=_as_of())
    pos["dividends"] = {"status": "inside",
                        "date": (TODAY + timedelta(days=5)).isoformat()}
    store = _store_with_open(pos)
    update_positions(store, {pos["key"]: _snap(mark=3.50)}, EXITS,
                     as_of=_as_of())
    assert store["open"][0]["dividends"]["status"] == "inside"


def test_ticker_missing_from_map_keeps_existing_exposure():
    pos = open_from_signal(csp_signal(), opened=_as_of())
    pos["dividends"] = {"status": "unknown", "date": None}
    store = _store_with_open(pos)
    update_positions(store, {pos["key"]: _snap(mark=3.50)}, EXITS,
                     as_of=_as_of(), dividends_map={"MSFT": TODAY})
    assert store["open"][0]["dividends"] == {"status": "unknown",
                                             "date": None}


def test_closed_position_carries_exposure():
    # A position closed by the profit rule still records whether an
    # ex-date was inside its remaining window at the last mark.
    pos = open_from_signal(csp_signal(), opened=_as_of())
    store = _store_with_open(pos)
    result = update_positions(
        store, {pos["key"]: _snap(mark=1.50)}, EXITS, as_of=_as_of(),
        dividends_map={"AAPL": TODAY + timedelta(days=7)})
    assert result["closed_today"][0]["exit_reason"] == "profit"
    assert result["closed_today"][0]["dividends"]["status"] == "inside"


def test_earnings_and_dividends_refresh_together():
    # The two maps are independent: one ticker can be inside on
    # earnings and outside on dividends at the same mark.
    pos = open_from_signal(csp_signal(), opened=_as_of())
    store = _store_with_open(pos)
    result = update_positions(
        store, {pos["key"]: _snap(mark=3.50)}, EXITS, as_of=_as_of(),
        earnings_map={"AAPL": TODAY + timedelta(days=7)},
        dividends_map={"AAPL": TODAY + timedelta(days=40)})
    assert result["updated"][0]["earnings"]["status"] == "inside"
    assert result["updated"][0]["dividends"]["status"] == "outside"


# --- report rendering ------------------------------------------------------

def _paper_payload(open_dividends, closed_dividends=None):
    pos = open_from_signal(csp_signal(), opened=_as_of())
    pos.update({"last_mark": 3.50, "last_dte": 28,
                "last_unrealized": 50.0, "days_held": 2})
    if open_dividends is not None:
        pos["dividends"] = open_dividends
    closed = []
    if closed_dividends is not None:
        c = open_from_signal(csp_signal(), opened=_as_of())
        c.update({"status": "closed", "closed_date": _as_of(),
                  "exit_reason": "profit", "realized": 200.0})
        c["dividends"] = closed_dividends
        closed = [c]
    return {"opened": [], "open": [pos], "closed_today": closed,
            "state_dir": "positions"}


def test_open_table_flags_inside_ex_date():
    lines = render_paper_section(_paper_payload(
        {"status": "inside", "date": (TODAY + timedelta(days=7)).isoformat()}))
    row = [ln for ln in lines if "AAPL" in ln and "$280" in ln][0]
    assert (TODAY + timedelta(days=7)).isoformat() in row
    assert "⚠" in row


def test_open_table_shows_unknown_and_never_checked():
    unknown = render_paper_section(
        _paper_payload({"status": "unknown", "date": None}))
    assert "unknown" in [ln for ln in unknown if "AAPL" in ln][0]
    unchecked = render_paper_section(_paper_payload(None))
    row = [ln for ln in unchecked if "AAPL" in ln][0]
    assert row.rstrip().endswith("| — |")


def test_closed_table_carries_dividend_cell():
    lines = render_paper_section(_paper_payload(
        {"status": "outside", "date": (TODAY + timedelta(days=40)).isoformat()},
        closed_dividends={"status": "inside",
                          "date": (TODAY + timedelta(days=3)).isoformat()}))
    closed_row = [ln for ln in lines
                  if ln.startswith("| AAPL") and "profit" in ln][0]
    assert (TODAY + timedelta(days=3)).isoformat() in closed_row
    assert "⚠" in closed_row
    # The section states the flag is informational only.
    assert any("informational only" in ln for ln in lines)


# --- config ----------------------------------------------------------------

def test_config_enables_dividend_check_by_default():
    path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["paper"]["dividend_check"] is True
