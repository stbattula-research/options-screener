"""Tests for paper position tracking (network-independent)."""

from datetime import date

import pytest

from options_screener.exits import (
    chain_index,
    csp_key,
    evaluate,
    known_keys,
    load_store,
    open_from_signal,
    open_from_spread,
    open_positions,
    save_store,
    spread_key,
    update_positions,
)


EXITS = {"profit_target_pct": 0.50, "manage_dte": 21,
        "loss_manage_multiple": 2.0}


def csp_signal(**kw):
    sig = {"ticker": "SPY", "strike": 600.0, "expiry": "2026-11-20",
           "dte": 33, "delta": -0.29, "mid": 4.00, "underlying": 612.5,
           "breakeven": 596.0}
    sig.update(kw)
    return sig


def bull_spread(**kw):
    sp = {"ticker": "SPY", "short_strike": 600.0, "long_strike": 585.0,
          "expiry": "2026-11-20", "dte": 33, "net_credit": 3.00,
          "max_loss": 12.00, "underlying": 612.5, "breakeven": 597.0}
    sp.update(kw)
    return sp


def snap(mark, dte=30, underlying=612.5, as_of=None):
    # Default to the run date so same-day evaluations stay date-robust.
    return {"date": as_of or date.today().isoformat(), "underlying": underlying,
            "mark": mark, "dte": dte}


# --- position construction ------------------------------------------------

def test_csp_position_fields():
    p = open_from_signal(csp_signal(), opened="2026-10-18")
    assert p["kind"] == "csp"
    assert p["entry_credit"] == 4.00
    assert p["max_profit"] == 4.00
    assert p["strike"] == 600.0
    assert p["opened"] == "2026-10-18"
    assert p["status"] == "open"


def test_spread_position_fields():
    p = open_from_spread(bull_spread(), opened="2026-10-18")
    assert p["kind"] == "spread"
    assert p["entry_credit"] == 3.00
    assert p["max_loss"] == 12.00
    assert p["key"] == spread_key("SPY", "2026-11-20", 600.0, 585.0)


def test_csp_key_matches_alert_key():
    # Same key the alert dedupe uses: an alert can open at most one position.
    assert csp_key("SPY", "2026-11-20", 600.0) == "SPY|2026-11-20|600.0"


# --- exit rules: cash-secured put -----------------------------------------

def test_profit_exit_at_half_max_profit():
    p = open_from_signal(csp_signal(mid=4.00))
    st = evaluate(p, snap(mark=2.00), EXITS)  # bought back for half -> profit
    assert st["action"] == "close" and st["reason"] == "profit"
    assert st["realized"] == pytest.approx(200.0)  # (4 - 2) x 100


def test_loss_exit_at_2x_premium_loss():
    p = open_from_signal(csp_signal(mid=4.00))
    # loss of 2x premium <-> mark is 3x the credit received
    st = evaluate(p, snap(mark=12.00), EXITS)
    assert st["action"] == "close" and st["reason"] == "loss"
    assert st["realized"] == pytest.approx(-800.0)


def test_time_exit_at_21_dte():
    p = open_from_signal(csp_signal(mid=4.00))
    st = evaluate(p, snap(mark=3.00, dte=21), EXITS)
    assert st["action"] == "close" and st["reason"] == "time"
    assert st["realized"] == pytest.approx(100.0)


def test_hold_when_nothing_triggered():
    p = open_from_signal(csp_signal(mid=4.00))
    st = evaluate(p, snap(mark=3.00, dte=30), EXITS)
    assert st["action"] == "hold" and st["reason"] is None
    assert st["unrealized"] == pytest.approx(100.0)
    assert st["realized"] is None
    assert st["days_held"] == 0


def test_expiry_otm_full_profit():
    p = open_from_signal(csp_signal(mid=4.00))
    st = evaluate(p, snap(mark=0.0, dte=0, underlying=612.5), EXITS)
    assert st["action"] == "close" and st["reason"] == "expiry"
    assert st["realized"] == pytest.approx(400.0)
    assert st.get("assigned") is False


def test_expiry_itm_assignment():
    p = open_from_signal(csp_signal(mid=4.00))
    st = evaluate(p, snap(mark=10.0, dte=0, underlying=590.0), EXITS)
    assert st["action"] == "close" and st["reason"] == "expiry"
    assert st.get("assigned") is True
    assert st["realized"] == pytest.approx((4.0 - 10.0) * 100)


def test_stale_mark_holds_honestly():
    p = open_from_signal(csp_signal(mid=4.00))
    st = evaluate(p, {"date": "2026-10-18", "underlying": 612.5,
                      "mark": None, "dte": 30}, EXITS)
    assert st["action"] == "hold" and st["reason"] == "stale"
    assert st["unrealized"] is None


def test_spread_profit_and_loss():
    p = open_from_spread(bull_spread())
    st = evaluate(p, snap(mark=1.50), EXITS)   # 50% of max profit
    assert st["action"] == "close" and st["reason"] == "profit"
    assert st["realized"] == pytest.approx(150.0)
    st = evaluate(p, snap(mark=9.00), EXITS)   # 3x credit
    assert st["action"] == "close" and st["reason"] == "loss"
    assert st["realized"] == pytest.approx(-600.0)


def test_spread_expiry_bands():
    p = open_from_spread(bull_spread())
    # above short strike: full profit
    st = evaluate(p, snap(mark=0.0, dte=0, underlying=605.0), EXITS)
    assert st["realized"] == pytest.approx(300.0)
    # below long strike: max loss
    st = evaluate(p, snap(mark=15.0, dte=0, underlying=580.0), EXITS)
    assert st["realized"] == pytest.approx(-1200.0)
    # between legs: partial
    st = evaluate(p, snap(mark=5.0, dte=0, underlying=592.0), EXITS)
    assert st["realized"] == pytest.approx((3.0 - 8.0) * 100)


# --- store behavior ---------------------------------------------------------

def test_store_round_trip(tmp_path):
    d = str(tmp_path)
    store = load_store(d)
    assert store == {"open": [], "closed": []}
    store["open"].append(open_from_signal(csp_signal()))
    save_store(d, store)
    reloaded = load_store(d)
    assert len(reloaded["open"]) == 1
    assert reloaded["open"][0]["key"] == "SPY|2026-11-20|600.0"


def test_store_tolerates_corrupt_file(tmp_path):
    d = str(tmp_path)
    with open(f"{d}/positions.json", "w") as f:
        f.write("{not json")
    assert load_store(d) == {"open": [], "closed": []}


def test_no_reopen_of_closed_position():
    store = {"open": [], "closed": [open_from_signal(csp_signal())]}
    store["closed"][0]["status"] = "closed"
    new = open_positions([csp_signal()], store, max_open=10)
    assert new == []  # same alert key must not re-open


def test_max_open_cap_respected():
    store = {"open": [], "closed": []}
    sigs = [csp_signal(strike=600.0 + i) for i in range(5)]
    new = open_positions(sigs, store, max_open=3)
    assert len(new) == 3
    assert len(store["open"]) == 3


def test_update_moves_closed_and_keeps_open():
    store = {"open": [open_from_signal(csp_signal(mid=4.00)),
                      open_from_signal(csp_signal(strike=590.0, mid=3.00))],
             "closed": []}
    # first position hits profit exit; second holds
    marks = {store["open"][0]["key"]: snap(mark=2.00, dte=30),
             store["open"][1]["key"]: snap(mark=2.90, dte=30)}
    res = update_positions(store, marks, EXITS, as_of="2026-10-18")
    assert len(res["closed_today"]) == 1
    assert res["closed_today"][0]["exit_reason"] == "profit"
    assert res["closed_today"][0]["realized"] == pytest.approx(200.0)
    assert len(store["open"]) == 1
    assert len(store["closed"]) == 1
    assert res["updated"][1]["action"] == "hold"


def test_update_with_missing_mark_keeps_position_open():
    store = {"open": [open_from_signal(csp_signal(mid=4.00))], "closed": []}
    res = update_positions(store, {}, EXITS, as_of="2026-10-18")
    assert res["updated"][0]["reason"] == "stale"
    assert len(store["open"]) == 1 and res["closed_today"] == []


# --- chain indexing / snapshots -------------------------------------------

def test_chain_index_rounds_and_skips_bad_rows():
    import pandas as pd
    puts = pd.DataFrame([
        {"strike": 600.0, "bid": 3.8, "ask": 4.2},
        {"strike": 590.0, "bid": 0.0, "ask": 0.0},   # no quote -> skipped
        {"strike": float("nan"), "bid": 1.0, "ask": 1.2},
    ])
    idx = chain_index(puts)
    assert idx == {600.0: 4.00}


def test_build_snapshot_csp_and_spread():
    from options_screener.exits import build_snapshot
    mkt = {"marks": {600.0: 4.00, 585.0: 1.00}, "underlying": 612.5}
    s = build_snapshot(open_from_signal(csp_signal()), mkt, "2026-10-18")
    assert s == {"date": "2026-10-18", "underlying": 612.5,
                 "mark": 4.00, "dte": 33}
    s = build_snapshot(open_from_spread(bull_spread()), mkt, "2026-10-18")
    assert s["mark"] == 3.00  # 4.00 - 1.00


def test_build_snapshot_missing_leg_is_none_mark():
    from options_screener.exits import build_snapshot
    mkt = {"marks": {600.0: 4.00}, "underlying": 612.5}  # long leg gone
    s = build_snapshot(open_from_spread(bull_spread()), mkt, "2026-10-18")
    assert s["mark"] is None


def test_build_snapshot_bad_expiry_is_none():
    from options_screener.exits import build_snapshot
    mkt = {"marks": {600.0: 4.00}, "underlying": 612.5}
    p = open_from_signal(csp_signal(expiry="not-a-date"))
    assert build_snapshot(p, mkt, "2026-10-18") is None


def test_evaluate_dte_none_holds_honestly():
    p = open_from_signal(csp_signal(mid=4.00))
    st = evaluate(p, {"date": "2026-10-18", "underlying": 612.5,
                      "mark": 4.00, "dte": None}, EXITS)
    assert st["action"] == "hold" and st["reason"] == "stale"


def test_days_between_open_and_close():
    assert _days_between_safe("2026-10-01", "2026-10-18") == 17
    assert _days_between_safe("2026-10-18", "2026-10-18") == 0


def _days_between_safe(a, b):
    from options_screener.exits import _days_between
    return _days_between(a, b)
