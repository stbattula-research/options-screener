"""Unit tests for filters and signal selection — no network access."""

import pandas as pd
import pytest

from options_screener.filters import liquidity_ok, quote_stats, strategy_ok
from options_screener.signals import exit_plan, find_candidates

CFG = {
    "strategy": {"dte_min": 30, "dte_max": 45, "target_delta": -0.30,
                 "delta_tolerance": 0.07, "min_premium_pct": 0.01},
    "liquidity": {"min_open_interest": 500, "min_volume": 100,
                  "max_spread_pct": 0.10},
    "risk_free_rate": 0.04,
    "exits": {"profit_target_pct": 0.50, "manage_dte": 21,
              "loss_manage_multiple": 2.0},
}

ROW = {"bid": 1.00, "ask": 1.10, "openInterest": 1000, "volume": 250}


def test_quote_stats_mid_and_spread():
    q = quote_stats(ROW)
    assert q["mid"] == pytest.approx(1.05)
    assert q["spread_pct"] == pytest.approx(0.10 / 1.05)


def test_quote_stats_rejects_bad_quotes():
    assert quote_stats({"bid": 0.0, "ask": 0.0, **{k: v for k, v in ROW.items()
                                                  if k not in ("bid", "ask")}}) is None
    assert quote_stats({"bid": 1.2, "ask": 1.0, "openInterest": 1, "volume": 1}) is None
    assert quote_stats({}) is None


def test_liquidity_ok_pass():
    ok, reason = liquidity_ok(ROW, CFG)
    assert ok and reason == ""


def test_liquidity_rejects_thin_book():
    ok, reason = liquidity_ok({**ROW, "openInterest": 10}, CFG)
    assert not ok and "open interest" in reason
    ok, reason = liquidity_ok({**ROW, "volume": 5}, CFG)
    assert not ok and "volume" in reason


def test_liquidity_rejects_wide_spread():
    ok, reason = liquidity_ok({**ROW, "bid": 0.50, "ask": 1.50}, CFG)
    assert not ok and "spread" in reason


def test_strategy_ok_delta_band():
    ok, _ = strategy_ok(-0.30, 0.02, CFG)
    assert ok
    ok, reason = strategy_ok(-0.10, 0.02, CFG)
    assert not ok and "delta" in reason


def test_strategy_ok_min_premium():
    ok, reason = strategy_ok(-0.30, 0.005, CFG)
    assert not ok and "premium" in reason


def _chain():
    # synthetic put chain: $100 underlying, 37 DTE.
    # strike 97 @ 22% IV lands at ~-0.30 delta (d1 ~ 0.52) -> the valid signal.
    rows = []
    for strike, iv, bid, ask, vol in [
            (97, 0.22, 1.30, 1.40, 300),  # ~30 delta, liquid -> signal
            (90, 0.24, 0.45, 0.55, 300),  # too far OTM (delta ~ -0.07)
            (100, 0.20, 2.80, 2.95, 300), # too ITM (delta ~ -0.46)
            (97, 0.22, 1.30, 1.40, 10),   # right strike, thin volume
    ]:
        rows.append({"strike": strike, "impliedVolatility": iv, "bid": bid,
                     "ask": ask, "openInterest": 800, "volume": vol})
    return pd.DataFrame(rows)


def test_find_candidates_picks_only_valid():
    signals, rejected = find_candidates(_chain(), 100.0, "2026-11-13", 37,
                                        (0.15, 0.35), CFG)
    assert len(signals) == 1
    s = signals[0]
    assert s["strike"] == 97.0
    assert abs(s["delta"] - (-0.30)) <= 0.07
    assert s["breakeven"] == pytest.approx(97.0 - 1.35)
    assert 0.0 <= s["iv_vs_hv_rank"] <= 1.0


def test_find_candidates_ranked_by_yield():
    df = pd.DataFrame([
        {"strike": 97, "impliedVolatility": 0.22, "bid": 1.30, "ask": 1.40,
         "openInterest": 800, "volume": 300},
        {"strike": 98, "impliedVolatility": 0.21, "bid": 1.60, "ask": 1.70,
         "openInterest": 900, "volume": 400},
    ])
    signals, _ = find_candidates(df, 100.0, "2026-11-13", 37, None, CFG)
    assert len(signals) == 2
    assert signals[0]["annualized_yield"] >= signals[1]["annualized_yield"]


def test_exit_plan_mentions_rules():
    sig = {"max_profit_per_share": 1.00}
    plan = exit_plan(sig, CFG)
    assert "50%" in plan["profit_target"]
    assert "21 DTE" in plan["time_exit"]
    assert "2x" in plan["loss_manage"]
