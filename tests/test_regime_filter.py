"""Tests for the v16 regime-aware forward entry filter (research only).

All tests use fixed-seed synthetic price series or hand-built result dicts
— no network, no fixtures.
"""

import json
import math
import os
import random

import yaml

from options_screener.backtest import (DEFAULT_REGIME_FILTER,
                                       build_regime_filter_payload,
                                       compare_regime_filter, estimate_iv,
                                       render_regime_filter_markdown,
                                       simulate, simulate_regime_filtered,
                                       summarize, trailing_iv_median)

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def gbmm_walk(n, start=100.0, mu=0.0003, sigma=0.011, seed=7):
    """Deterministic daily closes: GBM with a fixed seed."""
    rng = random.Random(seed)
    prices = [start]
    for _ in range(1, n):
        prices.append(prices[-1] * math.exp(
            mu - 0.5 * sigma ** 2 + sigma * rng.gauss(0, 1)))
    return prices


def burst_walk(n=400, start=100.0):
    """Low-vol closes, a high-vol burst in the middle, low-vol again.

    Returns alternate +a/-a, so the trailing realized vol is constant
    inside each regime and jumps only at the burst edges.
    """
    prices = [start]
    for i in range(1, n):
        a = 0.04 if 200 <= i < 240 else 0.01
        prices.append(prices[-1] * (1 + a if i % 2 else 1 - a))
    return prices


def flat_walk(n=200, start=100.0):
    """Constant closes: realized vol is 0, so the IV proxy is the fallback."""
    return [start] * n


# ------------------------------------------------------- trailing_iv_median

def test_trailing_iv_median_matches_manual_computation():
    cfg = load_cfg()
    bt = cfg["backtest"]
    prices = gbmm_walk(300, seed=11)
    day, lookback = 200, 60
    expected = sorted(estimate_iv(prices, d, bt["iv_window"],
                                  bt["iv_multiplier"], bt["iv_fallback"])
                      for d in range(day - lookback, day))
    mid = len(expected) // 2
    want = (expected[mid] if len(expected) % 2
            else (expected[mid - 1] + expected[mid]) / 2.0)
    assert trailing_iv_median(prices, day, lookback, bt["iv_window"],
                              bt["iv_multiplier"],
                              bt["iv_fallback"]) == want


def test_trailing_iv_median_needs_history():
    cfg = load_cfg()
    bt = cfg["backtest"]
    prices = gbmm_walk(300, seed=11)
    # day < lookback: the threshold window does not exist yet
    assert trailing_iv_median(prices, 30, 60, bt["iv_window"],
                              bt["iv_multiplier"], bt["iv_fallback"]) is None


def test_trailing_iv_median_rejects_bad_lookback():
    cfg = load_cfg()
    bt = cfg["backtest"]
    prices = gbmm_walk(300, seed=11)
    for bad in (0, -5):
        try:
            trailing_iv_median(prices, 200, bad, bt["iv_window"],
                               bt["iv_multiplier"], bt["iv_fallback"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"lookback {bad} should raise ValueError")


def test_trailing_iv_median_excludes_today():
    # A one-day lookback sees only yesterday's proxy, never today's.
    cfg = load_cfg()
    bt = cfg["backtest"]
    prices = gbmm_walk(300, seed=11)
    day = 200
    yesterday = estimate_iv(prices, day - 1, bt["iv_window"],
                            bt["iv_multiplier"], bt["iv_fallback"])
    assert trailing_iv_median(prices, day, 1, bt["iv_window"],
                              bt["iv_multiplier"],
                              bt["iv_fallback"]) == yesterday


# ------------------------------------------------------------ start_day

def test_simulate_start_day_shifts_first_entry():
    cfg = load_cfg()
    prices = gbmm_walk(400, seed=3)
    default_trades = simulate(prices, cfg)
    late_trades = simulate(prices, cfg, start_day=150)
    assert default_trades[0]["entry_day"] < 150
    assert all(t["entry_day"] >= 150 for t in late_trades)
    # default start_day is the iv_window, unchanged from before
    assert simulate(prices, cfg, start_day=cfg["backtest"]["iv_window"]) \
        == default_trades


# ------------------------------------------------- simulate_regime_filtered

def test_filtered_trades_all_above_threshold_and_skips_below():
    cfg = load_cfg()
    bt = cfg["backtest"]
    lookback = 60
    prices = burst_walk()
    res = simulate_regime_filtered(prices, cfg, lookback_days=lookback)
    assert res["trades"], "burst series should produce at least one trade"
    assert res["skipped"], "burst series should skip at least one attempt"
    for t in res["trades"]:
        day = t["entry_day"]
        sigma = estimate_iv(prices, day, bt["iv_window"],
                            bt["iv_multiplier"], bt["iv_fallback"])
        threshold = trailing_iv_median(prices, day, lookback,
                                       bt["iv_window"], bt["iv_multiplier"],
                                       bt["iv_fallback"])
        assert sigma > threshold, (
            f"trade entered at day {day} with IV {sigma} <= threshold "
            f"{threshold}")
    for s in res["skipped"]:
        assert s["entry_iv"] <= s["threshold_iv"], (
            f"skipped entry day {s['entry_day']} has IV "
            f"{s['entry_iv']} > threshold {s['threshold_iv']}")


def test_filtered_never_trades_before_warmup():
    cfg = load_cfg()
    bt = cfg["backtest"]
    lookback = 60
    prices = burst_walk()
    res = simulate_regime_filtered(prices, cfg, lookback_days=lookback)
    warmup = bt["iv_window"] + lookback
    assert res["warmup_days"] == warmup
    assert all(t["entry_day"] >= warmup for t in res["trades"])
    assert all(s["entry_day"] >= warmup for s in res["skipped"])


def test_filtered_skips_everything_when_iv_is_flat():
    cfg = load_cfg()
    bt = cfg["backtest"]
    lookback = 10
    prices = flat_walk(200)
    res = simulate_regime_filtered(prices, cfg, lookback_days=lookback)
    # constant closes -> every day's proxy is the fallback, so entry IV
    # always equals its own trailing median and the strict > never passes
    assert res["trades"] == []
    warmup = bt["iv_window"] + lookback
    dte = bt.get("entry_dte") or 35
    expected_attempts = len(range(warmup, len(prices) - dte,
                                  bt["entry_interval_days"]))
    assert len(res["skipped"]) == expected_attempts
    assert all(s["entry_iv"] == s["threshold_iv"] == bt["iv_fallback"]
               for s in res["skipped"])


def test_filtered_empty_on_too_short_series():
    cfg = load_cfg()
    res = simulate_regime_filtered(gbmm_walk(50, seed=5), cfg)
    assert res["trades"] == [] and res["skipped"] == []
    assert res["summary"]["trades"] == 0


def test_filtered_rejects_bad_lookback():
    cfg = load_cfg()
    try:
        simulate_regime_filtered(gbmm_walk(300, seed=5), cfg,
                                 lookback_days=0)
    except ValueError:
        pass
    else:
        raise AssertionError("lookback 0 should raise ValueError")


def test_filtered_uses_config_lookback_by_default():
    cfg = load_cfg()
    prices = burst_walk()
    res = simulate_regime_filtered(prices, cfg)
    assert res["lookback_days"] == DEFAULT_REGIME_FILTER["lookback_days"]
    assert res["lookback_days"] == cfg["regime_filter"]["lookback_days"]


# ------------------------------------------------------- compare + payload

def _plain_filtered(plain_pnls, filt_pnls, skipped=3):
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


def test_compare_ranks_by_pnl_delta_and_counts_summary():
    results = {
        "AAA": _plain_filtered([100] * 12, [300] * 6),   # delta +600
        "BBB": _plain_filtered([100] * 12, [-50] * 6),   # delta -1500
        "CCC": _plain_filtered([100] * 12, [150] * 6),   # delta -300
    }
    compared = compare_regime_filter(results)
    assert [r["ticker"] for r in compared["rows"]] == ["AAA", "CCC", "BBB"]
    s = compared["summary"]
    assert s["tickers_compared"] == 3
    assert s["readable_tickers"] == 3
    assert s["filtered_wins_pnl"] == 1
    assert s["median_pnl_delta"] == -300.0
    assert s["median_trade_share"] == 0.5


def test_compare_thin_rows_never_count_toward_summary():
    results = {
        "AAA": _plain_filtered([100] * 12, [300] * 6),  # readable, delta +600
        "THIN": _plain_filtered([100] * 12, [500] * 2),  # filtered < 5
    }
    compared = compare_regime_filter(results)
    thin = next(r for r in compared["rows"] if r["ticker"] == "THIN")
    assert thin["thin"] is True
    assert compared["summary"]["readable_tickers"] == 1
    assert compared["summary"]["filtered_wins_pnl"] == 1
    assert compared["summary"]["median_pnl_delta"] == 600.0


def test_compare_zero_filtered_trades_reports_delta_not_zeros():
    results = {"AAA": _plain_filtered([100] * 12, [], skipped=9)}
    compared = compare_regime_filter(results)
    row = compared["rows"][0]
    assert row["filtered_trades"] == 0
    assert row["pnl_delta"] == -1200.0  # 0 minus the plain total
    assert row["win_gap"] is None
    assert row["dd_gap"] is None
    assert row["thin"] is True


def test_payload_json_serializable_and_markdown_renders():
    cfg = load_cfg()
    results = {"AAA": _plain_filtered([100] * 12, [200] * 6)}
    payload = build_regime_filter_payload(
        results, cfg, DEFAULT_REGIME_FILTER["lookback_days"])
    json.dumps(payload)  # must not raise
    md = render_regime_filter_markdown(payload)
    assert "AAA" in md
    assert "trailing 252-day median" in md
    assert "No look-ahead" in md
    assert "one forward test" in md
