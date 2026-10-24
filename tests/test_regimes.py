"""Tests for the v6 IV-regime split (research only).

All tests use fixed-seed synthetic price series or hand-built trade dicts
— no network, no fixtures.
"""

import json
import math
import os
import random

import yaml

from options_screener.backtest import (build_regime_payload, median_iv,
                                       render_regime_markdown, simulate,
                                       split_by_iv_regime, summarize)

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


def trade(iv, pnl):
    """Minimal trade dict with the fields the split reads."""
    return {"iv": iv, "pnl_per_contract": pnl, "return_on_risk": pnl / 10000.0,
            "days_held": 10, "exit_reason": "profit" if pnl > 0 else "loss"}


# ---------------------------------------------------------------- median

def test_median_iv_odd_even_and_empty():
    assert median_iv([trade(0.1, 1), trade(0.3, 1), trade(0.2, 1)]) == 0.2
    assert median_iv([trade(0.1, 1), trade(0.3, 1)]) == 0.2
    assert median_iv([]) is None


# ---------------------------------------------------------------- split

def test_split_is_even_at_median_and_partitions():
    trades = [trade(iv, pnl) for iv, pnl in
              [(0.10, 100), (0.20, -50), (0.30, 200), (0.40, -100)]]
    split = split_by_iv_regime(trades)
    assert split["median_iv"] == 0.25
    assert len(split["low"]["trades"]) == 2
    assert len(split["high"]["trades"]) == 2
    # every trade lands in exactly one regime, none lost or duplicated
    assert (sorted(t["iv"] for t in split["low"]["trades"])
            + sorted(t["iv"] for t in split["high"]["trades"])) == \
        sorted(t["iv"] for t in trades)


def test_split_summaries_match_subsets():
    trades = [trade(iv, pnl) for iv, pnl in
              [(0.10, 100), (0.20, -50), (0.30, 200), (0.40, -100)]]
    split = split_by_iv_regime(trades)
    assert split["low"]["summary"]["total_pnl"] == 50.0
    assert split["high"]["summary"]["total_pnl"] == 100.0
    assert split["low"]["summary"]["win_rate"] == 0.5
    assert split["high"]["summary"]["win_rate"] == 0.5


def test_split_empty_and_single_trade():
    split = split_by_iv_regime([])
    assert split["median_iv"] is None
    assert split["low"]["summary"]["trades"] == 0
    assert split["high"]["summary"]["trades"] == 0
    assert split["high"]["summary"]["win_rate"] is None
    one = split_by_iv_regime([trade(0.22, 80)])
    assert one["median_iv"] == 0.22
    assert one["low"]["summary"]["trades"] == 1  # iv <= median goes low
    assert one["high"]["summary"]["trades"] == 0  # empty, not fake zeros


def test_split_all_same_iv_leaves_high_empty():
    trades = [trade(0.20, pnl) for pnl in (100, -50, 30)]
    split = split_by_iv_regime(trades)
    assert split["median_iv"] == 0.20
    assert split["low"]["summary"]["trades"] == 3
    assert split["high"]["summary"]["trades"] == 0


def test_split_on_real_simulation():
    cfg = load_cfg()
    trades = simulate(gbmm_walk(504), cfg)
    assert len(trades) > 10
    split = split_by_iv_regime(trades)
    lo, hi = split["low"]["trades"], split["high"]["trades"]
    assert abs(len(lo) - len(hi)) <= 1  # median split is roughly even
    assert all(t["iv"] <= split["median_iv"] for t in lo)
    assert all(t["iv"] > split["median_iv"] for t in hi)
    # regime P&Ls add up to the whole-period P&L (within rounding: each
    # summary rounds its total to cents independently)
    assert abs(split["low"]["summary"]["total_pnl"] +
               split["high"]["summary"]["total_pnl"] -
               summarize(trades)["total_pnl"]) < 0.02


# ---------------------------------------------------------------- payload/report

def test_regime_payload_json_serializable():
    cfg = load_cfg()
    trades = simulate(gbmm_walk(504), cfg)
    payload = build_regime_payload({"SPY": {"trades": trades,
                                            "summary": summarize(trades)}},
                                   cfg)
    json.dumps(payload)  # must not raise
    assert payload["mode"] == "IV-regime split"
    assert any("median" in a for a in payload["assumptions"])
    assert any("not a live entry filter" in a or "cannot be traded" in a
               for a in payload["assumptions"])


def test_regime_markdown_renders_both_regimes_and_gap():
    cfg = load_cfg()
    trades = [trade(iv, pnl) for iv, pnl in
              [(0.10, 100), (0.20, -50), (0.30, 200), (0.40, -100)]]
    payload = build_regime_payload({"SPY": {"trades": trades,
                                            "summary": summarize(trades)}},
                                   cfg)
    md = render_regime_markdown(payload)
    assert "Low IV" in md and "High IV" in md
    assert "Median entry IV" in md
    assert "High-IV minus low-IV" in md
    assert "$+50.00" in md  # high (100) minus low (50) total P&L gap
    assert "paper-trade before" in md  # risk disclaimer footer


def test_regime_markdown_empty_ticker():
    cfg = load_cfg()
    payload = build_regime_payload({"SPY": {"trades": [],
                                            "summary": summarize([])}}, cfg)
    md = render_regime_markdown(payload)
    assert "No completed trades" in md
