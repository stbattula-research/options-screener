"""Tests for the v15 multi-ticker regime comparison (research only).

All tests use hand-built trade dicts or fixed-seed synthetic series —
no network, no fixtures.
"""

import json
import os

import yaml

from options_screener.backtest import (build_regime_compare_payload,
                                       compare_regime_splits,
                                       render_regime_compare_markdown,
                                       split_by_iv_regime)

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def trade(iv, pnl, dd_days=10):
    """Minimal trade dict with the fields the split reads."""
    return {"iv": iv, "pnl_per_contract": pnl, "return_on_risk": pnl / 10000.0,
            "days_held": dd_days, "exit_reason": "profit" if pnl > 0 else "loss"}


def splits_for(pairs_by_ticker):
    """{ticker: [(iv, pnl), ...]} -> {ticker: split_by_iv_regime(...)}."""
    return {t: split_by_iv_regime([trade(iv, pnl) for iv, pnl in pairs])
            for t, pairs in pairs_by_ticker.items()}


# ---------------------------------------------------------------- gaps

def test_gap_math_high_minus_low():
    # low regime: 100 + (-50) = 50; high regime: 200 + (-100) = 100
    compared = compare_regime_splits(splits_for({
        "SPY": [(0.10, 100), (0.20, -50), (0.30, 200), (0.40, -100)],
    }))
    row = compared["rows"][0]
    assert row["median_iv"] == 0.25
    assert row["low_total_pnl"] == 50.0
    assert row["high_total_pnl"] == 100.0
    assert row["pnl_gap"] == 50.0
    # low win 1/2, high win 1/2 -> no win gap
    assert row["win_gap"] == 0.0


def test_rows_ranked_by_pnl_gap_descending():
    compared = compare_regime_splits(splits_for({
        "LOW": [(0.10, 0), (0.20, 0), (0.30, 0), (0.40, 0)],
        "MID": [(0.10, 100), (0.20, 100), (0.30, 300), (0.40, 300)],
        "HIGH": [(0.10, -100), (0.20, -100), (0.30, 500), (0.40, 500)],
    }))
    gaps = [r["pnl_gap"] for r in compared["rows"]]
    assert gaps == sorted(gaps, reverse=True)
    assert compared["rows"][0]["ticker"] == "HIGH"
    assert compared["rows"][-1]["ticker"] == "LOW"


def test_gaps_are_none_when_a_regime_is_empty():
    # every trade shares one IV -> high regime empty (v6 behaviour)
    compared = compare_regime_splits(splits_for({
        "FLAT": [(0.20, 100), (0.20, -50), (0.20, 200)],
    }))
    row = compared["rows"][0]
    assert row["high_trades"] == 0
    assert row["pnl_gap"] is None
    assert row["win_gap"] is None
    assert row["dd_gap"] is None
    assert row["thin"] is True


def test_empty_ticker_row_does_not_crash():
    compared = compare_regime_splits({"EMPTY": split_by_iv_regime([])})
    row = compared["rows"][0]
    assert row["median_iv"] is None
    assert row["low_trades"] == 0 and row["high_trades"] == 0
    assert row["pnl_gap"] is None
    assert row["thin"] is True
    assert compared["summary"]["readable_tickers"] == 0


# ---------------------------------------------------------------- thin

def test_thin_flag_below_min_regime_trades():
    # 4 trades per regime is below the 10-trade floor
    compared = compare_regime_splits(splits_for({
        "SPY": [(0.10, 100), (0.20, -50), (0.30, 200), (0.40, -100),
                (0.11, 50), (0.21, -25), (0.31, 150), (0.41, -75)],
    }))
    assert compared["rows"][0]["thin"] is True
    assert compared["summary"]["readable_tickers"] == 0


def test_readable_when_both_regimes_have_ten_or_more():
    pairs = [(0.10 + 0.001 * i, 100 if i % 2 == 0 else -50)
             for i in range(10)]
    pairs += [(0.30 + 0.001 * i, 200 if i % 2 == 0 else -100)
              for i in range(10)]
    compared = compare_regime_splits(splits_for({"SPY": pairs}))
    row = compared["rows"][0]
    assert row["low_trades"] == 10 and row["high_trades"] == 10
    assert row["thin"] is False
    assert compared["summary"]["readable_tickers"] == 1


# ---------------------------------------------------------------- summary

def test_summary_counts_only_readable_rows():
    big = [(0.10 + 0.001 * i, 100) for i in range(10)]
    big += [(0.30 + 0.001 * i, 200) for i in range(10)]
    compared = compare_regime_splits(splits_for({
        "BIG": big,                       # readable, high wins on P&L
        "SMALL": [(0.10, 1), (0.20, 2), (0.30, 3), (0.40, 4)],  # thin
    }))
    s = compared["summary"]
    assert s["tickers_compared"] == 2
    assert s["readable_tickers"] == 1
    assert s["high_wins_pnl"] == 1
    assert s["median_pnl_gap"] == 1000.0  # (10*200) - (10*100)
    assert s["median_win_gap"] == 0.0


def test_median_gap_is_the_middle_value():
    def big_pairs(low_pnl, high_pnl):
        pairs = [(0.10 + 0.001 * i, low_pnl) for i in range(10)]
        pairs += [(0.30 + 0.001 * i, high_pnl) for i in range(10)]
        return pairs
    compared = compare_regime_splits(splits_for({
        "A": big_pairs(0, 10),    # gap +100
        "B": big_pairs(0, 30),    # gap +300
        "C": big_pairs(0, 20),    # gap +200
    }))
    assert compared["summary"]["median_pnl_gap"] == 200.0


# ------------------------------------------------------- scale-free gaps

def test_same_pattern_at_different_iv_scales_gives_same_gaps():
    # v6's core honesty rule: medians differ per ticker, so the
    # comparison must be of gaps, not levels. The same trade pattern at
    # IV ~0.15 and at IV ~0.45 must produce identical gaps.
    pattern = [(0.10, 100), (0.20, -50), (0.30, 200), (0.40, -100)]
    shifted = [(iv + 0.30, pnl) for iv, pnl in pattern]
    compared = compare_regime_splits(splits_for({
        "CALM": pattern, "WILD": shifted,
    }))
    calm, wild = compared["rows"]
    assert calm["pnl_gap"] == wild["pnl_gap"] == 50.0
    assert calm["median_iv"] == 0.25
    assert wild["median_iv"] == 0.55


# ---------------------------------------------------------------- payload

def test_payload_is_json_serializable_and_carries_caveats():
    payload = build_regime_compare_payload(
        splits_for({"SPY": [(0.10, 100), (0.20, -50),
                            (0.30, 200), (0.40, -100)]}),
        load_cfg())
    json.dumps(payload)  # must not raise
    assert payload["mode"] == "Multi-ticker IV-regime comparison"
    joined = " ".join(payload["assumptions"])
    assert "NOT comparable" in joined
    assert "median" in joined.lower()


def test_markdown_renders_table_rows_gaps_and_caveat():
    payload = build_regime_compare_payload(
        splits_for({
            "SPY": [(0.10, 100), (0.20, -50), (0.30, 200), (0.40, -100)],
            "EMPTY": [],
        }),
        load_cfg())
    md = render_regime_compare_markdown(payload)
    assert "| SPY |" in md
    assert "| EMPTY |" in md
    assert "P&L gap" in md
    assert "thin ⚠" in md  # SPY has 2 trades per regime: thin
    assert "NOT comparable" in md
    assert "not a live rule" in md or "descriptive" in md
    # empty row renders dashes, never "None"
    assert "None" not in md


def test_markdown_summary_counts_high_wins():
    big = [(0.10 + 0.001 * i, 100) for i in range(10)]
    big += [(0.30 + 0.001 * i, 200) for i in range(10)]
    payload = build_regime_compare_payload(
        splits_for({"SPY": big}), load_cfg())
    md = render_regime_compare_markdown(payload)
    assert "1 of 1 readable tickers" in md
    assert "readable" in md
