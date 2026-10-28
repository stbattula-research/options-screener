"""Tests for the spread-width sweep backtest (v8).

All synthetic, no network. v7's note claimed narrower widths move the
credit/width ratio toward the live 1/3 rule; this sweep measures that
claim across widths, and these tests pin the two directions that
matter: credit fraction rises as the wing narrows, while absolute
credit collected per trade falls.
"""

import json
import math

import pytest

from options_screener.backtest import (build_width_payload,
                                       render_width_markdown,
                                       simulate_spread_trade,
                                       sweep_width_spreads)
from tests.test_spread_backtest import make_cfg


def test_credit_fraction_rises_as_width_narrows():
    cfg = make_cfg()
    prices = [100.0] * 300
    fracs = [simulate_spread_trade(prices, 20, 35, 0.25, cfg, skew=2.0,
                                   width_pct=w)["credit_fraction"]
             for w in (0.06, 0.04, 0.02, 0.01)]
    assert fracs[0] < fracs[1] < fracs[2] < fracs[3]


def test_absolute_credit_falls_as_width_narrows():
    cfg = make_cfg()
    prices = [100.0] * 300
    credits = [simulate_spread_trade(prices, 20, 35, 0.25, cfg, skew=2.0,
                                     width_pct=w)["net_credit_per_share"]
               for w in (0.06, 0.02, 0.01)]
    assert credits[0] > credits[1] > credits[2] > 0


def test_sweep_width_keys_and_first_trade_shape():
    cfg = make_cfg()
    prices = [100.0 + 8.0 * math.sin(i / 9.0) + 0.04 * i for i in range(500)]
    out = sweep_width_spreads(prices, cfg, [0.01, 0.02, 0.06], skew=2.0)
    assert set(out) == {0.01, 0.02, 0.06}
    for width, res in out.items():
        assert res["trades"]
        assert res["summary"]["avg_credit_fraction"] is not None
        assert res["summary"]["credit_rule_pass_rate"] is not None
        # Every trade's wing matches the swept width (within rounding).
        for t in res["trades"]:
            assert t["width"] == pytest.approx(
                t["strike"] * width, abs=0.05)
    # First trade: same entry day and short strike across widths — only
    # the long leg moves.
    firsts = [out[w]["trades"][0] for w in (0.01, 0.02, 0.06)]
    assert all(f["entry_day"] == firsts[0]["entry_day"] for f in firsts)
    assert all(f["strike"] == firsts[0]["strike"] for f in firsts)
    assert firsts[0]["long_strike"] > firsts[1]["long_strike"] > firsts[2]["long_strike"]


def test_sweep_avg_credit_fraction_monotone_in_width():
    cfg = make_cfg()
    prices = [100.0 + 8.0 * math.sin(i / 9.0) + 0.04 * i for i in range(500)]
    out = sweep_width_spreads(prices, cfg, [0.01, 0.02, 0.04, 0.06],
                              skew=2.0)
    fracs = [out[w]["summary"]["avg_credit_fraction"]
             for w in (0.01, 0.02, 0.04, 0.06)]
    assert fracs[0] > fracs[1] > fracs[2] > fracs[3]


def test_width_payload_renders_and_is_json_safe():
    cfg = make_cfg()
    prices = [100.0 + 8.0 * math.sin(i / 9.0) + 0.04 * i for i in range(500)]
    sweeps = {"SPY": sweep_width_spreads(prices, cfg, [0.01, 0.02], 2.0)}
    payload = build_width_payload(sweeps, cfg, [0.01, 0.02], 2.0)
    json.dumps(payload)
    info = payload["tickers"]["SPY"]
    assert info["best_pnl_width"] in (0.01, 0.02)
    # best_pnl_width must be the width with the highest total P&L.
    assert info["sweep"][info["best_pnl_width"]]["summary"]["total_pnl"] == max(
        info["sweep"][w]["summary"]["total_pnl"] for w in (0.01, 0.02))
    md = render_width_markdown(payload)
    assert "SPY" in md and "Width" in md and "credit/width" in md
    assert "1%" in md and "2%" in md
