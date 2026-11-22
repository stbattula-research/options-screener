"""Tests for the CSP vs bull-put-spread head-to-head (v20).

All synthetic, no network. The point of the mode is that both structures
run over the SAME entry-attempt schedule, so the comparison isolates the
structure — not the calendar. The tests pin that schedule down, plus the
honesty rule the report carries: avg return on risk divides by different
capital bases per structure, so the dollar columns (P&L per contract,
avg max loss per contract) are the comparable numbers.
"""

import json
import math

import pytest

from options_screener.backtest import (attempt_days,
                                       avg_max_loss_per_contract,
                                       build_structure_compare_payload,
                                       compare_structures,
                                       render_structure_compare_markdown,
                                       simulate_structure_on_schedule)


def make_cfg(**overrides):
    cfg = {
        "risk_free_rate": 0.04,
        "strategy": {"target_delta": -0.30, "dte_min": 30, "dte_max": 45},
        "exits": {"profit_target_pct": 0.50, "manage_dte": 21,
                  "loss_manage_multiple": 2.0},
        "backtest": {"entry_dte": 35, "entry_interval_days": 5,
                     "iv_window": 20, "iv_multiplier": 1.15,
                     "iv_fallback": 0.25},
        "spread": {"min_width_pct": 0.02, "max_width_pct": 0.06,
                   "min_credit_fraction": 0.33},
    }
    for section, vals in overrides.items():
        cfg[section].update(vals)
    return cfg


# --- attempt_days ----------------------------------------------------------

def test_attempt_days_spacing_and_tail():
    # n=100, window=20, interval=5, dte=35: last attempt is day 60
    # (60+35 < 100); day 65 would need 65+35 = 100, not < 100.
    assert attempt_days(100, 20, 5, 35) == [20, 25, 30, 35, 40, 45,
                                            50, 55, 60]


def test_attempt_days_empty_when_series_too_short():
    assert attempt_days(40, 20, 5, 35) == []
    assert attempt_days(55, 20, 5, 35) == []  # 20 + 35 = 55, not < 55


def test_attempt_days_honours_interval():
    assert attempt_days(200, 20, 10, 35)[1] - attempt_days(200, 20, 10, 35)[0] == 10


# --- schedule mechanics ----------------------------------------------------

def test_flat_market_csp_skips_two_attempts_per_trade():
    # Flat prices exit on the 21-DTE time rule: 14 days held (see
    # test_backtest.test_flat_market_exits_at_21_dte_management). With a
    # 5-day attempt interval, attempts at entry+5 and entry+10 land on
    # an open position and are skipped; entry+15 opens the next trade.
    cfg = make_cfg()
    prices = [100.0] * 300
    days = attempt_days(len(prices), 20, 5, 35)
    assert len(days) == 49
    trades, skipped = simulate_structure_on_schedule(prices, cfg, days, "csp")
    assert [t["entry_day"] for t in trades] == list(range(20, 261, 15))
    assert len(trades) == 17
    assert skipped == 32
    assert len(trades) + skipped == len(days)


def test_schedule_invariants_on_wavy_series():
    cfg = make_cfg()
    prices = [100.0 + 8.0 * math.sin(i / 9.0) + 0.04 * i for i in range(500)]
    days = attempt_days(len(prices), 20, 5, 35)
    for structure, kwargs in (("csp", {}),
                              ("spread", {"skew": 2.0, "width_pct": 0.02})):
        trades, skipped = simulate_structure_on_schedule(
            prices, cfg, days, structure, **kwargs)
        assert len(trades) + skipped == len(days)
        assert all(t["entry_day"] in days for t in trades)
        for prev, nxt in zip(trades, trades[1:]):
            assert nxt["entry_day"] > prev["exit_day"]
            assert nxt["entry_day"] - prev["entry_day"] >= 5


def test_unknown_structure_raises():
    cfg = make_cfg()
    prices = [100.0] * 100
    with pytest.raises(ValueError):
        simulate_structure_on_schedule(prices, cfg, [20, 25], "iron-condor")
    with pytest.raises(ValueError):
        avg_max_loss_per_contract([{"strike": 100.0,
                                    "premium_per_share": 2.0}], "straddle")


# --- avg max loss ----------------------------------------------------------

def test_avg_max_loss_csp_is_strike_minus_premium():
    trades = [{"strike": 100.0, "premium_per_share": 2.0},
              {"strike": 90.0, "premium_per_share": 1.0}]
    assert avg_max_loss_per_contract(trades, "csp") == pytest.approx(
        ((100 - 2) + (90 - 1)) / 2 * 100)


def test_avg_max_loss_spread_is_capped_width_minus_credit():
    trades = [{"max_loss_per_contract": 150.0},
              {"max_loss_per_contract": 170.0}]
    assert avg_max_loss_per_contract(trades, "spread") == pytest.approx(160.0)


def test_avg_max_loss_none_when_no_trades():
    assert avg_max_loss_per_contract([], "csp") is None
    assert avg_max_loss_per_contract([], "spread") is None


# --- compare_structures ----------------------------------------------------

def test_compare_uses_one_shared_schedule():
    cfg = make_cfg()
    prices = [100.0] * 300
    res = compare_structures(prices, cfg, skew=2.0, width_pct=0.02)
    assert res["attempts"] == 49
    assert res["skew"] == 2.0
    assert res["width_pct"] == 0.02
    for side in ("csp", "spread"):
        assert (len(res[side]["trades"]) + res[side]["skipped"]
                == res["attempts"])
        assert res[side]["summary"]["trades"] == len(res[side]["trades"])


def test_spread_risks_far_less_capital_than_csp():
    # Structural, not market-dependent: the spread's max loss is capped
    # at width minus credit (~2% of the strike), the CSP's at roughly
    # the strike itself. On the same schedule the spread's avg max loss
    # must be an order of magnitude smaller.
    cfg = make_cfg()
    prices = [100.0] * 300
    res = compare_structures(prices, cfg, skew=2.0, width_pct=0.02)
    csp_risk = res["csp"]["avg_max_loss_per_contract"]
    spr_risk = res["spread"]["avg_max_loss_per_contract"]
    assert spr_risk is not None and csp_risk is not None
    assert spr_risk < csp_risk / 10
    # and the spread summary carries its credit-rule stats
    assert res["spread"]["summary"]["avg_credit_fraction"] is not None
    assert res["spread"]["summary"]["credit_rule_pass_rate"] is not None


def test_compare_on_too_short_series_is_empty_not_broken():
    cfg = make_cfg()
    res = compare_structures([100.0] * 40, cfg, skew=2.0, width_pct=0.02)
    assert res["attempts"] == 0
    for side in ("csp", "spread"):
        assert res[side]["trades"] == []
        assert res[side]["summary"]["trades"] == 0
        assert res[side]["avg_max_loss_per_contract"] is None


# --- payload + markdown ----------------------------------------------------

def test_payload_json_serializable_and_marks_ror_caveat():
    cfg = make_cfg()
    prices = [100.0] * 300
    payload = build_structure_compare_payload(
        {"SPY": compare_structures(prices, cfg, skew=2.0, width_pct=0.02)},
        cfg, 2.0, 0.02)
    json.dumps(payload)  # must not raise
    assert payload["mode"] == "CSP vs bull-put-spread head-to-head"
    assert payload["skew"] == 2.0
    assert any("NOT comparable" in a for a in payload["assumptions"])


def test_markdown_rows_and_reading_line():
    cfg = make_cfg()
    prices = [100.0] * 300
    payload = build_structure_compare_payload(
        {"SPY": compare_structures(prices, cfg, skew=2.0, width_pct=0.02)},
        cfg, 2.0, 0.02)
    md = render_structure_compare_markdown(payload)
    assert "Cash-secured put" in md
    assert "Bull put spread" in md
    assert "Reading for SPY" in md
    assert "compare the dollar columns, not the percentages" in md
    assert "Educational research tool, not financial advice" in md


def test_markdown_empty_sides_render_dashes():
    cfg = make_cfg()
    payload = build_structure_compare_payload(
        {"SPY": compare_structures([100.0] * 40, cfg, skew=2.0,
                                   width_pct=0.02)},
        cfg, 2.0, 0.02)
    md = render_structure_compare_markdown(payload)
    assert "| Cash-secured put | 0 | — |" in md
    assert "| Bull put spread | 0 | — |" in md
    assert "Reading for SPY" not in md  # no reading when a side never traded
