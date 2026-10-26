"""Tests for the skew-aware bull-put-spread backtest (v7).

All synthetic, no network. The key assertion is the corrected v5
hypothesis: skew LOWERS net credit (the long leg's IV rises faster
than the short leg's), so a steeper skew cannot rescue the live
screener's 1/3-width credit rule — it moves the ratio the wrong way.
"""

import json
import math

import pytest

from options_screener.backtest import (build_spread_payload,
                                       render_spread_markdown,
                                       simulate_spread_trade,
                                       simulate_spreads, summarize_spreads,
                                       sweep_skew_spreads)
from options_screener.vol import (bs_put_value, bs_put_value_skewed,
                                  skewed_iv)


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


# --- skewed_iv -----------------------------------------------------------

def test_skewed_iv_atm_and_above_spot_unchanged():
    assert skewed_iv(100.0, 100.0, 0.20, 4.0) == pytest.approx(0.20)
    assert skewed_iv(100.0, 110.0, 0.20, 4.0) == pytest.approx(0.20)


def test_skewed_iv_rises_below_spot_and_with_skew():
    iv1 = skewed_iv(100.0, 95.0, 0.20, 1.0)
    iv2 = skewed_iv(100.0, 95.0, 0.20, 2.0)
    iv_far = skewed_iv(100.0, 90.0, 0.20, 2.0)
    assert 0.20 < iv1 < iv2 < iv_far
    assert iv2 == pytest.approx(0.20 * (1 + 2 * math.log(100 / 95)))


def test_skewed_iv_rejects_bad_inputs():
    with pytest.raises(ValueError):
        skewed_iv(100.0, 95.0, 0.20, -1.0)
    with pytest.raises(ValueError):
        skewed_iv(100.0, 95.0, 0.0, 2.0)
    with pytest.raises(ValueError):
        skewed_iv(0.0, 95.0, 0.20, 2.0)


def test_skewed_put_value_matches_flat_at_zero_skew():
    v_flat = bs_put_value(100.0, 95.0, 35 / 365, 0.04, 0.25)
    v_skew = bs_put_value_skewed(100.0, 95.0, 35 / 365, 0.04, 0.25, 0.0)
    assert v_skew == pytest.approx(v_flat)


def test_skew_raises_far_otm_put_more_than_near_put():
    # The long leg (further OTM) gains more value from skew than the
    # short leg — so net spread credit must FALL as skew rises. This is
    # the corrected v5 hypothesis, asserted directly.
    S, T, r, atm = 100.0, 35 / 365, 0.04, 0.25
    short_k, long_k = 97.0, 92.0
    nets = []
    for skew in (0.0, 2.0, 4.0):
        short_v = bs_put_value_skewed(S, short_k, T, r, atm, skew)
        long_v = bs_put_value_skewed(S, long_k, T, r, atm, skew)
        nets.append(short_v - long_v)
    assert nets[0] > nets[1] > nets[2]


# --- simulate_spread_trade ----------------------------------------------

def test_spread_trade_fields_and_max_loss():
    cfg = make_cfg()
    prices = [100.0] * 300
    t = simulate_spread_trade(prices, 20, 35, 0.25, cfg, skew=0.0,
                              width_pct=0.02)
    assert t["long_strike"] < t["strike"]
    assert t["width"] == pytest.approx(t["strike"] - t["long_strike"], abs=0.02)
    assert t["net_credit_per_share"] > 0
    assert t["credit_fraction"] == pytest.approx(
        t["net_credit_per_share"] / t["width"], abs=1e-3)
    assert t["max_loss_per_contract"] == pytest.approx(
        (t["width"] - t["net_credit_per_share"]) * 100, abs=1.0)
    assert t["return_on_risk"] == pytest.approx(
        t["pnl_per_contract"] / t["max_loss_per_contract"], abs=0.02)


def test_spread_expires_worthless_above_short_strike():
    cfg = make_cfg(exits={"profit_target_pct": 1.0, "manage_dte": 0,
                          "loss_manage_multiple": 100.0})
    prices = [100.0] * 300
    t = simulate_spread_trade(prices, 20, 35, 0.25, cfg, skew=0.0,
                              width_pct=0.02)
    # With profit_target=1.0 the profit trigger is a zero spread value, so
    # expiry day itself fires "profit" instead of "expired" — same economics.
    assert t["exit_reason"] in ("expired", "profit")
    assert t["exit_value_per_share"] == 0.0
    assert t["pnl_per_contract"] == pytest.approx(
        t["net_credit_per_share"] * 100, abs=1.0)


def test_spread_expires_at_full_width_below_long_strike():
    cfg = make_cfg(exits={"profit_target_pct": 1.0, "manage_dte": 0,
                          "loss_manage_multiple": 100.0})
    prices = [100.0] * 25 + [60.0] * 200
    t = simulate_spread_trade(prices, 20, 35, 0.25, cfg, skew=0.0,
                              width_pct=0.02)
    assert t["exit_reason"] == "expired"
    assert t["exit_value_per_share"] == pytest.approx(t["width"], abs=0.05)
    assert t["pnl_per_contract"] < 0


def test_spread_profit_exit_on_flat_market():
    cfg = make_cfg()
    prices = [100.0] * 300
    t = simulate_spread_trade(prices, 20, 35, 0.25, cfg, skew=0.0,
                              width_pct=0.02)
    # Flat market decays the spread; either profit or time exit, never a loss.
    assert t["exit_reason"] in ("profit", "time")
    assert t["pnl_per_contract"] > 0


def test_spread_credit_fraction_falls_with_skew():
    cfg = make_cfg()
    prices = [100.0] * 300
    fracs = [simulate_spread_trade(prices, 20, 35, 0.25, cfg, skew=s,
                                   width_pct=0.02)["credit_fraction"]
             for s in (0.0, 2.0, 4.0)]
    assert fracs[0] > fracs[1] > fracs[2]


# --- simulate_spreads / sweep --------------------------------------------

def test_spread_trades_do_not_overlap():
    cfg = make_cfg()
    prices = [100.0 + 8.0 * math.sin(i / 9.0) + 0.04 * i for i in range(500)]
    trades = simulate_spreads(prices, cfg, skew=0.0, width_pct=0.02)
    assert len(trades) > 5
    for prev, nxt in zip(trades, trades[1:]):
        assert nxt["entry_day"] > prev["exit_day"]


def test_summarize_spreads_credit_stats():
    trades = [
        {"pnl_per_contract": 50.0, "return_on_risk": 0.1, "days_held": 10,
         "exit_reason": "profit", "credit_fraction": 0.30,
         "passes_credit_rule": False},
        {"pnl_per_contract": 80.0, "return_on_risk": 0.16, "days_held": 12,
         "exit_reason": "profit", "credit_fraction": 0.40,
         "passes_credit_rule": True},
    ]
    s = summarize_spreads(trades)
    assert s["avg_credit_fraction"] == pytest.approx(0.35)
    assert s["credit_rule_pass_rate"] == 0.5
    assert s["total_pnl"] == 130.0


def test_sweep_skew_isolates_pricing_effect():
    cfg = make_cfg()
    prices = [100.0 + 8.0 * math.sin(i / 9.0) + 0.04 * i for i in range(500)]
    out = sweep_skew_spreads(prices, cfg, [0.0, 2.0, 4.0], 0.02)
    assert set(out) == {0.0, 2.0, 4.0}
    # Same entry/strike rule across skews: the FIRST trade (same entry
    # day, same strikes) is identical in shape; later entries may shift
    # because exit timing differs by skew — that is the simulation
    # working, not a leak. The pricing claim is the per-trade credit
    # fraction, which must fall monotonically with skew.
    firsts = [out[s]["trades"][0] for s in (0.0, 2.0, 4.0)]
    assert all(f["entry_day"] == firsts[0]["entry_day"] for f in firsts)
    assert all(f["strike"] == firsts[0]["strike"] for f in firsts)
    fracs = [out[s]["summary"]["avg_credit_fraction"] for s in (0.0, 2.0, 4.0)]
    assert fracs[0] > fracs[1] > fracs[2]


def test_spread_report_renders_and_is_json_safe():
    cfg = make_cfg()
    prices = [100.0 + 8.0 * math.sin(i / 9.0) + 0.04 * i for i in range(500)]
    sweeps = {"SPY": sweep_skew_spreads(prices, cfg, [0.0, 2.0], 0.02)}
    payload = build_spread_payload(sweeps, cfg, [0.0, 2.0], 0.02)
    json.dumps(payload)
    md = render_spread_markdown(payload)
    assert "SPY" in md and "Skew" in md and "credit/width" in md
    assert "1 + skew" in md or "skew" in md.lower()
