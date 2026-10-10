"""Unit tests for bull put spread construction — no network access."""

import pandas as pd
import pytest

from options_screener.report import build_report_payload, render_markdown
from options_screener.spreads import find_spreads, spread_exit_plan

CFG = {
    "strategy": {"dte_min": 30, "dte_max": 45, "target_delta": -0.30,
                 "delta_tolerance": 0.07, "min_premium_pct": 0.01},
    "liquidity": {"min_open_interest": 500, "min_volume": 100,
                  "max_spread_pct": 0.10},
    "risk_free_rate": 0.04,
    "spread": {"enabled": True, "min_width_pct": 0.02, "max_width_pct": 0.06,
               "min_credit_fraction": 0.33},
    "exits": {"profit_target_pct": 0.50, "manage_dte": 21,
              "loss_manage_multiple": 2.0},
}

# Short candidate: 100-strike put, mid 2.05 (as find_candidates would emit).
SHORT = [{"strike": 100.0, "mid": 2.05, "delta": -0.30}]


def chain(*rows):
    return pd.DataFrame(rows)


def leg(strike, bid, ask, oi=800, vol=200, iv=0.32):
    return {"strike": strike, "bid": bid, "ask": ask,
            "openInterest": oi, "volume": vol, "impliedVolatility": iv}


def test_picks_highest_roi_long_leg():
    puts = chain(
        leg(97, 0.88, 0.92),   # width 3, credit 1.15 -> roi 0.6216
        leg(95, 0.29, 0.31),   # width 5, credit 1.75 -> roi 0.5385
    )
    spreads, rejected = find_spreads(puts, SHORT, 102.5, "2026-11-13", 35, CFG)
    assert rejected == []
    assert len(spreads) == 1
    sp = spreads[0]
    assert sp["long_strike"] == 97.0
    assert sp["width"] == 3.0
    assert sp["net_credit"] == pytest.approx(1.15)
    assert sp["max_loss"] == pytest.approx(1.85)
    assert sp["max_profit"] == pytest.approx(1.15)
    assert sp["breakeven"] == pytest.approx(98.85)
    assert sp["roi"] == pytest.approx(round(1.15 / 1.85, 4))
    assert sp["annualized_roi"] == pytest.approx(round((1.15 / 1.85) * 365 / 35, 4))
    assert sp["long_delta"] is not None and sp["long_delta"] < 0


def test_width_bounds_reject():
    puts = chain(
        leg(93, 0.10, 0.20),    # width 7 > 6% of 100: too wide
        leg(98.5, 1.30, 1.40),  # width 1.5 < 2% of 100: too tight
    )
    spreads, rejected = find_spreads(puts, SHORT, 102.5, "2026-11-13", 35, CFG)
    assert spreads == []
    assert rejected == [(100.0, "no long leg met spread rules")]


def test_credit_fraction_rule_rejects_thin_credit():
    # width 5 needs credit >= 5 * 0.33 = 1.65; mid 0.90 leaves only 1.15
    puts = chain(leg(95, 0.85, 0.95))
    spreads, rejected = find_spreads(puts, SHORT, 102.5, "2026-11-13", 35, CFG)
    assert spreads == []
    assert len(rejected) == 1


def test_long_leg_must_pass_liquidity():
    # credit 1.75 clears the 1.65 bar and the quote is tight, but OI 10 kills it
    puts = chain(leg(95, 0.29, 0.31, oi=10))
    spreads, rejected = find_spreads(puts, SHORT, 102.5, "2026-11-13", 35, CFG)
    assert spreads == []
    assert len(rejected) == 1


def test_wide_spread_long_leg_rejected():
    # 10% spread on the long leg exceeds the liquidity gate
    puts = chain(leg(95, 0.20, 0.40))
    spreads, _ = find_spreads(puts, SHORT, 102.5, "2026-11-13", 35, CFG)
    assert spreads == []


def test_spread_exit_plan():
    sp = {"max_profit": 1.15, "net_credit": 1.15, "max_loss": 1.85}
    e = spread_exit_plan(sp, CFG)
    assert "50%" in e["profit_target"] and "$0.57" in e["profit_target"]
    assert "21 DTE" in e["time_exit"]
    assert "2x net credit" in e["loss_manage"]
    assert "$1.85" in e["loss_manage"]  # capped max loss stated


def test_spread_ranked_by_roi_desc():
    short2 = [{"strike": 100.0, "mid": 2.05, "delta": -0.30},
              {"strike": 90.0, "mid": 1.20, "delta": -0.28}]
    puts = chain(
        leg(97, 0.88, 0.92),      # pairs with 100: roi 0.6216
        leg(87, 0.195, 0.215),    # pairs with 90: width 3, credit 0.995 -> roi 0.496
    )
    spreads, _ = find_spreads(puts, short2, 102.5, "2026-11-13", 35, CFG)
    assert len(spreads) == 2
    assert spreads[0]["roi"] >= spreads[1]["roi"]
    assert spreads[0]["short_strike"] == 100.0


def test_report_renders_spreads():
    spread = {
        "ticker": "SPY", "underlying": 505.00,
        "short_strike": 490.0, "short_mid": 2.05, "short_delta": -0.30,
        "long_strike": 485.0, "long_delta": -0.18,
        "long_bid": 0.85, "long_ask": 0.95, "long_mid": 0.90,
        "width": 5.0, "net_credit": 1.15, "max_loss": 3.85,
        "max_profit": 1.15, "breakeven": 488.85,
        "roi": 0.2987, "annualized_roi": 2.79,
        "expiry": "2026-11-13", "dte": 39,
        "exit_plan": spread_exit_plan(
            {"max_profit": 1.15, "net_credit": 1.15, "max_loss": 3.85}, CFG),
    }
    p = build_report_payload(
        {"signals": [], "spreads": [spread], "skipped": [], "errors": []},
        CFG, as_of="2026-10-10")
    md = render_markdown(p)
    assert "Bull put spreads" in md
    assert "$490.00/$485.00" in md
    assert "29.9%" in md  # roi rendered
    json_ok = p["spreads"][0]["ticker"] == "SPY"
    assert json_ok
