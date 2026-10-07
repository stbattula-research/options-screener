"""Unit tests for report rendering — no network access."""

import json

from options_screener.report import (DISCLAIMER, build_report_payload,
                                     render_markdown)


def _payload():
    signal = {
        "ticker": "SPY", "underlying": 500.00, "strike": 485.00,
        "expiry": "2026-11-13", "dte": 39, "delta": -0.29, "iv": 0.18,
        "iv_vs_hv_rank": 0.62, "bid": 2.10, "ask": 2.25, "mid": 2.18,
        "spread_pct": 0.0688, "volume": 1200, "open_interest": 5400,
        "premium_pct": 0.0045, "annualized_yield": 0.0421,
        "max_profit_per_share": 2.18, "breakeven": 482.82,
        "exit_plan": {"profit_target": "Close at 50% of max profit (~$1.09/share)",
                      "time_exit": "Exit or roll at 21 DTE",
                      "loss_manage": "Manage the trade if the loss reaches 2x premium received (~$4.36/share)"},
    }
    return build_report_payload(
        {"signals": [signal],
         "skipped": [{"ticker": "XYZ", "reason": "no strikes met all gates"}],
         "errors": []},
        {"strategy": {"dte_min": 30, "dte_max": 45, "target_delta": -0.30},
         "liquidity": {"min_open_interest": 500}},
        as_of="2026-10-05")


def test_payload_schema_json_serializable():
    p = _payload()
    json.dumps(p)  # must not raise
    assert p["signals"][0]["ticker"] == "SPY"
    assert "disclaimer" in p


def test_markdown_contains_signal_table():
    md = render_markdown(_payload())
    assert "# Options premium-selling signals" in md
    assert "| SPY |" in md
    assert "485.00" in md


def test_markdown_contains_exit_plan_and_skipped():
    md = render_markdown(_payload())
    assert "Exit plan" in md
    assert "21 DTE" in md
    assert "XYZ" in md


def test_disclaimer_present_in_both():
    p = _payload()
    assert "not financial advice" in p["disclaimer"]
    assert "not financial advice" in render_markdown(p)
    assert "Paper-trade" in DISCLAIMER


def test_empty_signals_renders_gracefully():
    p = build_report_payload(
        {"signals": [], "skipped": [], "errors": ["SPY: boom"]},
        {"strategy": {"dte_min": 30, "dte_max": 45, "target_delta": -0.30},
         "liquidity": {}},
        as_of="2026-10-05")
    md = render_markdown(p)
    assert "No signals passed all filters today." in md
    assert "SPY: boom" in md
