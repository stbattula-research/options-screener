"""Tests for the v5 exit-rule sensitivity sweep (research only).

All tests use a fixed-seed synthetic price series — no network, no fixtures.
"""

import json
import math
import os
import random

import pytest
import yaml

from options_screener.backtest import (DEFAULT_SWEEP_GRID, default_sweep_key,
                                       rank_sweep, render_sweep_markdown,
                                       build_sweep_payload, simulate,
                                       simulate_trade, sweep_exit_rules,
                                       sweep_key)

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


@pytest.fixture(scope="module")
def cfg():
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


@pytest.fixture(scope="module")
def prices():
    return gbmm_walk(504)  # ~2 trading years


# ---------------------------------------------------------------- exit overrides

def test_exit_overrides_change_behavior(prices, cfg):
    low_pt = simulate(prices, cfg, exits={"profit_target_pct": 0.25})
    high_pt = simulate(prices, cfg, exits={"profit_target_pct": 0.75})
    # a lower profit target closes winners faster: shorter average hold and
    # a higher share of profit exits than a high target on the same series
    avg_low = sum(t["days_held"] for t in low_pt) / len(low_pt)
    avg_high = sum(t["days_held"] for t in high_pt) / len(high_pt)
    assert avg_low < avg_high
    prof_low = sum(1 for t in low_pt if t["exit_reason"] == "profit") / len(low_pt)
    prof_high = sum(1 for t in high_pt if t["exit_reason"] == "profit") / len(high_pt)
    assert prof_low > prof_high


def test_loss_multiple_changes_stop_behavior(prices, cfg):
    tight = simulate(prices, cfg, exits={"loss_manage_multiple": 1.5})
    wide = simulate(prices, cfg, exits={"loss_manage_multiple": 3.0})
    loss_tight = sum(1 for t in tight if t["exit_reason"] == "loss")
    loss_wide = sum(1 for t in wide if t["exit_reason"] == "loss")
    assert loss_tight >= loss_wide  # wider stop -> fewer loss-managed exits
    worst_tight = min(t["pnl_per_contract"] for t in tight)
    worst_wide = min(t["pnl_per_contract"] for t in wide)
    assert worst_tight >= worst_wide  # wider stop -> worse worst trade


def test_manage_dte_changes_time_exits(prices, cfg):
    early = simulate(prices, cfg, exits={"manage_dte": 14})
    late = simulate(prices, cfg, exits={"manage_dte": 21})
    time_early = sum(1 for t in early if t["exit_reason"] == "time")
    time_late = sum(1 for t in late if t["exit_reason"] == "time")
    # 21-DTE management bites before 14-DTE management does
    assert time_late >= time_early


def test_no_override_matches_default(prices, cfg):
    plain = simulate(prices, cfg)
    explicit = simulate(prices, cfg, exits=dict(cfg["exits"]))
    assert [t["pnl_per_contract"] for t in plain] == [
        t["pnl_per_contract"] for t in explicit]


def test_overrides_do_not_mutate_cfg(prices, cfg):
    before = dict(cfg["exits"])
    simulate(prices, cfg, exits={"profit_target_pct": 0.10})
    simulate_trade(prices, 21, 35, 0.2, cfg, exits={"manage_dte": 5})
    assert cfg["exits"] == before


# ---------------------------------------------------------------- sweep engine

def test_sweep_covers_full_grid(prices, cfg):
    grid = {"profit_targets": [0.25, 0.75], "loss_multiples": [1.5, 3.0],
            "manage_dtes": [14]}
    out = sweep_exit_rules(prices, cfg, grid)
    assert len(out) == 2 * 2 * 1
    assert set(out) == {sweep_key(0.25, 1.5, 14), sweep_key(0.25, 3.0, 14),
                        sweep_key(0.75, 1.5, 14), sweep_key(0.75, 3.0, 14)}
    for res in out.values():
        assert "trades" in res and "summary" in res and "rules" in res


def test_sweep_default_combo_present(prices, cfg):
    out = sweep_exit_rules(prices, cfg, DEFAULT_SWEEP_GRID)
    assert default_sweep_key(cfg) in out
    ex = cfg["exits"]
    assert default_sweep_key(cfg) == sweep_key(ex["profit_target_pct"],
                                              ex["loss_manage_multiple"],
                                              ex["manage_dte"])


def test_sweep_key_format():
    assert sweep_key(0.5, 2.0, 21) == "profit=0.50,loss=2.0x,dte=21"


def test_rank_sweep_orders_by_pnl(prices, cfg):
    out = sweep_exit_rules(prices, cfg, DEFAULT_SWEEP_GRID)
    rows = rank_sweep(out, default_sweep_key(cfg))
    assert len(rows) == len(out)
    pnls = [r["summary"]["total_pnl"] for r in rows]
    assert pnls == sorted(pnls, reverse=True)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


def test_rank_sweep_flags_default_and_deltas(prices, cfg):
    out = sweep_exit_rules(prices, cfg, DEFAULT_SWEEP_GRID)
    default_key = default_sweep_key(cfg)
    rows = rank_sweep(out, default_key)
    defaults = [r for r in rows if r["is_default"]]
    assert len(defaults) == 1
    d = defaults[0]
    assert d["delta_vs_default"] == 0.0
    assert all(r["delta_vs_default"] == round(
        r["summary"]["total_pnl"] - out[default_key]["summary"]["total_pnl"], 2)
               for r in rows)


def test_sweep_on_empty_series(cfg):
    out = sweep_exit_rules([100.0] * 10, cfg, {"profit_targets": [0.5],
                                               "loss_multiples": [2.0],
                                               "manage_dtes": [21]})
    s = list(out.values())[0]["summary"]
    assert s["trades"] == 0
    assert s["win_rate"] is None
    ranked = rank_sweep(out, default_sweep_key(cfg))
    assert len(ranked) == 1  # zero-trade combo still ranks cleanly


# ---------------------------------------------------------------- payload/report

def test_sweep_payload_json_serializable(prices, cfg):
    out = sweep_exit_rules(prices, cfg, DEFAULT_SWEEP_GRID)
    rows = rank_sweep(out, default_sweep_key(cfg))
    payload = build_sweep_payload({"SPY": rows}, cfg, DEFAULT_SWEEP_GRID)
    json.dumps(payload)  # must not raise
    assert payload["tickers"]["SPY"][0]["rank"] == 1
    assert any("robust" in a for a in payload["assumptions"])


def test_sweep_markdown_marks_default_and_ranks(prices, cfg):
    out = sweep_exit_rules(prices, cfg, DEFAULT_SWEEP_GRID)
    rows = rank_sweep(out, default_sweep_key(cfg))
    md = render_sweep_markdown(build_sweep_payload({"SPY": rows}, cfg,
                                                   DEFAULT_SWEEP_GRID))
    assert "★ default" in md
    assert "| 1 |" in md  # rank column present
    assert "How to read this" in md
    assert "paper-trade before" in md  # risk disclaimer footer
