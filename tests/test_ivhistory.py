"""Unit tests for the observed-IV history store and true IV rank — no network."""

import json
import os

import pytest

from options_screener.ivhistory import (history_path, iv_rank,
                                        load_observations, record_observation)
from screener import attach_iv_rank


def _hist(*ivs, dte=37):
    return [(f"2026-09-{i:02d}", dte, iv) for i, iv in enumerate(ivs, start=1)]


def test_iv_rank_midpoint():
    rank, n = iv_rank(0.30, _hist(0.20, 0.30, 0.40), min_observations=3)
    assert rank == pytest.approx(0.5)
    assert n == 3


def test_iv_rank_clamps_out_of_range():
    rank, _ = iv_rank(0.10, _hist(0.20, 0.30, 0.40), min_observations=3)
    assert rank == 0.0
    rank, _ = iv_rank(0.99, _hist(0.20, 0.30, 0.40), min_observations=3)
    assert rank == 1.0


def test_iv_rank_needs_minimum_observations():
    rank, n = iv_rank(0.30, _hist(0.20, 0.30), min_observations=3)
    assert rank is None
    assert n == 2


def test_iv_rank_flat_history_returns_none():
    rank, n = iv_rank(0.25, _hist(0.25, 0.25, 0.25), min_observations=3)
    assert rank is None
    assert n == 3


def test_record_and_load_roundtrip(tmp_path):
    d = str(tmp_path)
    for day, iv in [("2026-09-01", 0.20), ("2026-09-03", 0.30),
                    ("2026-09-02", 0.25)]:
        record_observation(d, "spy", 37, iv, observed=day)
    obs = load_observations(d, "SPY")
    assert [o[0] for o in obs] == ["2026-09-01", "2026-09-02", "2026-09-03"]
    assert [o[2] for o in obs] == [0.20, 0.25, 0.30]
    assert all(o[1] == 37 for o in obs)


def test_record_same_day_replaces(tmp_path):
    d = str(tmp_path)
    record_observation(d, "qqq", 35, 0.20, observed="2026-09-01")
    record_observation(d, "qqq", 35, 0.22, observed="2026-09-01")
    obs = load_observations(d, "QQQ")
    assert len(obs) == 1
    assert obs[0][2] == pytest.approx(0.22)


def test_load_missing_ticker_returns_empty(tmp_path):
    assert load_observations(str(tmp_path), "DIA") == []


def test_load_skips_corrupt_lines(tmp_path):
    d = str(tmp_path)
    path = history_path(d, "AAPL")
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        f.write('{"date": "2026-09-01", "dte": 37, "iv": 0.25}\n')
        f.write("not json at all\n")
        f.write('{"date": "2026-09-02", "dte": "junk", "iv": 0.26}\n')
        f.write('{"date": "2026-09-03", "dte": 37, "iv": 0.27}\n')
    obs = load_observations(d, "AAPL")
    assert len(obs) == 2
    assert obs[-1][2] == pytest.approx(0.27)


def test_load_dte_band_filter(tmp_path):
    d = str(tmp_path)
    record_observation(d, "nvda", 37, 0.30, observed="2026-09-01")
    record_observation(d, "nvda", 90, 0.35, observed="2026-09-02")
    all_obs = load_observations(d, "NVDA")
    band_obs = load_observations(d, "NVDA", 30, 45)
    assert len(all_obs) == 2
    assert len(band_obs) == 1
    assert band_obs[0][1] == 37


def test_history_path_sanitizes(tmp_path):
    p = history_path(str(tmp_path), "brk.b")
    assert os.path.basename(p) == "BRKB.jsonl"


def _cfg(tmp_path, min_obs=3):
    return {
        "strategy": {"dte_min": 30, "dte_max": 45},
        "iv_rank": {"enabled": True, "history_dir": str(tmp_path),
                    "min_observations": min_obs},
    }


def test_attach_iv_rank_cold_start_uses_proxy(tmp_path):
    best = {"iv": 0.22, "iv_vs_hv_rank": 0.61}
    out = attach_iv_rank("SPY", best, 37, _cfg(tmp_path, min_obs=10))
    assert out["iv_rank_true"] is None
    assert out["iv_rank_obs"] == 1  # today's snapshot was recorded
    assert out["iv_vs_hv_rank"] == 0.61  # proxy untouched


def test_attach_iv_rank_true_after_enough_snapshots(tmp_path):
    d = str(tmp_path)
    for i, iv in enumerate([0.20, 0.25, 0.30]):
        record_observation(d, "SPY", 37, iv, observed=f"2026-09-0{i + 1}")
    best = {"iv": 0.28, "iv_vs_hv_rank": 0.61}
    out = attach_iv_rank("SPY", best, 37, _cfg(tmp_path, min_obs=3))
    # window now: 0.20, 0.25, 0.30, 0.28 -> rank = (0.28-0.20)/(0.30-0.20)
    assert out["iv_rank_true"] == pytest.approx(0.8)
    assert out["iv_rank_obs"] == 4


def test_attach_iv_rank_disabled(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["iv_rank"]["enabled"] = False
    best = {"iv": 0.22}
    out = attach_iv_rank("SPY", best, 37, cfg)
    assert "iv_rank_true" not in out
    assert not os.listdir(tmp_path)  # nothing recorded


def test_iv_rank_cell_render():
    from options_screener.report import _iv_rank_cell
    assert _iv_rank_cell({"iv_rank_true": 0.823, "iv_rank_obs": 42,
                          "iv_vs_hv_rank": 0.61}) == "0.82 (true, 42d)"
    assert _iv_rank_cell({"iv_rank_true": None, "iv_vs_hv_rank": 0.61}) == \
        "0.61 (proxy)"
    assert _iv_rank_cell({"iv_rank_true": None, "iv_vs_hv_rank": None}) == "n/a"
