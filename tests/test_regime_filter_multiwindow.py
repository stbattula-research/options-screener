"""Tests for the v18 multi-window regime-filter comparison (research only).

`compare_regime_filter_windows` and the multi-window payload/report
are pure and tested here with synthetic per-window compare rows — no
network. The thin `backtest_ticker_multiwindow` wrapper (one yfinance
fetch + slice per window) is covered by the live smoke run, not by
unit tests.
"""

import json
import os

import yaml

from options_screener.backtest import (
    build_regime_filter_multiwindow_payload,
    compare_regime_filter_windows,
    parse_regime_windows,
    render_regime_filter_multiwindow_markdown)

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _row(ticker, pnl_delta, win_gap=None, dd_gap=None, trade_share=0.5,
         thin=False):
    """A compare_regime_filter-style row with only the fields v18 reads."""
    return {
        "ticker": ticker,
        "plain_trades": 20,
        "plain_win_rate": 0.6,
        "plain_total_pnl": 1000.0,
        "plain_max_drawdown": 2000.0,
        "filtered_trades": 10,
        "filtered_win_rate": 0.65,
        "filtered_total_pnl": 1000.0 + (pnl_delta or 0),
        "filtered_max_drawdown": 2000.0 + (dd_gap or 0),
        "skipped_entries": 10,
        "trade_share": trade_share,
        "pnl_delta": pnl_delta,
        "win_gap": win_gap,
        "dd_gap": dd_gap,
        "thin": thin,
    }


def _window(rows, offset):
    return {
        "rows": rows,
        "summary": {
            "tickers_compared": len(rows),
            "readable_tickers": sum(1 for r in rows if not r["thin"]),
            "filtered_wins_pnl": sum(
                1 for r in rows
                if not r["thin"] and r["pnl_delta"] is not None
                and r["pnl_delta"] > 0),
            "filtered_wins_win_rate": 0,
            "median_pnl_delta": None,
            "median_win_gap": None,
            "median_dd_gap": None,
            "median_trade_share": 0.5,
        },
        "window": {"start": "2020-11-18", "end": "2022-11-15",
                   "years": 2, "offset_years": offset},
    }


def test_parse_regime_windows_valid_and_sorted():
    assert parse_regime_windows("0,2,4") == [0, 2, 4]
    assert parse_regime_windows("4,0,2") == [0, 2, 4]
    assert parse_regime_windows("0,2") == [0, 2]


def test_parse_regime_windows_rejects_bad_values():
    for bad in ("0", "0,2,2", "-1,2", "a,b", "", None, "0, 2, x"):
        try:
            parse_regime_windows(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have raised ValueError")


def test_median_delta_and_wins_across_windows():
    windowed = {
        0: _window([_row("AAA", 100.0, 0.05, -100.0)], 0),
        2: _window([_row("AAA", -50.0, -0.02, -50.0)], 2),
        4: _window([_row("AAA", 200.0, 0.10, -200.0)], 4),
    }
    out = compare_regime_filter_windows(windowed)
    row = out["rows"][0]
    assert row["ticker"] == "AAA"
    assert row["median_pnl_delta"] == 100.0
    assert row["median_win_gap"] == 0.05
    assert row["median_dd_gap"] == -100.0
    assert row["wins_pnl"] == 2
    assert row["windows_readable"] == 3
    assert row["verdict"] == "mixed"
    assert row["dd_always_lower"] is True


def test_consistent_winner_and_loser_verdicts():
    windowed = {
        off: _window([_row("BBB", 100.0 + off, 0.05, -10.0),
                      _row("CCC", -100.0 - off, -0.05, 50.0)], off)
        for off in (0, 2, 4)
    }
    out = compare_regime_filter_windows(windowed)
    by_ticker = {r["ticker"]: r for r in out["rows"]}
    assert by_ticker["BBB"]["verdict"] == "filter won every readable window"
    assert by_ticker["CCC"]["verdict"] == "filter lost every readable window"
    assert out["summary"]["consistent_winners"] == 1
    assert out["summary"]["consistent_losers"] == 1
    assert out["summary"]["mixed"] == 0


def test_thin_window_never_counted():
    windowed = {
        0: _window([_row("DDD", 100.0, 0.05, -10.0)], 0),
        2: _window([_row("DDD", 200.0, 0.05, -10.0)], 2),
        # the thin window's huge delta must not move the median
        4: _window([_row("DDD", 9999.0, 0.99, -10.0, thin=True)], 4),
    }
    out = compare_regime_filter_windows(windowed)
    row = out["rows"][0]
    assert row["windows_readable"] == 2
    assert row["median_pnl_delta"] == 150.0
    assert row["wins_pnl"] == 2
    assert row["per_window"][4]["thin"] is True


def test_missing_ticker_in_one_window():
    windowed = {
        0: _window([_row("EEE", 100.0, 0.05, -10.0)], 0),
        2: _window([_row("EEE", 300.0, 0.05, -10.0)], 2),
        4: _window([], 4),
    }
    out = compare_regime_filter_windows(windowed)
    row = out["rows"][0]
    assert row["per_window"][4] is None
    assert row["median_pnl_delta"] == 200.0
    assert row["windows_readable"] == 2
    assert out["summary"]["tickers_in_all_windows"] == 0


def test_dd_always_lower_needs_every_readable_window():
    windowed = {
        0: _window([_row("FFF", 10.0, 0.0, -100.0),
                    _row("GGG", 10.0, 0.0, -100.0)], 0),
        2: _window([_row("FFF", 10.0, 0.0, -50.0),
                    _row("GGG", 10.0, 0.0, 50.0)], 2),
    }
    out = compare_regime_filter_windows(windowed)
    by_ticker = {r["ticker"]: r for r in out["rows"]}
    assert by_ticker["FFF"]["dd_always_lower"] is True
    assert by_ticker["GGG"]["dd_always_lower"] is False
    assert out["summary"]["dd_always_lower_count"] == 1


def test_thin_everywhere_has_no_median():
    windowed = {
        off: _window([_row("HHH", 100.0, 0.05, -10.0, thin=True)], off)
        for off in (0, 2)
    }
    out = compare_regime_filter_windows(windowed)
    row = out["rows"][0]
    assert row["verdict"] == "thin everywhere"
    assert row["median_pnl_delta"] is None
    assert row["windows_readable"] == 0
    assert row["dd_always_lower"] is False


def test_rows_ranked_by_median_delta_none_last():
    windowed = {
        0: _window([_row("AAA", 10.0), _row("BBB", 500.0),
                    _row("CCC", 50.0, thin=True)], 0),
        2: _window([_row("AAA", 20.0), _row("BBB", 400.0),
                    _row("CCC", 60.0, thin=True)], 2),
    }
    out = compare_regime_filter_windows(windowed)
    assert [r["ticker"] for r in out["rows"]] == ["BBB", "AAA", "CCC"]


def test_summary_median_of_medians():
    windowed = {
        0: _window([_row("AAA", 100.0), _row("BBB", 300.0)], 0),
        2: _window([_row("AAA", 200.0), _row("BBB", 400.0)], 2),
    }
    out = compare_regime_filter_windows(windowed)
    # per-ticker medians: AAA 150, BBB 350 -> median of medians 250
    assert out["summary"]["median_of_median_pnl_delta"] == 250.0
    assert out["summary"]["windows_compared"] == 2
    assert out["summary"]["tickers_compared"] == 2
    assert out["summary"]["tickers_in_all_windows"] == 2


def test_payload_json_and_markdown():
    windowed = {
        0: _window([_row("AAA", 100.0, 0.05, -100.0),
                    _row("BBB", -200.0, -0.05, 50.0)], 0),
        2: _window([_row("AAA", 200.0, 0.05, -100.0),
                    _row("BBB", -100.0, -0.05, 50.0)], 2),
        4: _window([_row("AAA", 300.0, 0.05, -100.0)], 4),
    }
    payload = build_regime_filter_multiwindow_payload(
        windowed, load_cfg(), 252)
    json.dumps(payload)  # must be JSON-serializable
    assert payload["mode"].startswith("Regime-aware forward entry filter")
    assert payload["offsets_years"] == [0, 2, 4]
    md = render_regime_filter_multiwindow_markdown(payload)
    assert "Across 3 Windows" in md
    assert "AAA" in md and "BBB" in md
    assert "filter won every readable window" in md
    assert "filter lost every readable window" in md
    assert "still not a rule" in md
    assert "—" in md  # BBB absent from the -4y window renders a dash
    # the thin-window legend is always printed; with no thin rows in
    # this fixture it is the only dagger in the report
    assert md.count("†") == 1
