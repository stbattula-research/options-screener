#!/usr/bin/env python3
"""Backtest the options-screener premium-selling strategy (research only).

Simulates ~30-delta cash-secured puts over historical daily closes and
writes a Markdown + JSON report. Model-based: premiums are priced with
Black-Scholes against an IV proxy, NOT from historical option chains —
results are illustrative research, not expected returns.

Signals only. This tool never places orders and holds no broker credentials.

Usage:
    python backtest.py [--config config.yaml] [--out-dir reports]
                       [--ticker SPY | --all] [--years 3] [--sweep]
"""

import argparse
import json
import sys
import traceback
from datetime import date

import yaml

from options_screener.backtest import (DEFAULT_SWEEP_GRID,
                                       build_backtest_payload,
                                       build_sweep_payload,
                                       default_sweep_key, rank_sweep,
                                       render_backtest_markdown,
                                       render_sweep_markdown,
                                       simulate, summarize, sweep_exit_rules)
from options_screener.data import DataError, get_closes


def backtest_ticker(ticker, cfg, years, sweep=False):
    closes = get_closes(ticker, period=f"{years}y")
    if not sweep:
        trades = simulate(closes, cfg)
        return {"trades": trades, "summary": summarize(trades)}
    grid = cfg.get("backtest_sweep") or DEFAULT_SWEEP_GRID
    results = sweep_exit_rules(closes, cfg, grid)
    return rank_sweep(results, default_sweep_key(cfg))


def write_reports(payload, out_dir, prefix, render):
    import os
    os.makedirs(out_dir, exist_ok=True)
    stamp = date.today().isoformat()
    md_path = os.path.join(out_dir, f"{prefix}-{stamp}.md")
    json_path = os.path.join(out_dir, f"{prefix}-{stamp}.json")
    with open(md_path, "w") as f:
        f.write(render(payload))
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return md_path, json_path


def write_backtest_reports(payload, out_dir):
    return write_reports(payload, out_dir, "backtest", render_backtest_markdown)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest the premium-selling strategy")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--all", action="store_true",
                    help="backtest every ticker in the config watchlist")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--sweep", action="store_true",
                    help="run the exit-rule sensitivity sweep instead of the "
                         "plain backtest (grid from backtest_sweep: in config)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.all:
        tickers = cfg["watchlist"]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [cfg["watchlist"][0]]

    results, errors = {}, []
    for ticker in tickers:
        try:
            res = backtest_ticker(ticker, cfg, args.years, sweep=args.sweep)
            results[ticker] = res
            if args.sweep:
                best = res[0]
                default = next(r for r in res if r["is_default"])
                print(f"[{ticker}] best {best['key']} "
                      f"${best['summary']['total_pnl']:+,.2f}; default ranks "
                      f"#{default['rank']}/{len(res)} "
                      f"(${default['summary']['total_pnl']:+,.2f})")
            else:
                s = res["summary"]
                print(f"[{ticker}] {s['trades']} trades, win {s['win_rate']}, "
                      f"P&L ${s['total_pnl']:,.2f}, max DD ${s['max_drawdown']:,.2f}")
        except DataError as exc:
            errors.append(str(exc))
            print(f"[error] {exc}", file=sys.stderr)
        except Exception:  # never let one ticker kill the run
            errors.append(f"{ticker}: unexpected error\n{traceback.format_exc(limit=3)}")

    if args.sweep:
        grid = cfg.get("backtest_sweep") or DEFAULT_SWEEP_GRID
        payload = build_sweep_payload(results, cfg, grid)
        md_path, json_path = write_reports(payload, args.out_dir, "backtest-sweep",
                                           render_sweep_markdown)
    else:
        payload = build_backtest_payload(results, cfg)
        md_path, json_path = write_backtest_reports(payload, args.out_dir)
    print(f"\nwrote {md_path}\nwrote {json_path}")
    if errors:
        print(f"{len(errors)} error(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
