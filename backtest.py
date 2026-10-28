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
                       [--regimes] [--spreads] [--spread-widths]
"""

import argparse
import json
import sys
import traceback
from datetime import date

import yaml

from options_screener.backtest import (DEFAULT_SPREAD_BACKTEST,
                                       DEFAULT_SPREAD_WIDTH_SWEEP,
                                       DEFAULT_SWEEP_GRID,
                                       build_backtest_payload,
                                       build_regime_payload,
                                       build_spread_payload,
                                       build_sweep_payload,
                                       build_width_payload,
                                       default_sweep_key, rank_sweep,
                                       render_backtest_markdown,
                                       render_regime_markdown,
                                       render_spread_markdown,
                                       render_sweep_markdown,
                                       render_width_markdown,
                                       simulate, split_by_iv_regime,
                                       summarize, sweep_exit_rules,
                                       sweep_skew_spreads,
                                       sweep_width_spreads)
from options_screener.data import DataError, get_closes


def backtest_ticker(ticker, cfg, years, sweep=False, regimes=False,
                    spreads=False, spread_widths=False):
    closes = get_closes(ticker, period=f"{years}y")
    if spread_widths:
        sw = cfg.get("spread_width_sweep") or DEFAULT_SPREAD_WIDTH_SWEEP
        return sweep_width_spreads(closes, cfg, sw["widths"], sw["skew"])
    if spreads:
        sb = cfg.get("spread_backtest") or DEFAULT_SPREAD_BACKTEST
        return sweep_skew_spreads(closes, cfg, sb["skews"], sb["width_pct"])
    if regimes:
        trades = simulate(closes, cfg)
        return split_by_iv_regime(trades)
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
    ap.add_argument("--regimes", action="store_true",
                    help="split the plain backtest into low/high entry-IV "
                         "regimes at each ticker's median entry IV "
                         "(descriptive only, not a live entry filter)")
    ap.add_argument("--spreads", action="store_true",
                    help="run the skew-aware bull-put-spread backtest: one "
                         "spread simulation per skew value in "
                         "spread_backtest: (same entries, only leg pricing "
                         "changes), to test whether skew rescues the "
                         "1/3-width credit rule")
    ap.add_argument("--spread-widths", action="store_true",
                    help="run the spread-width sweep: one spread simulation "
                         "per wing width in spread_width_sweep: at a fixed "
                         "skew, to measure where the 1/3-width credit rule "
                         "starts passing and what passing costs in P&L")
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
            res = backtest_ticker(ticker, cfg, args.years, sweep=args.sweep,
                                  regimes=args.regimes, spreads=args.spreads,
                                  spread_widths=args.spread_widths)
            results[ticker] = res
            if args.spread_widths:
                sw = (cfg.get("spread_width_sweep")
                      or DEFAULT_SPREAD_WIDTH_SWEEP)
                narrow = res[sw["widths"][0]]["summary"]
                wide = res[sw["widths"][-1]]["summary"]
                print(f"[{ticker}] spread widths: {sw['widths'][0]:.0%} "
                      f"avg credit/width {narrow['avg_credit_fraction']} "
                      f"vs {sw['widths'][-1]:.0%} "
                      f"{wide['avg_credit_fraction']}")
            elif args.spreads:
                flat_skew = (cfg.get("spread_backtest")
                             or DEFAULT_SPREAD_BACKTEST)["skews"][0]
                flat = res[flat_skew]["summary"]
                print(f"[{ticker}] spreads: {flat['trades']} trades at skew "
                      f"{flat_skew}, avg credit/width "
                      f"{flat['avg_credit_fraction']}, pass rate "
                      f"{flat['credit_rule_pass_rate']}")
            elif args.regimes:
                med = res["median_iv"]
                lo, hi = res["low"]["summary"], res["high"]["summary"]
                print(f"[{ticker}] median IV {med}, low: {lo['trades']} trades "
                      f"${lo['total_pnl']:+,.2f}, high: {hi['trades']} trades "
                      f"${hi['total_pnl']:+,.2f}")
            elif args.sweep:
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

    if args.spread_widths:
        sw = cfg.get("spread_width_sweep") or DEFAULT_SPREAD_WIDTH_SWEEP
        payload = build_width_payload(results, cfg, sw["widths"], sw["skew"])
        md_path, json_path = write_reports(payload, args.out_dir,
                                           "backtest-spread-widths",
                                           render_width_markdown)
    elif args.spreads:
        sb = cfg.get("spread_backtest") or DEFAULT_SPREAD_BACKTEST
        payload = build_spread_payload(results, cfg, sb["skews"],
                                       sb["width_pct"])
        md_path, json_path = write_reports(payload, args.out_dir,
                                           "backtest-spreads",
                                           render_spread_markdown)
    elif args.regimes:
        # results here are per-ticker splits; rebuild plain trade lists for
        # the payload builder (it splits them itself, identically).
        plain = {t: {"trades": sorted(
                                [tr for r in (res["low"], res["high"])
                                 for tr in r["trades"]],
                                key=lambda tr: tr["entry_day"]),
                      "summary": None}
                 for t, res in results.items()}
        # summaries are recomputed inside the split; the None above is unused
        for t, res in results.items():
            plain[t]["summary"] = summarize(plain[t]["trades"])
        payload = build_regime_payload(plain, cfg)
        md_path, json_path = write_reports(payload, args.out_dir, "backtest-regimes",
                                           render_regime_markdown)
    elif args.sweep:
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
