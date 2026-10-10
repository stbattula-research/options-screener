#!/usr/bin/env python3
"""Daily options premium-selling signal screener (research only).

Scans a watchlist of liquid underlyings, pulls put chains via yfinance,
and emits concrete cash-secured-put candidates: ~30 delta, 30-45 DTE,
passing liquidity filters. Output: Markdown + JSON daily report.

Signals only. This tool never places orders and holds no broker credentials.

Usage:
    python screener.py [--config config.yaml] [--out-dir reports/]
"""

import argparse
import sys
import traceback

import yaml

from options_screener.data import (DataError, get_closes, get_expirations,
                                   get_put_chain, pick_expiry)
from options_screener.report import build_report_payload, write_reports
from options_screener.signals import exit_plan, find_candidates
from options_screener.spreads import find_spreads, spread_exit_plan
from options_screener.vol import hv_range


def screen_ticker(ticker: str, cfg: dict) -> dict:
    """Screen one ticker. Returns {'signal'|'skipped', ...}; raises DataError."""
    closes = get_closes(ticker)
    underlying = closes[-1]

    expirations = get_expirations(ticker)
    strat = cfg["strategy"]
    expiry, dte = pick_expiry(expirations, strat["dte_min"], strat["dte_max"])

    puts = get_put_chain(ticker, expiry)
    hv = hv_range(closes)

    signals, rejected = find_candidates(puts, underlying, expiry, dte, hv, cfg)
    if not signals:
        why = "; ".join(f"${s:.0f}: {r}" for s, r in rejected) or "no strikes met all gates"
        return {"status": "skipped", "ticker": ticker,
                "reason": f"{expiry} ({dte} DTE): {why}"}

    best = signals[0]
    best.update({"ticker": ticker, "underlying": round(underlying, 2),
                 "exit_plan": exit_plan(best, cfg)})

    spread = None
    if cfg.get("spread", {}).get("enabled", False):
        spreads, _ = find_spreads(puts, signals, underlying, expiry, dte, cfg)
        if spreads:
            spread = spreads[0]
            spread.update({"ticker": ticker,
                           "underlying": round(underlying, 2),
                           "exit_plan": spread_exit_plan(spread, cfg)})
    return {"status": "signal", "signal": best, "spread": spread}


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily options signal screener")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    signals, spreads, skipped, errors = [], [], [], []
    for ticker in cfg["watchlist"]:
        try:
            res = screen_ticker(ticker, cfg)
        except DataError as exc:
            errors.append(str(exc))
            print(f"[error] {exc}", file=sys.stderr)
            continue
        except Exception:  # never let one ticker kill the run
            errors.append(f"{ticker}: unexpected error\n{traceback.format_exc(limit=3)}")
            continue
        if res["status"] == "signal":
            signals.append(res["signal"])
            print(f"[signal] {ticker}: ${res['signal']['strike']:.0f} put "
                  f"{res['signal']['expiry']} Δ={res['signal']['delta']:.2f}")
            if res.get("spread"):
                spreads.append(res["spread"])
                print(f"[spread] {ticker}: ${res['spread']['short_strike']:.0f}/"
                      f"${res['spread']['long_strike']:.0f} "
                      f"credit ${res['spread']['net_credit']:.2f} "
                      f"ROI {res['spread']['roi']:.0%}")
        else:
            skipped.append({"ticker": res["ticker"], "reason": res["reason"]})
            print(f"[skip] {ticker}: {res['reason'][:90]}")

    signals.sort(key=lambda s: s["annualized_yield"] or 0, reverse=True)
    spreads.sort(key=lambda s: s["roi"], reverse=True)
    payload = build_report_payload(
        {"signals": signals, "spreads": spreads,
         "skipped": skipped, "errors": errors}, cfg)
    md_path, json_path = write_reports(payload, args.out_dir)
    print(f"\nwrote {md_path}\nwrote {json_path}")
    print(f"{len(signals)} signal(s), {len(spreads)} spread(s), "
          f"{len(skipped)} skipped, {len(errors)} error(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
