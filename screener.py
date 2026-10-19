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

from options_screener.alerts import (run_alerts, signal_key, spread_key)
from options_screener.data import (DataError, get_closes, get_expirations,
                                   get_put_chain, pick_expiry)
from options_screener import exits as exits_mod
from options_screener.ivhistory import (iv_rank, load_observations,
                                        record_observation)
from options_screener.report import build_report_payload, write_reports
from options_screener.signals import exit_plan, find_candidates
from options_screener.spreads import find_spreads, spread_exit_plan
from options_screener.vol import hv_range


def attach_iv_rank(ticker: str, best: dict, dte: int, cfg: dict) -> dict:
    """Snapshot today's IV and attach the true IV rank to a signal.

    Records the day's ~30-delta short-put IV into the ticker's observed-IV
    history, then computes the true IV rank (None until enough snapshots
    exist — the report falls back to the labeled IV-vs-HV proxy in that
    case). Returns the signal dict with `iv_rank_true`/`iv_rank_obs` set.
    """
    ivc = cfg.get("iv_rank", {})
    if not ivc.get("enabled", True):
        return best
    strat = cfg["strategy"]
    history_dir = ivc.get("history_dir", "iv_history")
    try:
        record_observation(history_dir, ticker, dte, best["iv"])
        obs = load_observations(history_dir, ticker,
                                strat["dte_min"], strat["dte_max"])
        rank, n = iv_rank(best["iv"], obs,
                          ivc.get("min_observations", 10))
    except OSError:
        return best
    best["iv_rank_true"] = round(rank, 3) if rank is not None else None
    best["iv_rank_obs"] = n
    return best


def screen_ticker(ticker: str, cfg: dict) -> dict:
    """Screen one ticker. Returns {'signal'|'skipped', ...}; raises DataError."""
    closes = get_closes(ticker)
    underlying = closes[-1]

    expirations = get_expirations(ticker)
    strat = cfg["strategy"]
    expiry, dte = pick_expiry(expirations, strat["dte_min"], strat["dte_max"])

    puts = get_put_chain(ticker, expiry)
    hv = hv_range(closes)
    marks = exits_mod.chain_index(puts)

    signals, rejected = find_candidates(puts, underlying, expiry, dte, hv, cfg)
    if not signals:
        why = "; ".join(f"${s:.0f}: {r}" for s, r in rejected) or "no strikes met all gates"
        return {"status": "skipped", "ticker": ticker,
                "reason": f"{expiry} ({dte} DTE): {why}",
                "marks": marks, "underlying": round(underlying, 2)}

    best = signals[0]
    best.update({"ticker": ticker, "underlying": round(underlying, 2),
                 "exit_plan": exit_plan(best, cfg)})
    attach_iv_rank(ticker, best, dte, cfg)

    spread = None
    if cfg.get("spread", {}).get("enabled", False):
        spreads, _ = find_spreads(puts, signals, underlying, expiry, dte, cfg)
        if spreads:
            spread = spreads[0]
            spread.update({"ticker": ticker,
                           "underlying": round(underlying, 2),
                           "exit_plan": spread_exit_plan(spread, cfg)})
    return {"status": "signal", "signal": best, "spread": spread,
            "marks": marks, "underlying": round(underlying, 2)}


def run_paper_tracking(payload: dict, cfg: dict, market: dict,
                       alerts_result: dict | None) -> dict:
    """Open paper positions from newly fired alerts; mark and exit-check all
    open positions against today's chains. Returns a report-ready dict."""
    pcfg = cfg.get("paper", {}) or {}
    as_of = payload["as_of"]
    state_dir = pcfg.get("state_dir", "positions")
    store = exits_mod.load_store(state_dir)

    # Open: only from NEWLY fired alerts (dedupe in alerts.py already
    # prevents re-fire), and never re-open a closed position's key.
    fired = (alerts_result or {}).get("fired", [])
    opened = []
    if pcfg.get("open_from_alerts", True) and fired:
        by_key = {signal_key(s): s for s in payload["signals"]}
        by_key.update({spread_key(sp): sp for sp in payload["spreads"]})
        contracts = pcfg.get("contracts", 1)
        candidates = []
        for it in fired:
            cand = by_key.get(it["key"])
            if cand is None:
                continue
            candidates.append({**cand, "contracts": contracts})
        opened = exits_mod.open_positions(candidates, store,
                                          pcfg.get("max_open", 10))

    # Mark + exit-check every open position against today's chains.
    snapshots = {}
    for pos in store["open"]:
        mkt = market.get(pos["ticker"])
        if mkt is None:
            continue
        snap = exits_mod.build_snapshot(pos, mkt, as_of)
        if snap is not None:
            snapshots[pos["key"]] = snap
    result = exits_mod.update_positions(store, snapshots, cfg["exits"],
                                        as_of=as_of)
    exits_mod.save_store(state_dir, store)

    for p in opened:
        print(f"[paper] opened {p['ticker']} {p['key'].split('|')[-1]} "
              f"({p['kind']}) @ ${p['entry_credit']:.2f}/share credit")
    for c in result["closed_today"]:
        print(f"[paper] closed {c['ticker']} ({c['exit_reason']}): "
              f"${c['realized']:+.2f} per contract")
    return {"opened": opened, "open": store["open"],
            "closed_today": result["closed_today"],
            "state_dir": state_dir}


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily options signal screener")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--alerts", action="store_true",
                    help="run v3 alerting after the report (email/webhook, "
                         "config alerts: section; off by default)")
    ap.add_argument("--alerts-dry-run", action="store_true",
                    help="evaluate alert rules and print what would fire, "
                         "without sending or marking state")
    ap.add_argument("--paper-track", action="store_true",
                    help="paper position tracking: open paper positions from "
                         "newly fired alerts and mark/exit-check all open "
                         "positions (config paper: section; off by default; "
                         "research only, never places orders)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    signals, spreads, skipped, errors = [], [], [], []
    market = {}
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
        market[ticker] = {"marks": res["marks"],
                          "underlying": res["underlying"]}

    signals.sort(key=lambda s: s["annualized_yield"] or 0, reverse=True)
    spreads.sort(key=lambda s: s["roi"], reverse=True)

    alerts_result = None
    if args.alerts or args.alerts_dry_run:
        if not cfg.get("alerts", {}).get("enabled", False):
            print("[alerts] alerts.enabled is false in config — nothing sent "
                  "(see config.yaml; credentials come from env vars only)")
        else:
            alerts_result = run_alerts(
                build_report_payload(
                    {"signals": signals, "spreads": spreads,
                     "skipped": skipped, "errors": errors}, cfg),
                cfg, dry_run=args.alerts_dry_run)
            res = alerts_result
            for it in (res["fired"] or res["would_fire"]):
                print(f"[alert] {it['ticker']} [{it['kind']}] {it['reason']}")
            for ch, err in res["channel_errors"].items():
                print(f"[alert-error] {ch}: {err}", file=sys.stderr)
            if not res["fired"] and not res["would_fire"] and not res["channel_errors"]:
                print("[alerts] no new qualifying signals")

    paper = None
    if args.paper_track:
        if not cfg.get("paper", {}).get("enabled", False):
            print("[paper] paper.enabled is false in config — tracking "
                  "skipped (see config.yaml)")
        else:
            paper = run_paper_tracking(
                build_report_payload(
                    {"signals": signals, "spreads": spreads,
                     "skipped": skipped, "errors": errors}, cfg),
                cfg, market, alerts_result)

    payload = build_report_payload(
        {"signals": signals, "spreads": spreads,
         "skipped": skipped, "errors": errors, "paper": paper}, cfg)
    md_path, json_path = write_reports(payload, args.out_dir)
    print(f"\nwrote {md_path}\nwrote {json_path}")
    print(f"{len(signals)} signal(s), {len(spreads)} spread(s), "
          f"{len(skipped)} skipped, {len(errors)} error(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
