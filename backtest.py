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
                       [--regimes] [--regime-compare] [--spreads]
                       [--spread-widths] [--earnings-calendar calendar.json]
    Combine --spreads with --earnings-calendar for the v11
    earnings-aware spread backtest (plain vs gated per skew).
    Combine --spreads with --dividend-calendar for the v19
    dividend-aware spread backtest: same shape, but it tests the
    opt-in dividend exclusion (dividends.exclude: true) — with the
    shipped flag-only default the gate is inactive and the filtered
    run equals the plain run.
    --regime-compare runs the v6 split per ticker and compares the
    high-minus-low gaps across tickers (v15).
    --regime-filter runs the v16 forward entry filter: enter only when
    today's entry IV is above its trailing 252-day median, plain vs
    filtered side by side over the same warmup start.
    --offset-years N (with --regime-filter only) re-runs that same
    filter on an earlier window: the N-year window ending N calendar
    years ago (v17 re-test of the v16 hypothesis on a different period).
    --regime-windows 0,2,4 (with --regime-filter only) runs the filter
    on several non-overlapping windows at once and aggregates per
    ticker (v18: the third window v16/v17 left open, counted rather
    than eyeballed). One fetch per ticker; windows are sliced from the
    same series.
    --structure-compare runs the v1 CSP and the v2 bull-put spread over
    the SAME entry-attempt schedule (v20: same strikes, same DTE, same
    exits — the only difference is the structure) and reports both side
    by side, with avg max loss per contract next to avg P&L per trade.
"""

import argparse
import json
import sys
import traceback
from datetime import date

import yaml

from options_screener.backtest import (DEFAULT_REGIME_FILTER,
                                       DEFAULT_SPREAD_BACKTEST,
                                       DEFAULT_SPREAD_WIDTH_SWEEP,
                                       DEFAULT_STRUCTURE_COMPARE,
                                       DEFAULT_SWEEP_GRID,
                                       build_backtest_payload,
                                       build_earnings_payload,
                                       build_regime_compare_payload,
                                       build_regime_filter_multiwindow_payload,
                                       build_regime_filter_payload,
                                       build_regime_payload,
                                       build_spread_dividend_payload,
                                       build_spread_earnings_payload,
                                       build_spread_payload,
                                       build_structure_compare_payload,
                                       build_sweep_payload,
                                       build_width_payload,
                                       compare_regime_filter,
                                       compare_regime_splits,
                                       compare_structures,
                                       default_sweep_key, rank_sweep,
                                       parse_regime_windows,
                                       render_backtest_markdown,
                                       render_earnings_markdown,
                                       render_regime_compare_markdown,
                                       render_regime_filter_markdown,
                                       render_regime_filter_multiwindow_markdown,
                                       render_regime_markdown,
                                       render_spread_dividend_markdown,
                                       render_spread_earnings_markdown,
                                       render_spread_markdown,
                                       render_structure_compare_markdown,
                                       render_sweep_markdown,
                                       render_width_markdown,
                                       simulate, simulate_earnings_aware,
                                       simulate_regime_filtered,
                                       simulate_spreads,
                                       simulate_spreads_dividend_aware,
                                       simulate_spreads_earnings_aware,
                                       split_by_iv_regime,
                                       summarize, summarize_spreads,
                                       sweep_exit_rules,
                                       sweep_skew_spreads,
                                       sweep_width_spreads)
from options_screener.data import (DataError, get_closes,
                                   get_closes_with_dates,
                                   get_closes_window, slice_window)


def _regime_filter_on_closes(closes, cfg):
    """Plain vs regime-filtered run over one closes series.

    The plain run shares the filtered run's warmup start, so the P&L
    delta isolates the filter itself, not the later start. Shared by
    the single-window branch and the multi-window mode.
    """
    rf = cfg.get("regime_filter") or DEFAULT_REGIME_FILTER
    lookback = rf.get("lookback_days",
                      DEFAULT_REGIME_FILTER["lookback_days"])
    # plain run shares the filtered run's warmup start, so the P&L
    # delta isolates the filter itself, not the later start
    warmup = cfg["backtest"]["iv_window"] + lookback
    plain_trades = simulate(closes, cfg, start_day=warmup)
    filt = simulate_regime_filtered(closes, cfg)
    return {"plain": {"trades": plain_trades,
                      "summary": summarize(plain_trades)},
            "filtered": filt}


def backtest_ticker_multiwindow(ticker, cfg, years, offsets):
    """Run the regime filter in several non-overlapping windows at once.

    Fetches the ticker once over years + max(offsets) years and slices
    each window out of the same series (slice_window), so the windows
    share identical underlying data and the fetch count stays one per
    ticker instead of one per window. Returns ({offset: result},
    [errors]): a window that fails its floor check is skipped for that
    ticker (the error is recorded) instead of killing the other
    windows; a failed fetch raises DataError and kills the ticker.
    """
    closes, dates = get_closes_with_dates(
        ticker, period=f"{years + max(offsets)}y")
    out, errors = {}, []
    for off in offsets:
        try:
            window_closes, w_start, w_end = slice_window(
                closes, dates, years, off, date.today())
        except DataError as exc:
            errors.append(str(exc))
            continue
        res = _regime_filter_on_closes(window_closes, cfg)
        res["window"] = {"start": w_start.isoformat(),
                         "end": w_end.isoformat(),
                         "years": years, "offset_years": off}
        out[off] = res
    return out, errors


def backtest_ticker(ticker, cfg, years, sweep=False, regimes=False,
                    spreads=False, spread_widths=False,
                    earnings_calendar=None, dividend_calendar=None,
                    regime_filter=False, offset_years=0,
                    structure_compare=False):
    if structure_compare:
        closes = get_closes(ticker, period=f"{years}y")
        sc = cfg.get("structure_compare") or DEFAULT_STRUCTURE_COMPARE
        return compare_structures(
            closes, cfg,
            sc.get("skew", DEFAULT_STRUCTURE_COMPARE["skew"]),
            sc.get("width_pct", DEFAULT_STRUCTURE_COMPARE["width_pct"]))
    if regime_filter:
        window = None
        if offset_years:
            closes, w_start, w_end = get_closes_window(
                ticker, years, offset_years)
            window = {"start": w_start.isoformat(),
                      "end": w_end.isoformat(),
                      "years": years, "offset_years": offset_years}
        else:
            closes = get_closes(ticker, period=f"{years}y")
        res = _regime_filter_on_closes(closes, cfg)
        res["window"] = window
        return res
    if dividend_calendar is not None:
        closes, dates = get_closes_with_dates(ticker, period=f"{years}y")
        sb = cfg.get("spread_backtest") or DEFAULT_SPREAD_BACKTEST
        per_skew = {}
        for skew in sb["skews"]:
            plain_trades = simulate_spreads(closes, cfg, skew,
                                            sb["width_pct"])
            filt = simulate_spreads_dividend_aware(
                closes, cfg, skew, sb["width_pct"], dates,
                dividend_calendar.get(ticker, []))
            per_skew[skew] = {
                "plain": {"trades": plain_trades,
                          "summary": summarize_spreads(plain_trades)},
                "filtered": filt}
        return {"skews": per_skew,
                "calendar_dates": len(dividend_calendar.get(ticker, []))}
    if earnings_calendar is not None:
        closes, dates = get_closes_with_dates(ticker, period=f"{years}y")
        if spreads:
            sb = cfg.get("spread_backtest") or DEFAULT_SPREAD_BACKTEST
            per_skew = {}
            for skew in sb["skews"]:
                plain_trades = simulate_spreads(closes, cfg, skew,
                                                sb["width_pct"])
                filt = simulate_spreads_earnings_aware(
                    closes, cfg, skew, sb["width_pct"], dates,
                    earnings_calendar.get(ticker, []))
                per_skew[skew] = {
                    "plain": {"trades": plain_trades,
                              "summary": summarize_spreads(plain_trades)},
                    "filtered": filt}
            return {"skews": per_skew,
                    "calendar_dates": len(earnings_calendar.get(ticker, []))}
        plain_trades = simulate(closes, cfg)
        filt = simulate_earnings_aware(
            closes, cfg, dates,
            earnings_calendar.get(ticker, []))
        return {"plain": {"trades": plain_trades,
                          "summary": summarize(plain_trades)},
                "filtered": filt,
                "calendar_dates": len(earnings_calendar.get(ticker, []))}
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
    ap.add_argument("--regime-compare", action="store_true",
                    help="run the v6 regime split per ticker and compare "
                         "the high-minus-low gaps across tickers, ranked "
                         "by P&L gap (v15; descriptive only — medians "
                         "differ per ticker, so only the within-ticker "
                         "gaps are compared, never absolute IV levels)")
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
    ap.add_argument("--regime-filter", action="store_true",
                    help="run the v16 forward entry filter: enter only when "
                         "today's entry IV is above the trailing median of "
                         "the same IV proxy over regime_filter.lookback_days "
                         "trading days (default 252), computed only from "
                         "data available at entry time. Reports plain vs "
                         "filtered P&L side by side over the same warmup "
                         "start, with skipped counts and trade share.")
    ap.add_argument("--offset-years", type=int, default=0,
                    help="with --regime-filter only: re-test the filter on "
                         "an earlier window — the --years-long window "
                         "ending this many calendar years ago (v17: the "
                         "re-test v16 asked for, same rule and lookback, "
                         "different period). Must be >= 0.")
    ap.add_argument("--regime-windows", default=None, metavar="0,2,4",
                    help="with --regime-filter only: run the filter on "
                         "several non-overlapping windows at once — "
                         "comma-separated offsets in years, at least two, "
                         "e.g. 0,2,4 (v18: the same rule on the recent "
                         "window plus the windows ending 2 and 4 years "
                         "ago, aggregated per ticker). Cannot be combined "
                         "with --offset-years.")
    ap.add_argument("--structure-compare", action="store_true",
                    help="run the v20 head-to-head: the v1 CSP and the v2 "
                         "bull-put spread over the SAME entry-attempt "
                         "schedule (same strikes, same DTE, same exits), "
                         "reported side by side with avg max loss per "
                         "contract next to avg P&L per trade. Spread skew "
                         "and width come from structure_compare: in config.")
    ap.add_argument("--earnings-calendar", default=None, metavar="FILE",
                    help="JSON file {ticker: [\"YYYY-MM-DD\", ...]} of known "
                         "earnings dates: re-run the simulation with the "
                         "live earnings gate applied and report plain vs "
                         "filtered side by side (no free source publishes "
                         "historical earnings calendars, so the dates are "
                         "caller-supplied; a ticker absent from the file "
                         "runs unfiltered and is flagged as such). "
                         "Combine with --spreads for the earnings-aware "
                         "spread backtest (plain vs gated per skew).")
    ap.add_argument("--dividend-calendar", default=None, metavar="FILE",
                    help="JSON file {ticker: [\"YYYY-MM-DD\", ...]} of known "
                         "ex-dividend dates: with --spreads, re-run the "
                         "spread simulation with the opt-in dividend "
                         "exclusion applied and report plain vs filtered "
                         "side by side per skew (v19; no free source "
                         "publishes historical ex-dividend calendars, so "
                         "the dates are caller-supplied; a ticker absent "
                         "from the file runs unfiltered and is flagged as "
                         "such). Requires --spreads; cannot be combined "
                         "with --earnings-calendar.")
    args = ap.parse_args()

    if args.dividend_calendar and not args.spreads:
        ap.error("--dividend-calendar only applies with --spreads")
    if args.dividend_calendar and args.earnings_calendar:
        ap.error("--dividend-calendar and --earnings-calendar cannot be "
                 "combined; run one comparison at a time")

    if args.offset_years and not args.regime_filter:
        ap.error("--offset-years only applies with --regime-filter")
    if args.offset_years < 0:
        ap.error("--offset-years must be >= 0")
    if args.regime_windows is not None and not args.regime_filter:
        ap.error("--regime-windows only applies with --regime-filter")
    if args.regime_windows is not None and args.offset_years:
        ap.error("--regime-windows and --offset-years cannot be combined; "
                 "use --regime-windows for several windows at once")
    offsets = None
    if args.regime_windows is not None:
        try:
            offsets = parse_regime_windows(args.regime_windows)
        except ValueError as exc:
            ap.error(str(exc))
    multiwindow = args.regime_filter and offsets is not None

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.all:
        tickers = cfg["watchlist"]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [cfg["watchlist"][0]]

    earnings_calendar = None
    if args.earnings_calendar:
        from options_screener.backtest import load_earnings_calendar
        earnings_calendar = load_earnings_calendar(args.earnings_calendar)

    dividend_calendar = None
    if args.dividend_calendar:
        from options_screener.backtest import load_dividend_calendar
        dividend_calendar = load_dividend_calendar(args.dividend_calendar)

    results, errors = {}, []
    for ticker in tickers:
        try:
            if multiwindow:
                res, werrs = backtest_ticker_multiwindow(
                    ticker, cfg, args.years, offsets)
                errors.extend(werrs)
                results[ticker] = res
                for off in offsets:
                    r = res.get(off)
                    if r is None:
                        print(f"[{ticker}] window -{off}y: no data",
                              file=sys.stderr)
                        continue
                    p = r["plain"]["summary"]
                    f = r["filtered"]["summary"]
                    print(f"[{ticker}] window -{off}y "
                          f"({r['window']['start']} to "
                          f"{r['window']['end']}): plain {p['trades']} "
                          f"trades ${p['total_pnl']:+,.2f}; filtered "
                          f"{f['trades']} trades "
                          f"${f['total_pnl']:+,.2f} "
                          f"({len(r['filtered']['skipped'])} skipped, "
                          f"lookback {r['filtered']['lookback_days']}d)")
                continue
            res = backtest_ticker(ticker, cfg, args.years, sweep=args.sweep,
                                  regimes=args.regimes or args.regime_compare,
                                  spreads=args.spreads,
                                  spread_widths=args.spread_widths,
                                  earnings_calendar=earnings_calendar,
                                  dividend_calendar=dividend_calendar,
                                  regime_filter=args.regime_filter,
                                  offset_years=args.offset_years,
                                  structure_compare=args.structure_compare)
            results[ticker] = res
            if args.structure_compare:
                c, s = res["csp"]["summary"], res["spread"]["summary"]
                cml = res["csp"]["avg_max_loss_per_contract"]
                sml = res["spread"]["avg_max_loss_per_contract"]
                cml_cell = f"${cml:,.2f}" if cml is not None else "—"
                sml_cell = f"${sml:,.2f}" if sml is not None else "—"
                print(f"[{ticker}] CSP {c['trades']} trades "
                      f"${c['total_pnl']:+,.2f} (avg max loss {cml_cell}); "
                      f"spread {s['trades']} trades "
                      f"${s['total_pnl']:+,.2f} (avg max loss {sml_cell}); "
                      f"{res['attempts']} attempts")
            elif args.regime_filter:
                p, f = res["plain"]["summary"], res["filtered"]["summary"]
                print(f"[{ticker}] regime filter: plain {p['trades']} trades "
                      f"${p['total_pnl']:+,.2f}; filtered {f['trades']} trades "
                      f"${f['total_pnl']:+,.2f} "
                      f"({len(res['filtered']['skipped'])} skipped, "
                      f"lookback {res['filtered']['lookback_days']}d)")
            elif dividend_calendar is not None and args.spreads:
                sb = (cfg.get("spread_backtest")
                      or DEFAULT_SPREAD_BACKTEST)
                first = sb["skews"][0]
                pair = res["skews"][first]
                p, f = pair["plain"]["summary"], pair["filtered"]["summary"]
                print(f"[{ticker}] spreads skew {first}: plain "
                      f"{p['trades']} trades ${p['total_pnl']:+,.2f}; "
                      f"dividend-filtered {f['trades']} trades "
                      f"${f['total_pnl']:+,.2f} "
                      f"({len(pair['filtered']['skipped'])} entries skipped, "
                      f"{res['calendar_dates']} calendar dates)")
            elif earnings_calendar is not None and args.spreads:
                sb = (cfg.get("spread_backtest")
                      or DEFAULT_SPREAD_BACKTEST)
                first = sb["skews"][0]
                pair = res["skews"][first]
                p, f = pair["plain"]["summary"], pair["filtered"]["summary"]
                print(f"[{ticker}] spreads skew {first}: plain "
                      f"{p['trades']} trades ${p['total_pnl']:+,.2f}; "
                      f"earnings-filtered {f['trades']} trades "
                      f"${f['total_pnl']:+,.2f} "
                      f"({len(pair['filtered']['skipped'])} entries skipped, "
                      f"{res['calendar_dates']} calendar dates)")
            elif earnings_calendar is not None:
                p, f = res["plain"]["summary"], res["filtered"]["summary"]
                print(f"[{ticker}] plain {p['trades']} trades "
                      f"${p['total_pnl']:+,.2f}; earnings-filtered "
                      f"{f['trades']} trades ${f['total_pnl']:+,.2f} "
                      f"({len(res['filtered']['skipped'])} entries skipped, "
                      f"{res['calendar_dates']} calendar dates)")
            elif args.spread_widths:
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
            elif args.regime_compare:
                row = compare_regime_splits({ticker: res})["rows"][0]
                gap = (f"${row['pnl_gap']:+,.2f}" if row["pnl_gap"] is not None
                       else "no gap")
                print(f"[{ticker}] median IV {res['median_iv']}, "
                      f"low: {row['low_trades']} trades "
                      f"${row['low_total_pnl']:+,.2f}, high: "
                      f"{row['high_trades']} trades "
                      f"${row['high_total_pnl']:+,.2f}, P&L gap {gap}"
                      + (" (thin)" if row["thin"] else ""))
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

    if multiwindow:
        rf = cfg.get("regime_filter") or DEFAULT_REGIME_FILTER
        lookback = rf.get("lookback_days",
                          DEFAULT_REGIME_FILTER["lookback_days"])
        windowed = {}
        for off in offsets:
            per = {t: results[t][off] for t in results
                   if off in results[t]}
            compared = compare_regime_filter(per)
            window = next((results[t][off]["window"] for t in results
                           if off in results[t]), None)
            windowed[off] = {"rows": compared["rows"],
                             "summary": compared["summary"],
                             "window": window}
        payload = build_regime_filter_multiwindow_payload(
            windowed, cfg, lookback)
        md_path, json_path = write_reports(
            payload, args.out_dir, "backtest-regime-filter-multiwindow",
            render_regime_filter_multiwindow_markdown)
    elif args.structure_compare:
        sc = cfg.get("structure_compare") or DEFAULT_STRUCTURE_COMPARE
        payload = build_structure_compare_payload(
            results, cfg,
            sc.get("skew", DEFAULT_STRUCTURE_COMPARE["skew"]),
            sc.get("width_pct", DEFAULT_STRUCTURE_COMPARE["width_pct"]))
        md_path, json_path = write_reports(
            payload, args.out_dir, "backtest-structure-compare",
            render_structure_compare_markdown)
    elif dividend_calendar is not None and args.spreads:
        sb = cfg.get("spread_backtest") or DEFAULT_SPREAD_BACKTEST
        payload = build_spread_dividend_payload(results, cfg, sb["skews"],
                                                sb["width_pct"])
        md_path, json_path = write_reports(payload, args.out_dir,
                                           "backtest-spread-dividends",
                                           render_spread_dividend_markdown)
    elif args.regime_filter:
        rf = cfg.get("regime_filter") or DEFAULT_REGIME_FILTER
        lookback = rf.get("lookback_days",
                          DEFAULT_REGIME_FILTER["lookback_days"])
        # every ticker shares the same window (same trading calendar),
        # so take the first recorded one; a run with no window ran on
        # the recent period, which the payload renders as before
        window = next((r.get("window") for r in results.values()
                       if r.get("window")), None)
        payload = build_regime_filter_payload(results, cfg, lookback,
                                              window=window)
        md_path, json_path = write_reports(payload, args.out_dir,
                                           "backtest-regime-filter",
                                           render_regime_filter_markdown)
    elif earnings_calendar is not None and args.spreads:
        sb = cfg.get("spread_backtest") or DEFAULT_SPREAD_BACKTEST
        payload = build_spread_earnings_payload(results, cfg, sb["skews"],
                                                sb["width_pct"])
        md_path, json_path = write_reports(payload, args.out_dir,
                                           "backtest-spread-earnings",
                                           render_spread_earnings_markdown)
    elif earnings_calendar is not None:
        payload = build_earnings_payload(results, cfg)
        md_path, json_path = write_reports(payload, args.out_dir,
                                           "backtest-earnings",
                                           render_earnings_markdown)
    elif args.spread_widths:
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
    elif args.regime_compare:
        payload = build_regime_compare_payload(results, cfg)
        md_path, json_path = write_reports(payload, args.out_dir,
                                           "backtest-regime-compare",
                                           render_regime_compare_markdown)
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
