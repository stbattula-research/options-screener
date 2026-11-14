"""Backtest engine for the premium-selling strategy (research only).

Simulates the screener's cash-secured-put strategy over historical daily
closes: weekly ~30-delta short puts, 30-45 DTE, exits at 50% of max profit,
21 DTE, or 2x-premium loss. Output: per-trade log + aggregate stats +
Markdown/JSON report.

Model-based, NOT historical-chain data: free sources do not publish
historical option chains, so every premium is priced with Black-Scholes
against an IV proxy (trailing realized vol x multiplier). Treat results as
illustrative research, never as expected returns. The README and every
report carry this caveat.

Signals only. This module never places orders and holds no broker
credentials.
"""

import math

from .vol import (bs_put_delta, bs_put_value, bs_put_value_skewed,
                   realized_vol, skewed_iv)


def strike_for_delta(price, years_to_expiry, risk_free_rate, sigma,
                     target_delta, tol=1e-4):
    """Find the put strike whose Black-Scholes delta ~= target_delta.

    Put delta is monotonically decreasing in strike (higher strike -> deeper
    in the money -> closer to -1), so bisection on [0.5*S, 1.5*S] converges.
    """
    lo, hi = price * 0.5, price * 1.5
    d_lo = bs_put_delta(price, lo, years_to_expiry, risk_free_rate, sigma)
    d_hi = bs_put_delta(price, hi, years_to_expiry, risk_free_rate, sigma)
    if not (d_lo >= target_delta >= d_hi):
        raise ValueError(f"target delta {target_delta} outside bracket "
                         f"[{d_hi:.3f}, {d_lo:.3f}]")
    for _ in range(100):
        mid = (lo + hi) / 2.0
        d = bs_put_delta(price, mid, years_to_expiry, risk_free_rate, sigma)
        if abs(d - target_delta) < tol:
            return mid
        if d < target_delta:   # too negative -> strike too high
            hi = mid
        else:                  # not negative enough -> strike too low
            lo = mid
    return (lo + hi) / 2.0


def estimate_iv(prices, day, window, multiplier, fallback):
    """IV proxy for simulation: trailing realized vol x multiplier.

    Realized vol systematically understates implied vol (the variance risk
    premium), hence the configurable multiplier. `fallback` covers the first
    `window` days where trailing history is too short.
    """
    seg = prices[max(0, day - window):day + 1]
    rv = realized_vol(seg, window) if len(seg) >= window + 1 else None
    if rv is None or rv <= 0:
        return fallback
    return rv * multiplier


def simulate_trade(prices, entry_day, dte, sigma, cfg, exits=None):
    """Simulate one short-put trade from entry through its first exit.

    Requires entry_day + dte < len(prices). Returns a trade dict; per-share
    premiums are scaled to a 1-contract position in `pnl_per_contract`.
    `exits` optionally overrides cfg["exits"] (used by the sensitivity
    sweep); the caller's cfg dict is never mutated.
    """
    r = cfg["risk_free_rate"]
    strat = cfg["strategy"]
    ex = dict(cfg["exits"])
    if exits:
        ex.update(exits)
    expiry_day = entry_day + dte

    S = prices[entry_day]
    T = dte / 365.0
    K = strike_for_delta(S, T, r, sigma, strat["target_delta"])
    premium = bs_put_value(S, K, T, r, sigma)  # per share received
    entry_delta = bs_put_delta(S, K, T, r, sigma)

    profit_trigger = premium * (1 - ex["profit_target_pct"])
    loss_trigger = premium * (1 + ex["loss_manage_multiple"])

    reason, exit_day, exit_value = "expired", expiry_day, None
    for j in range(entry_day + 1, expiry_day + 1):
        t_rem = (expiry_day - j) / 365.0
        V = bs_put_value(prices[j], K, t_rem, r, sigma)
        if V <= profit_trigger:
            reason, exit_day, exit_value = "profit", j, V
            break
        if j < expiry_day and (expiry_day - j) <= ex["manage_dte"]:
            reason, exit_day, exit_value = "time", j, V
            break
        if j < expiry_day and V >= loss_trigger:
            reason, exit_day, exit_value = "loss", j, V
            break
    if exit_value is None:  # held to expiry: value = intrinsic
        exit_value = max(K - prices[expiry_day], 0.0)

    pnl_per_share = premium - exit_value
    return {
        "entry_day": entry_day,
        "exit_day": exit_day,
        "days_held": exit_day - entry_day,
        "entry_price": round(S, 2),
        "strike": round(K, 2),
        "dte": dte,
        "iv": round(sigma, 4),
        "delta": round(entry_delta, 3),
        "premium_per_share": round(premium, 3),
        "exit_reason": reason,
        "exit_value_per_share": round(exit_value, 3),
        "pnl_per_contract": round(pnl_per_share * 100, 2),
        # per-contract return on cash-secured capital (strike x 100)
        "return_on_risk": round(pnl_per_share / K, 4),
        "assigned": bool(reason == "expired" and exit_value > 0),
    }


def _run_simulation(prices, cfg, exits=None, earnings_gate=None,
                    start_day=None):
    """Shared simulation loop. Returns (trades, skipped_entries).

    `earnings_gate`, when given, is a callable (entry_day, dte) -> an
    earnings `date` inside that entry's option window, or None. A gated
    entry is never opened: it is recorded in `skipped_entries` and the
    next attempt moves up by `entry_interval_days` only (no position was
    opened, so there is no exit day to advance past). Without a gate the
    skipped list is always empty and behavior is exactly `simulate`'s.

    `start_day` is the first entry-attempt day; it defaults to
    `iv_window` (the first day with trailing history for the IV proxy).
    Callers that need a later universe — the v16 regime filter, whose
    trailing-median threshold needs a warmup — pass a later start so the
    plain and filtered runs attempt entries over the same days.
    """
    bt = cfg["backtest"]
    dte = bt.get("entry_dte") or (cfg["strategy"]["dte_min"]
                                  + cfg["strategy"]["dte_max"]) // 2
    window, interval = bt["iv_window"], bt["entry_interval_days"]
    n = len(prices)

    trades, skipped = [], []
    day = window if start_day is None else start_day
    while day + dte < n:
        if earnings_gate is not None:
            hit = earnings_gate(day, dte)
            if hit is not None:
                skipped.append({"entry_day": day, "dte": dte,
                                "earnings_date": hit.isoformat()})
                day += interval
                continue
        sigma = estimate_iv(prices, day, window,
                            bt["iv_multiplier"], bt["iv_fallback"])
        trade = simulate_trade(prices, day, dte, sigma, cfg, exits=exits)
        trades.append(trade)
        day = max(trade["exit_day"] + 1, day + interval)
    return trades, skipped


def simulate(prices, cfg, exits=None, start_day=None):
    """Run the strategy over a price series. Returns a list of trade dicts.

    Entries are attempted every `entry_interval_days` trading days (one
    position at a time), each needing `entry_dte` of future data to resolve.
    Trades that cannot complete are never opened. `exits` overrides the exit
    rules from config (sensitivity sweep), without mutating cfg.
    `start_day` overrides the first entry-attempt day (default: the first
    day with trailing history for the IV proxy); the v16 regime filter
    passes its warmup day so plain and filtered runs share one universe.
    """
    trades, _ = _run_simulation(prices, cfg, exits=exits,
                                start_day=start_day)
    return trades


def summarize(trades):
    """Aggregate stats over a trade list. Returns a JSON-serializable dict."""
    n = len(trades)
    pnls = [t["pnl_per_contract"] for t in trades]
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)

    # equity curve + max drawdown (per-contract cumulative P&L)
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    exits = {}
    for t in trades:
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1

    return {
        "trades": n,
        "wins": wins,
        "win_rate": round(wins / n, 4) if n else None,
        "total_pnl": round(total, 2),
        "avg_pnl_per_trade": round(total / n, 2) if n else None,
        "avg_return_on_risk": (round(sum(t["return_on_risk"] for t in trades) / n, 4)
                               if n else None),
        "best_trade": round(max(pnls), 2) if n else None,
        "worst_trade": round(min(pnls), 2) if n else None,
        "avg_days_held": (round(sum(t["days_held"] for t in trades) / n, 1)
                          if n else None),
        "max_drawdown": round(max_dd, 2),
        "exits": exits,
    }


ASSUMPTIONS = [
    "Model-based backtest: premiums priced with Black-Scholes, not from "
    "historical option chains (no free source publishes them).",
    "IV held constant per trade: entry IV proxy = trailing 20-day realized "
    "vol x 1.15; no volatility shocks mid-trade.",
    "European exercise, no early assignment; no dividends.",
    "No commissions, fees, or slippage; fills at model mid.",
    "One position at a time per ticker; entries every 5 trading days.",
    "Incomplete trades (not enough future data) are never opened.",
]


def build_backtest_payload(results, cfg):
    """results: {ticker: {'trades': [...], 'summary': {...}}} -> report payload."""
    return {
        "strategy": "cash-secured puts: ~30 delta, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS,
        "tickers": results,
    }


def render_backtest_markdown(payload):
    """Human-readable backtest report."""
    lines = ["# Options Screener — Strategy Backtest (research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy (trailing realized vol x "
             "1.15) — this is NOT historical option-chain data. Results are "
             "illustrative research, not expected returns. See Assumptions.",
             ""]
    lines.append(f"**Strategy:** {payload['strategy']}")
    lines.append("")
    lines.append("## Results by ticker")
    lines.append("")
    lines.append("| Ticker | Trades | Win rate | Total P&L | Avg/trade | "
                 "Avg RoR | Max DD | Avg days |")
    lines.append("|--------|--------|----------|-----------|-----------|---------|"
                 "--------|----------|")
    for ticker, res in payload["tickers"].items():
        s = res["summary"]
        if s["trades"] == 0:
            lines.append(f"| {ticker} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {ticker} | {s['trades']} | {s['win_rate']:.1%} | "
            f"${s['total_pnl']:,.2f} | ${s['avg_pnl_per_trade']:,.2f} | "
            f"{s['avg_return_on_risk']:.2%} | ${s['max_drawdown']:,.2f} | "
            f"{s['avg_days_held']:.0f} |")
    for ticker, res in payload["tickers"].items():
        s = res["summary"]
        if s["trades"] == 0:
            continue
        lines.append("")
        lines.append(f"## {ticker} — trade log")
        lines.append("")
        lines.append("| # | Entry | Strike | Δ | Prem | Exit | Days | P&L | RoR |")
        lines.append("|---|-------|--------|---|------|------|------|-----|-----|")
        for i, t in enumerate(res["trades"], 1):
            lines.append(
                f"| {i} | day {t['entry_day']} @ ${t['entry_price']:.2f} | "
                f"${t['strike']:.2f} | {t['delta']:.2f} | "
                f"${t['premium_per_share']:.2f} | {t['exit_reason']} | "
                f"{t['days_held']} | ${t['pnl_per_contract']:+.2f} | "
                f"{t['return_on_risk']:+.2%} |")
        lines.append("")
        exits = ", ".join(f"{k}: {v}" for k, v in s["exits"].items())
        lines.append(f"Exits — {exits}. "
                     f"Best ${s['best_trade']:+.2f}, worst ${s['worst_trade']:+.2f}.")
    lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)




# ---------------------------------------------------------------------------
# IV-regime split (v6)
#
# The headline backtest averages over every market mood. This split asks a
# narrower question: does the strategy behave differently when it enters
# in a high-volatility regime vs a low-volatility one? Regimes are defined
# per ticker by the MEDIAN entry IV of that ticker's own trades (low =
# entry IV <= median, high = entry IV > median), so the split is always
# roughly even and never cherry-picks a threshold after seeing results.
# The cost of that honesty: medians differ per ticker, so regimes are NOT
# comparable across tickers, and "high IV" here means the trailing-realized
# IV proxy, not traded implied vol.
#
# Model-based like the rest of the backtester: Black-Scholes premiums
# against the trailing-realized-vol IV proxy, no commissions/slippage.
# Signals only — never places orders, holds no broker credentials.
# ---------------------------------------------------------------------------

def median_iv(trades):
    """Median of the trades' entry-IV values; None when there are none."""
    if not trades:
        return None
    ivs = sorted(t["iv"] for t in trades)
    n = len(ivs)
    mid = n // 2
    if n % 2:
        return ivs[mid]
    return (ivs[mid - 1] + ivs[mid]) / 2.0


def split_by_iv_regime(trades):
    """Split trades into low/high entry-IV regimes at their median.

    Returns {"median_iv": float|None, "low": {"trades", "summary"},
    "high": {"trades", "summary"}}. When every trade shares one IV the
    high regime is legitimately empty (its summary reports 0 trades,
    not zeros dressed up as results).
    """
    med = median_iv(trades)
    if med is None:
        empty = {"trades": [], "summary": summarize([])}
        return {"median_iv": None, "low": empty, "high": empty}
    low = [t for t in trades if t["iv"] <= med]
    high = [t for t in trades if t["iv"] > med]
    return {"median_iv": med,
            "low": {"trades": low, "summary": summarize(low)},
            "high": {"trades": high, "summary": summarize(high)}}


def build_regime_payload(results, cfg):
    """results: {ticker: {'trades': [...], 'summary': {...}}} -> payload."""
    return {
        "mode": "IV-regime split",
        "strategy": "cash-secured puts: ~30 delta, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS + [
            "Regimes are split per ticker at the median entry IV of that "
            "ticker's own trades (low: IV <= median, high: IV > median); "
            "medians differ per ticker, so regimes are not comparable "
            "across tickers.",
            "Entry IV is the trailing-realized-vol proxy (x 1.15), not "
            "traded implied vol; a high-IV regime also means larger "
            "premiums AND larger moves — the split shows whether the "
            "extra premium compensated for the extra risk.",
            "This is a description of past model trades, not a rule for "
            "skipping low- or high-IV entries: the median is only known "
            "after the whole period, so it cannot be traded in real time.",
        ],
        "tickers": {ticker: split_by_iv_regime(res["trades"])
                    for ticker, res in results.items()},
    }


def _regime_row_md(label, iv_note, res):
    s = res["summary"]
    if s["trades"] == 0:
        return f"| {label} | {iv_note} | 0 | — | — | — | — | — | — |"
    return (f"| {label} | {iv_note} | {s['trades']} | {s['win_rate']:.1%} | "
            f"${s['total_pnl']:+,.2f} | ${s['avg_pnl_per_trade']:+,.2f} | "
            f"{s['avg_return_on_risk']:+.2%} | ${s['max_drawdown']:,.2f} | "
            f"{s['avg_days_held']:.0f} |")


def render_regime_markdown(payload):
    """Human-readable regime-split report."""
    lines = ["# Options Screener — IV-Regime Split (research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy (trailing realized vol x "
             "1.15) — this is NOT historical option-chain data. The split "
             "describes past model trades; the median is only known after "
             "the whole period, so it is not a live entry filter.",
             "",
             f"**Strategy:** {payload['strategy']}",
             ""]
    for ticker, split in payload["tickers"].items():
        med = split["median_iv"]
        lines.append(f"## {ticker}")
        lines.append("")
        if med is None:
            lines.append("No completed trades — nothing to split.")
            lines.append("")
            continue
        lines.append(f"Median entry IV: {med:.4f} "
                     f"(low: IV <= median, high: IV > median).")
        lines.append("")
        lines.append("| Regime | Entry IV | Trades | Win rate | Total P&L | "
                     "Avg/trade | Avg RoR | Max DD | Avg days |")
        lines.append("|--------|----------|--------|----------|-----------|"
                     "-----------|---------|--------|----------|")
        lines.append(_regime_row_md("Low IV", f"<= {med:.4f}", split["low"]))
        lines.append(_regime_row_md("High IV", f"> {med:.4f}", split["high"]))
        lo, hi = split["low"]["summary"], split["high"]["summary"]
        if lo["trades"] and hi["trades"]:
            gap = hi["total_pnl"] - lo["total_pnl"]
            lines.append("")
            lines.append(f"High-IV minus low-IV total P&L: ${gap:+,.2f} "
                         f"(low {lo['win_rate']:.0%} win, high "
                         f"{hi['win_rate']:.0%} win). A positive gap means "
                         "the extra premium collected in high-IV entries "
                         "more than paid for the extra risk on this period "
                         "— it does not promise the next period behaves "
                         "the same.")
        lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- Compare win rate AND worst trade AND max drawdown, not "
                 "just total P&L: high-IV regimes usually win more often "
                 "but lose bigger when they lose.")
    lines.append("- A regime with very few trades says little; treat small "
                 "gaps as noise.")
    lines.append("- Never turn the median into a live rule (\"only trade "
                 "above X\") without re-testing it as a forward rule on "
                 "unseen data — this split is descriptive, not a strategy.")
    lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-ticker regime comparison (v15)
#
# v6 answered the regime question for one ticker at a time: on SPY the
# high-IV regime won more often and made more, at the cost of a bigger
# drawdown. v15 asks whether that pattern generalizes: run the same
# per-ticker median split across the watchlist and compare the GAPS
# (high minus low) — win-rate gap, P&L gap, drawdown gap — ticker by
# ticker, ranked by P&L gap.
#
# The honest limit, carried in the payload and the report: each ticker's
# regimes are defined by THAT ticker's median entry IV, and medians
# differ wildly per ticker (a "high" IV for SPY can be a "low" IV for
# NVDA). So absolute IV levels are NOT comparable across tickers, and
# this report never compares them. What it compares is the pattern:
# within each ticker, did high-IV entries beat low-IV entries, and by
# how much? The median-IV column is printed per ticker so the reader
# can see how different "high" is from one row to the next.
#
# Same honesty rules as v6: entry IV is the trailing-realized-vol
# proxy, not traded implied vol; the median is only known after the
# whole period, so the split is descriptive, never a live entry filter.
# ---------------------------------------------------------------------------

MIN_REGIME_TRADES = 10  # a regime with fewer trades is flagged "thin"


def _median(values):
    """Median of a non-empty list of numbers; None when empty."""
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def compare_regime_splits(splits):
    """Compare per-ticker IV-regime splits by their high-minus-low gaps.

    `splits`: {ticker: split_by_iv_regime(trades)} -> {"rows": [...],
    "summary": {...}}. Rows are ranked by P&L gap, highest first; rows
    with no measurable gap (an empty regime) sort last. A row is
    "thin" when either regime has fewer than MIN_REGIME_TRADES trades
    — its gaps are printed but must be read as noise, not signal.
    """
    rows = []
    for ticker, split in splits.items():
        lo, hi = split["low"]["summary"], split["high"]["summary"]
        measurable = lo["trades"] > 0 and hi["trades"] > 0
        rows.append({
            "ticker": ticker,
            "median_iv": split["median_iv"],
            "low_trades": lo["trades"],
            "high_trades": hi["trades"],
            "low_win_rate": lo["win_rate"],
            "high_win_rate": hi["win_rate"],
            "low_total_pnl": lo["total_pnl"],
            "high_total_pnl": hi["total_pnl"],
            "low_max_drawdown": lo["max_drawdown"],
            "high_max_drawdown": hi["max_drawdown"],
            "pnl_gap": (round(hi["total_pnl"] - lo["total_pnl"], 2)
                        if measurable else None),
            "win_gap": (round(hi["win_rate"] - lo["win_rate"], 4)
                        if measurable and hi["win_rate"] is not None
                        and lo["win_rate"] is not None else None),
            "dd_gap": (round(hi["max_drawdown"] - lo["max_drawdown"], 2)
                       if measurable else None),
            "thin": (not measurable
                     or min(lo["trades"], hi["trades"]) < MIN_REGIME_TRADES),
        })
    rows.sort(key=lambda r: (r["pnl_gap"] is None,
                             -(r["pnl_gap"] or 0)))
    readable = [r for r in rows if not r["thin"]]
    summary = {
        "tickers_compared": len(rows),
        "readable_tickers": len(readable),
        "high_wins_pnl": sum(1 for r in readable
                             if r["pnl_gap"] is not None
                             and r["pnl_gap"] > 0),
        "high_wins_win_rate": sum(1 for r in readable
                                  if r["win_gap"] is not None
                                  and r["win_gap"] > 0),
        "median_pnl_gap": _median([r["pnl_gap"] for r in readable
                                   if r["pnl_gap"] is not None]),
        "median_win_gap": _median([r["win_gap"] for r in readable
                                   if r["win_gap"] is not None]),
        "median_dd_gap": _median([r["dd_gap"] for r in readable
                                  if r["dd_gap"] is not None]),
    }
    return {"rows": rows, "summary": summary}


def build_regime_compare_payload(splits, cfg):
    """splits: {ticker: split_by_iv_regime(trades)} -> payload."""
    compared = compare_regime_splits(splits)
    return {
        "mode": "Multi-ticker IV-regime comparison",
        "strategy": "cash-secured puts: ~30 delta, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS + [
            "Each ticker's regimes are split at that ticker's own median "
            "entry IV (low: IV <= median, high: IV > median); medians "
            "differ per ticker, so absolute IV levels are NOT comparable "
            "across tickers. Only the within-ticker gaps (high minus low) "
            "are compared — the median-IV column is printed per ticker so "
            "the reader can see how different 'high' is from row to row.",
            "Entry IV is the trailing-realized-vol proxy (x 1.15), not "
            "traded implied vol; a high-IV regime also means larger "
            "premiums AND larger moves — the gaps show whether the extra "
            "premium compensated for the extra risk, per ticker.",
            "This is a description of past model trades, not a rule for "
            "skipping low- or high-IV entries: the median is only known "
            "after the whole period, so it cannot be traded in real time.",
            "A row is flagged thin when either regime has fewer than "
            f"{MIN_REGIME_TRADES} trades; thin gaps are printed but must "
            "be read as noise, and thin rows never count toward the "
            "cross-ticker summary.",
        ],
        "rows": compared["rows"],
        "summary": compared["summary"],
    }


def _compare_cell(value, fmt):
    return "—" if value is None else fmt.format(value)


def render_regime_compare_markdown(payload):
    """Human-readable multi-ticker regime comparison."""
    lines = ["# Options Screener — Multi-Ticker IV-Regime Comparison "
             "(research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy (trailing realized vol x "
             "1.15) — this is NOT historical option-chain data. Each "
             "ticker's regimes are split at that ticker's own median "
             "entry IV, so absolute IV levels are NOT comparable across "
             "tickers: what is compared here is the *pattern* — within "
             "each ticker, did high-IV entries beat low-IV entries, and "
             "by how much?",
             "",
             f"**Strategy:** {payload['strategy']}",
             ""]
    s = payload["summary"]
    lines.append("## Cross-ticker summary")
    lines.append("")
    lines.append(f"- {s['tickers_compared']} tickers compared, "
                 f"{s['readable_tickers']} with enough trades in both "
                 "regimes to read.")
    if s["readable_tickers"]:
        lines.append(f"- High-IV entries made more than low-IV entries on "
                     f"{s['high_wins_pnl']} of {s['readable_tickers']} "
                     "readable tickers, and won more often on "
                     f"{s['high_wins_win_rate']} of "
                     f"{s['readable_tickers']}.")
        lines.append(f"- Median P&L gap (high minus low): "
                     f"${s['median_pnl_gap']:+,.2f} per contract; median "
                     f"win-rate gap: {s['median_win_gap']:+.1%}; median "
                     f"max-drawdown gap: ${s['median_dd_gap']:+,.2f} "
                     "(positive means the high-IV regime drew down "
                     "further).")
    else:
        lines.append("- No ticker has enough trades in both regimes to "
                     "summarize — every row is thin. Run more years or "
                     "more tickers before reading anything into this.")
    lines.append("")
    lines.append("## Per-ticker gaps (ranked by P&L gap)")
    lines.append("")
    lines.append("| Ticker | Median IV | Low trades | Low win | Low P&L | "
                 "High trades | High win | High P&L | P&L gap | Win gap | "
                 "Max-DD gap | Read |")
    lines.append("|--------|-----------|------------|---------|---------|"
                 "-------------|----------|----------|---------|---------|"
                 "------------|------|")
    for r in payload["rows"]:
        lines.append(
            f"| {r['ticker']} | "
            f"{_compare_cell(r['median_iv'], '{:.4f}')} | "
            f"{r['low_trades']} | "
            f"{_compare_cell(r['low_win_rate'], '{:.1%}')} | "
            f"${r['low_total_pnl']:+,.2f} | "
            f"{r['high_trades']} | "
            f"{_compare_cell(r['high_win_rate'], '{:.1%}')} | "
            f"${r['high_total_pnl']:+,.2f} | "
            f"{_compare_cell(r['pnl_gap'], '${:+,.2f}')} | "
            f"{_compare_cell(r['win_gap'], '{:+.1%}')} | "
            f"{_compare_cell(r['dd_gap'], '${:+,.2f}')} | "
            f"{'thin ⚠' if r['thin'] else 'readable'} |")
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- Read the gaps, not the levels: a $500 P&L gap on a "
                 "ticker whose median IV is 0.45 and a $500 gap on one "
                 "whose median is 0.15 are the same pattern at very "
                 "different volatilities — the median-IV column is there "
                 "so the difference is visible, not hidden.")
    lines.append("- A positive P&L gap with a positive drawdown gap is the "
                 "usual trade: extra premium paid for extra risk. A "
                 "positive P&L gap with a *negative* drawdown gap would "
                 "be the interesting row — more money for less pain — "
                 "and deserves a second look, not a rule change.")
    lines.append("- Thin rows (⚠) say nothing. A 3-trade regime can show "
                 "any gap by luck; the summary counts only readable rows.")
    lines.append("- Never turn any of this into a live rule (\"only trade "
                 "high-IV tickers\") without re-testing it as a forward "
                 "rule on unseen data — this comparison is descriptive, "
                 "not a strategy.")
    lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exit-rule sensitivity sweep (v5)
#
# The screener's mechanical exit rules (profit target / loss multiple /
# manage-DTE) are config values, not validated truths. The sweep re-runs the
# same simulation over a grid of those rules and ranks the outcomes, so the
# report can answer: is the default combo near the top (robust), or does
# performance collapse one notch away (fragile/overfit)?
#
# Model-based like the rest of the backtester: Black-Scholes premiums
# against the trailing-realized-vol IV proxy, no commissions/slippage.
# Signals only — never places orders, holds no broker credentials.
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_GRID = {
    "profit_targets": [0.25, 0.50, 0.75],   # close at this fraction of max profit
    "loss_multiples": [1.5, 2.0, 3.0],      # manage when cost-to-close hits this x premium
    "manage_dtes": [14, 21],                # exit/roll at this DTE
}


def sweep_key(pt, lm, md):
    """Human + JSON-friendly key for one exit-rule combo."""
    return f"profit={pt:.2f},loss={lm:.1f}x,dte={md}"


def default_sweep_key(cfg):
    """The combo matching the config's current exit rules."""
    ex = cfg["exits"]
    return sweep_key(ex["profit_target_pct"], ex["loss_manage_multiple"],
                     ex["manage_dte"])


def sweep_exit_rules(prices, cfg, grid):
    """Run the simulation over every exit-rule combo in `grid`.

    Returns {sweep_key: {"trades": [...], "summary": {...}}}. The config's
    own exits dict is never mutated by overrides.
    """
    out = {}
    for pt in grid["profit_targets"]:
        for lm in grid["loss_multiples"]:
            for md in grid["manage_dtes"]:
                ov = {"profit_target_pct": pt, "loss_manage_multiple": lm,
                      "manage_dte": md}
                trades = simulate(prices, cfg, exits=ov)
                out[sweep_key(pt, lm, md)] = {
                    "trades": trades,
                    "summary": summarize(trades),
                    "rules": {"profit_target_pct": pt,
                              "loss_manage_multiple": lm, "manage_dte": md},
                }
    return out


def rank_sweep(sweep_results, default_key):
    """Rank combos by total P&L, flagging the config default.

    Returns a list of rows sorted by total_pnl desc, each with
    `delta_vs_default` (combo P&L minus default-combo P&L).
    """
    default_total = sweep_results[default_key]["summary"]["total_pnl"]
    rows = []
    for key, res in sweep_results.items():
        s = res["summary"]
        rows.append({
            "key": key,
            "rules": res["rules"],
            "summary": s,
            "is_default": key == default_key,
            "delta_vs_default": round(s["total_pnl"] - default_total, 2),
        })
    rows.sort(key=lambda r: r["summary"]["total_pnl"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def build_sweep_payload(sweeps, cfg, grid):
    """sweeps: {ticker: ranked rows} -> JSON-serializable report payload."""
    return {
        "mode": "exit-rule sensitivity sweep",
        "grid": grid,
        "strategy": "cash-secured puts: ~30 delta, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS + [
            "Sweep varies ONLY the exit rules (profit target, loss "
            "multiple, manage DTE); entries, strikes, IV proxy and all "
            "other mechanics are identical across combos.",
            "Rankings are by model total P&L and are not expected returns; "
            "use them to check robustness (default near the top, smooth "
            "neighbors), not to pick a 'winner'.",
        ],
        "tickers": sweeps,
    }


def _sweep_row_md(row):
    s = row["summary"]
    mark = " ★ default" if row["is_default"] else ""
    if s["trades"] == 0:
        return (f"| {row['rank']} | `{row['key']}` | 0 | — | — | — | — | — |"
                f"{mark} |")
    return (f"| {row['rank']} | `{row['key']}` | {s['trades']} | "
            f"{s['win_rate']:.1%} | ${s['total_pnl']:+,.2f} | "
            f"${row['delta_vs_default']:+,.2f} | ${s['worst_trade']:+,.2f} | "
            f"${s['max_drawdown']:,.2f} | {s['avg_days_held']:.0f}{mark} |")


def render_sweep_markdown(payload):
    """Human-readable sweep report."""
    lines = ["# Options Screener — Exit-Rule Sensitivity Sweep (research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy (trailing realized vol x "
             "1.15) — this is NOT historical option-chain data. The sweep "
             "answers one question: are the config's exit rules robust, or "
             "does performance hinge on them?",
             "",
             f"**Strategy:** {payload['strategy']}",
             ""]
    for ticker, rows in payload["tickers"].items():
        lines.append(f"## {ticker}")
        lines.append("")
        lines.append("| Rank | Exit rules | Trades | Win rate | Total P&L | "
                     "Δ vs default | Worst | Max DD | Avg days |")
        lines.append("|------|------------|--------|----------|-----------|"
                     "--------------|-------|--------|----------|")
        for row in rows:
            lines.append(_sweep_row_md(row))
        best, default_rows = rows[0], [r for r in rows if r["is_default"]]
        lines.append("")
        lines.append(f"Best: `{best['key']}` at "
                     f"${best['summary']['total_pnl']:+,.2f} total. "
                     f"Default (`{default_rows[0]['key']}`) ranks "
                     f"#{default_rows[0]['rank']} of {len(rows)} "
                     f"(${default_rows[0]['summary']['total_pnl']:+,.2f}).")
        lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- Default near the top with smooth neighbors → the rules "
                 "are robust, not overfit to one setting.")
    lines.append("- Default far from the top, or a lone spike combo → treat "
                 "the settings as fragile; re-test on a different period "
                 "before trusting them.")
    lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skew-aware bull-put-spread backtest (v7)
#
# This closes the question deferred in v5 (2026-10-21): under flat
# Black-Scholes, NO width in the live spread range (2-6% of the short
# strike) passes the live screener's "collect at least 1/3 of the width"
# entry rule (measured 0/147 combos). The hope was that downside skew
# would rescue it. It does not — and the model shows why, plainly:
#
# Skew raises the LONG leg's IV more than the short leg's (the long
# strike sits further below spot), so the long leg costs MORE, net
# credit FALLS as skew rises, and the credit-to-width ratio falls with
# it. The earlier note's direction was backwards; this section corrects
# it with a test that asserts the direction.
#
# So what DOES pass the rule? Narrower widths (around 1% of the short
# strike, i.e. a $5-6 wing on a $600 underlying) and higher vol — the
# ratio climbs toward 0.30-0.33 there even at skew 0. The live 2-6%
# width band is therefore the binding constraint, not the vol model.
# That is a screener-design finding, not a trading recommendation:
# narrowing the band would also shrink max loss per trade and raise
# the number of wings that qualify, which the live screener should
# weigh before changing anything.
#
# Mechanics: the short leg is the same ~30-delta CSP strike as the plain
# backtest (found at the ATM IV proxy); the long leg sits width_pct
# below it. Both legs are priced at their own skewed IVs, and the
# spread is marked the same way every day. Exits mirror the CSP rules:
# close at profit_target_pct of max profit (spread value falls to
# (1 - target) x net credit), manage at manage_dte, manage when the
# cost to close reaches (1 + loss_manage_multiple) x net credit.
# At expiry the spread settles at its intrinsic value, capped at the
# width. Model-based like the rest of the backtester; signals only.
# ---------------------------------------------------------------------------

DEFAULT_SPREAD_BACKTEST = {
    "skews": [0.0, 1.0, 2.0, 4.0],
    "width_pct": 0.02,  # narrow end of the live 2-6% band: gives the
    # 1/3-width credit rule its best chance inside the live rules
}


def spread_value_at(price, short_strike, long_strike, years_to_expiry,
                    risk_free_rate, atm_iv, skew):
    """Model value of one bull-put-spread unit (short minus long, per share)."""
    short_v = bs_put_value_skewed(price, short_strike, years_to_expiry,
                                  risk_free_rate, atm_iv, skew)
    long_v = bs_put_value_skewed(price, long_strike, years_to_expiry,
                                 risk_free_rate, atm_iv, skew)
    return short_v - long_v


def simulate_spread_trade(prices, entry_day, dte, sigma_atm, cfg, skew,
                          width_pct, exits=None):
    """Simulate one bull-put-spread trade from entry through its first exit.

    Returns a trade dict in the same shape as simulate_trade (so
    summarize() works unchanged), plus spread fields: long_strike,
    width, net_credit_per_share, credit_fraction (net credit / width),
    passes_credit_rule (credit_fraction >= spread.min_credit_fraction),
    and max_loss_per_contract. Trades are recorded even when the credit
    rule fails — that is the point: the report counts how often the
    rule passes, rather than hiding the failures by never opening.
    """
    r = cfg["risk_free_rate"]
    strat = cfg["strategy"]
    ex = dict(cfg["exits"])
    if exits:
        ex.update(exits)
    expiry_day = entry_day + dte

    S = prices[entry_day]
    T = dte / 365.0
    short_strike = strike_for_delta(S, T, r, sigma_atm, strat["target_delta"])
    long_strike = short_strike * (1.0 - width_pct)
    width = short_strike - long_strike
    net_credit = spread_value_at(S, short_strike, long_strike, T, r,
                                 sigma_atm, skew)
    entry_delta = bs_put_delta(S, short_strike, T, r, sigma_atm)
    min_credit_fraction = cfg.get("spread", {}).get("min_credit_fraction", 0.33)

    profit_trigger = net_credit * (1 - ex["profit_target_pct"])
    loss_trigger = net_credit * (1 + ex["loss_manage_multiple"])

    reason, exit_day, exit_value = "expired", expiry_day, None
    for j in range(entry_day + 1, expiry_day + 1):
        t_rem = (expiry_day - j) / 365.0
        if t_rem <= 0:
            v = (max(short_strike - prices[j], 0.0)
                 - max(long_strike - prices[j], 0.0))
        else:
            v = spread_value_at(prices[j], short_strike, long_strike,
                                t_rem, r, sigma_atm, skew)
        if v <= profit_trigger:
            reason, exit_day, exit_value = "profit", j, v
            break
        if j < expiry_day and (expiry_day - j) <= ex["manage_dte"]:
            reason, exit_day, exit_value = "time", j, v
            break
        if j < expiry_day and v >= loss_trigger:
            reason, exit_day, exit_value = "loss", j, v
            break
    if exit_value is None:  # held to expiry: intrinsic spread value
        exit_value = (max(short_strike - prices[expiry_day], 0.0)
                      - max(long_strike - prices[expiry_day], 0.0))

    pnl_per_share = net_credit - exit_value
    return {
        "entry_day": entry_day,
        "exit_day": exit_day,
        "days_held": exit_day - entry_day,
        "entry_price": round(S, 2),
        "strike": round(short_strike, 2),
        "long_strike": round(long_strike, 2),
        "width": round(width, 2),
        "dte": dte,
        "iv": round(sigma_atm, 4),
        "skew": skew,
        "delta": round(entry_delta, 3),
        "premium_per_share": round(net_credit, 3),
        "net_credit_per_share": round(net_credit, 3),
        "credit_fraction": round(net_credit / width, 4) if width > 0 else None,
        "passes_credit_rule": bool(width > 0 and net_credit / width >= min_credit_fraction),
        "max_loss_per_contract": round(max(width - net_credit, 0.0) * 100, 2),
        "exit_reason": reason,
        "exit_value_per_share": round(exit_value, 3),
        "pnl_per_contract": round(pnl_per_share * 100, 2),
        # return on the defined-risk capital actually at stake
        "return_on_risk": round(pnl_per_share / max(width - net_credit, 1e-9), 4),
        "assigned": bool(reason == "expired" and exit_value > 0),
    }


def simulate_spreads(prices, cfg, skew, width_pct):
    """Run the spread strategy over a price series at one skew value."""
    bt = cfg["backtest"]
    dte = bt.get("entry_dte") or (cfg["strategy"]["dte_min"]
                                  + cfg["strategy"]["dte_max"]) // 2
    window, interval = bt["iv_window"], bt["entry_interval_days"]
    n = len(prices)

    trades = []
    day = window
    while day + dte < n:
        sigma = estimate_iv(prices, day, window,
                            bt["iv_multiplier"], bt["iv_fallback"])
        trade = simulate_spread_trade(prices, day, dte, sigma, cfg, skew,
                                      width_pct)
        trades.append(trade)
        day = max(trade["exit_day"] + 1, day + interval)
    return trades


def summarize_spreads(trades):
    """summarize() plus the spread-specific credit-rule stats."""
    s = summarize(trades)
    if trades:
        fracs = [t["credit_fraction"] for t in trades
                 if t["credit_fraction"] is not None]
        s["avg_credit_fraction"] = round(sum(fracs) / len(fracs), 4) if fracs else None
        s["credit_rule_pass_rate"] = round(
            sum(1 for t in trades if t["passes_credit_rule"]) / len(trades), 4)
    else:
        s["avg_credit_fraction"] = None
        s["credit_rule_pass_rate"] = None
    return s


def sweep_skew_spreads(prices, cfg, skews, width_pct):
    """Run the spread simulation once per skew value.

    Returns {skew: {"trades": [...], "summary": {...}}}. The entry rule,
    strike rule and exit rules are identical across skews — only the leg
    pricing changes. Later entries may shift a day or two because exit
    timing differs by skew, so per-skew trade counts can differ slightly;
    the credit-fraction comparison is per trade, which is what isolates
    the pricing effect.
    """
    out = {}
    for skew in skews:
        trades = simulate_spreads(prices, cfg, skew, width_pct)
        out[skew] = {"trades": trades, "summary": summarize_spreads(trades)}
    return out


def build_spread_payload(sweeps, cfg, skews, width_pct):
    """sweeps: {ticker: {skew: {...}}} -> JSON-serializable report payload."""
    return {
        "mode": "skew-aware bull-put-spread backtest",
        "skews": skews,
        "width_pct": width_pct,
        "strategy": "bull put spreads over ~30-delta CSP strikes, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS + [
            "Both spread legs are priced with Black-Scholes at their own "
            "skewed IVs: iv(K) = atm_iv * (1 + skew * max(0, log(S/K))). "
            "Skew is a shape parameter, not a fitted smile — no "
            "term-structure, no upside wing, no calibration to a real chain.",
            "The short strike is found at the ATM IV proxy (same as the "
            "plain CSP backtest); skew changes leg pricing only. Later "
            "entries may shift because exit timing differs by skew, so "
            "the clean comparison is the per-trade credit fraction, not "
            "the trade count.",
            f"Spread width is fixed at {width_pct:.0%} of the short strike "
            "(the narrow end of the live screener's 2-6% band) and the "
            "credit rule is the live one (net credit >= "
            f"{cfg.get('spread', {}).get('min_credit_fraction', 0.33):.0%} "
            "of width). Trades are recorded even when the rule fails, so "
            "the pass rate is measured, not assumed.",
            "A spread's max loss is capped at (width - net credit) by "
            "construction; return on risk is P&L over that capped capital, "
            "not over the full strike as in the CSP backtest.",
        ],
        "tickers": sweeps,
    }


def _spread_row_md(skew, res):
    s = res["summary"]
    if s["trades"] == 0:
        return f"| {skew} | 0 | — | — | — | — | — |"
    return (f"| {skew} | {s['trades']} | {s['win_rate']:.1%} | "
            f"${s['total_pnl']:+,.2f} | {s['avg_credit_fraction']:.3f} | "
            f"{s['credit_rule_pass_rate']:.0%} | ${s['max_drawdown']:,.2f} |")


def render_spread_markdown(payload):
    """Human-readable skew-aware spread backtest report."""
    lines = ["# Options Screener — Skew-Aware Spread Backtest (research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy — this is NOT historical "
             "option-chain data. Skew here is a one-parameter shape: "
             "iv(K) = atm_iv x (1 + skew x max(0, log(S/K))), not a "
             "fitted market smile. Results are illustrative research, not "
             "expected returns.",
             "",
             f"**Strategy:** {payload['strategy']}",
             f"**Spread width:** {payload['width_pct']:.0%} of the short "
             "strike. **Skews swept:** "
             + ", ".join(str(s) for s in payload["skews"]) + ".",
             ""]
    for ticker, per_skew in payload["tickers"].items():
        lines.append(f"## {ticker}")
        lines.append("")
        lines.append("| Skew | Trades | Win rate | Total P&L | Avg credit/width | "
                     "Credit-rule pass | Max DD |")
        lines.append("|------|--------|----------|-----------|------------------|"
                     "-------------------|--------|")
        for skew in payload["skews"]:
            lines.append(_spread_row_md(skew, per_skew[skew]))
        lines.append("")
        flat = per_skew[payload["skews"][0]]["summary"]
        steep = per_skew[payload["skews"][-1]]["summary"]
        if flat["trades"] and steep["trades"]:
            lines.append(
                "Reading: if avg credit/width FALLS as skew rises, skew is "
                "hurting the credit rule, not helping it — the long leg's "
                "IV rises faster than the short leg's, so it costs more "
                "and net credit shrinks. That is what the model shows at "
                f"this width: {flat['avg_credit_fraction']:.3f} at skew "
                f"{payload['skews'][0]} vs {steep['avg_credit_fraction']:.3f} "
                f"at skew {payload['skews'][-1]}. Narrower widths, not "
                "steeper skew, are what move the ratio toward the 1/3 rule.")
        lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)


# Spread-width sweep (v8)
#
# v7 fixed the width at 2% and swept skew; its note claimed narrower
# widths move the credit/width ratio toward the live 1/3 rule, but that
# was a single-width observation plus a directional argument, not a
# measurement. This sweep measures it: same skew, same entries/strikes/
# exits, only the wing width changes (1% up through the live 2-6%
# band). Two things are reported per width, because they pull opposite
# ways: the credit fraction (which should rise as the wing narrows)
# and the total P&L per contract (which should fall, because a
# narrower wing collects less absolute credit and caps max loss lower).
# The sweep therefore answers "where does the rule start passing?" and
# "what does passing cost?" together — it does NOT change the live
# screener's width band, which stays a decision for the owner.
# ---------------------------------------------------------------------------

DEFAULT_SPREAD_WIDTH_SWEEP = {
    "widths": [0.01, 0.02, 0.03, 0.04, 0.06],
    "skew": 2.0,  # ~= observed SPY put skew; the realistic case, not flat
}


def sweep_width_spreads(prices, cfg, widths, skew):
    """Run the spread simulation once per wing width at one skew value.

    Returns {width: {"trades": [...], "summary": {...}}}. Only the long
    leg moves (long strike = short strike x (1 - width)); entries,
    short strikes and exit rules are identical across widths, though
    later entries may shift a day or two because exit timing differs
    by width — the same caveat as the skew sweep.
    """
    out = {}
    for width in widths:
        trades = simulate_spreads(prices, cfg, skew, width)
        out[width] = {"trades": trades, "summary": summarize_spreads(trades)}
    return out


def _first_width_meeting(widths, sweeps, key, threshold):
    """First width (narrowest first) whose summary[key] meets threshold."""
    for w in sorted(widths):
        s = sweeps[w]["summary"]
        if s["trades"] and s[key] is not None and s[key] >= threshold:
            return w
    return None


def build_width_payload(sweeps, cfg, widths, skew):
    """sweeps: {ticker: {width: {...}}} -> JSON-serializable report payload."""
    min_frac = cfg.get("spread", {}).get("min_credit_fraction", 0.33)
    per_ticker = {}
    for ticker, per_width in sweeps.items():
        best_pnl = max(per_width, key=lambda w: (
            per_width[w]["summary"]["total_pnl"]
            if per_width[w]["summary"]["trades"] else float("-inf")))
        per_ticker[ticker] = {
            "sweep": per_width,
            "first_width_avg_credit_meets_rule": _first_width_meeting(
                widths, per_width, "avg_credit_fraction", min_frac),
            "first_width_pass_rate_meets_rule": _first_width_meeting(
                widths, per_width, "credit_rule_pass_rate", 0.5),
            "best_pnl_width": best_pnl,
        }
    return {
        "mode": "spread-width sweep backtest",
        "widths": widths,
        "skew": skew,
        "min_credit_fraction": min_frac,
        "strategy": "bull put spreads over ~30-delta CSP strikes, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS + [
            "Both spread legs are priced with Black-Scholes at their own "
            "skewed IVs: iv(K) = atm_iv * (1 + skew * max(0, log(S/K))). "
            f"Skew is held fixed at {skew} across the whole sweep, so "
            "width is the only thing varying.",
            "The short strike is found at the ATM IV proxy (same as the "
            "plain CSP backtest); only the long leg moves with width. "
            "Later entries may shift because exit timing differs by "
            "width, so the clean comparison is the per-trade credit "
            "fraction, not the trade count.",
            "Trades are recorded even when the credit rule fails, so "
            "the pass rate is measured, not assumed. A narrower wing "
            "raises the credit fraction but collects less absolute "
            "credit and caps max loss lower — the P&L column shows "
            "what that tradeoff costs on this period.",
        ],
        "tickers": per_ticker,
    }


def _width_row_md(width, res):
    s = res["summary"]
    if s["trades"] == 0:
        return f"| {width:.0%} | 0 | — | — | — | — | — |"
    return (f"| {width:.0%} | {s['trades']} | {s['win_rate']:.1%} | "
            f"${s['total_pnl']:+,.2f} | {s['avg_credit_fraction']:.3f} | "
            f"{s['credit_rule_pass_rate']:.0%} | ${s['max_drawdown']:,.2f} |")


def render_width_markdown(payload):
    """Human-readable spread-width sweep report."""
    lines = ["# Options Screener — Spread-Width Sweep Backtest (research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy — this is NOT historical "
             "option-chain data. Results are illustrative research, not "
             "expected returns.",
             "",
             f"**Strategy:** {payload['strategy']}",
             f"**Skew held fixed:** {payload['skew']}. "
             f"**Widths swept:** "
             + ", ".join(f"{w:.0%}" for w in payload["widths"])
             + f". **Credit rule:** net credit >= "
             f"{payload['min_credit_fraction']:.0%} of width.",
             ""]
    for ticker, info in payload["tickers"].items():
        per_width = info["sweep"]
        lines.append(f"## {ticker}")
        lines.append("")
        lines.append("| Width | Trades | Win rate | Total P&L | Avg credit/width | "
                     "Credit-rule pass | Max DD |")
        lines.append("|-------|--------|----------|-----------|------------------|"
                     "-------------------|--------|")
        for width in payload["widths"]:
            lines.append(_width_row_md(width, per_width[width]))
        lines.append("")
        first_avg = info["first_width_avg_credit_meets_rule"]
        best = info["best_pnl_width"]
        if first_avg is not None:
            lines.append(
                f"Reading: the average credit/width first reaches the "
                f"{payload['min_credit_fraction']:.0%} rule at a "
                f"{first_avg:.0%} wing (narrowest width meeting it on "
                f"average). Narrower wings raise the ratio but collect "
                f"less absolute credit — the best total P&L on this "
                f"period came from the {best:.0%} wing, which is the "
                f"tradeoff to weigh before touching the live 2-6% band.")
        else:
            lines.append(
                "Reading: no swept width averages the "
                f"{payload['min_credit_fraction']:.0%} credit rule on "
                "this period — even the narrowest wing falls short on "
                "average, so narrowing the live band alone would not "
                "make the rule pass here.")
        lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Earnings-aware backtest filter (v10)
#
# The live screener (v9) skips any signal whose option window contains a
# known earnings date, but the plain backtest traded straight through
# earnings windows — so its P&L described a strategy the screener would
# never actually run. This mode closes that gap: it re-runs the same
# simulation with the same gate and reports plain vs filtered side by
# side, so the cost (or saving) of skipping earnings windows is measured
# rather than assumed.
#
# Honest limitation, stated in the report: no free source publishes
# historical earnings calendars, so the backtester cannot fetch them.
# Earnings dates come from a caller-supplied calendar (a JSON file of
# ticker -> [ISO dates] for the CLI, or date lists in code). A ticker
# with no calendar dates runs unfiltered and says so — an empty calendar
# is a data gap, never evidence of no earnings. The window rule is the
# same as the live gate: entry date through entry date + DTE calendar
# days, inclusive on both ends (earnings_status from earnings.py).
#
# The gate is also config-gated like the live one: with earnings.enabled
# false or earnings.exclude false the filtered run equals the plain run
# (flag-only mode trades through, same as the live screener) and the
# report says the gate was inactive.
# ---------------------------------------------------------------------------

import json as _json
from datetime import timedelta as _timedelta

from .earnings import earnings_status as _earnings_status
from .earnings import parse_earnings_date as _parse_earnings_date


def make_earnings_gate(entry_dates, earnings_dates):
    """Build a gate callable for `_run_simulation`.

    `entry_dates`: one `date` per price index (trading-day dates, in
    order). `earnings_dates`: iterable of `date` (known announcement
    dates). Returns gate(entry_day, dte) -> the first earnings date
    inside [entry_date, entry_date + dte calendar days], or None.

    Raises ValueError if `entry_dates` is empty or its length differs
    from the price series it will gate — a misaligned calendar must fail
    loudly, not silently skip the wrong entries.
    """
    if not entry_dates:
        raise ValueError("entry_dates must not be empty")
    earn = sorted(d for d in earnings_dates if d is not None)

    def gate(entry_day, dte):
        if entry_day >= len(entry_dates):
            raise ValueError(
                f"entry_day {entry_day} beyond entry_dates "
                f"({len(entry_dates)} entries)")
        start = entry_dates[entry_day]
        end = start + _timedelta(days=dte)
        for e in earn:
            status, _ = _earnings_status(e, start, end)
            if status == "inside":
                return e
        return None

    return gate


def load_earnings_calendar(path):
    """Load a JSON earnings calendar: {ticker: ["YYYY-MM-DD", ...]}.

    Unparseable dates are dropped (a garbage calendar entry is an
    "unknown", same as the live gate's rule), and tickers with no usable
    dates map to an empty list. Raises on a missing file or non-object
    JSON — a calendar that cannot be read must not silently mean "no
    earnings anywhere".
    """
    with open(path) as f:
        raw = _json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("earnings calendar must be a JSON object "
                         "{ticker: [dates]}")
    calendar = {}
    for ticker, dates in raw.items():
        parsed = []
        if isinstance(dates, (list, tuple)):
            for d in dates:
                got = _parse_earnings_date(d)
                if got is not None:
                    parsed.append(got)
        calendar[ticker] = sorted(set(parsed))
    return calendar


def simulate_earnings_aware(prices, cfg, entry_dates, earnings_dates,
                            exits=None):
    """Run the simulation with the earnings gate applied.

    Returns {"trades", "skipped", "summary", "gate_active"} where
    "skipped" lists each gated entry (entry_day, dte, earnings_date).
    When the config gate is inactive (earnings.enabled false or
    earnings.exclude false) the run equals the plain simulation and
    "gate_active" is False. `entry_dates` must align with `prices`
    (same length); a mismatch raises ValueError.
    """
    ecfg = cfg.get("earnings", {}) or {}
    gate_active = bool(ecfg.get("enabled", True)
                       and ecfg.get("exclude", True))
    if not prices:
        return {"trades": [], "skipped": [], "summary": summarize([]),
                "gate_active": gate_active}
    if len(entry_dates) != len(prices):
        raise ValueError(
            f"entry_dates ({len(entry_dates)}) must align with prices "
            f"({len(prices)})")
    gate = (make_earnings_gate(entry_dates, earnings_dates)
            if gate_active else None)
    trades, skipped = _run_simulation(prices, cfg, exits=exits,
                                      earnings_gate=gate)
    return {"trades": trades, "skipped": skipped,
            "summary": summarize(trades), "gate_active": gate_active}


def build_earnings_payload(results, cfg):
    """results: {ticker: {"plain": {"trades", "summary"},
    "filtered": simulate_earnings_aware result}} -> report payload."""
    return {
        "mode": "Earnings-aware backtest",
        "strategy": "cash-secured puts: ~30 delta, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS + [
            "Earnings gate mirrors the live screener (v9): an entry is "
            "skipped when a KNOWN earnings date falls inside its option "
            "window (entry date through entry date + DTE calendar days, "
            "inclusive). A skipped entry is never opened; the next "
            "attempt moves up by entry_interval_days.",
            "Historical earnings dates are caller-supplied (no free "
            "source publishes them). A ticker with no calendar dates "
            "runs unfiltered and is flagged as such — an empty calendar "
            "is a data gap, never evidence of no earnings.",
            "This comparison is descriptive: the filtered run shows what "
            "the live gate would have done to past model trades. It does "
            "not prove skipping earnings helps or hurts next period.",
        ],
        "tickers": results,
    }


def render_earnings_markdown(payload):
    """Human-readable plain-vs-filtered earnings report."""
    lines = ["# Options Screener — Earnings-Aware Backtest (research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy (trailing realized vol x "
             "1.15) — this is NOT historical option-chain data. The "
             "filtered run applies the live screener's earnings gate; "
             "the plain run does not. Results are illustrative research, "
             "not expected returns. See Assumptions.",
             "",
             f"**Strategy:** {payload['strategy']}",
             ""]
    lines.append("| Ticker | Plain trades | Plain P&L | Filtered trades | "
                 "Filtered P&L | Skipped entries | Gate |")
    lines.append("|--------|--------------|-----------|-----------------|"
                 "--------------|-----------------|------|")
    for ticker, res in payload["tickers"].items():
        plain, filt = res["plain"]["summary"], res["filtered"]["summary"]
        gate = "active" if res["filtered"]["gate_active"] else "inactive"
        if not res["filtered"]["gate_active"]:
            skipped_note = "0 (gate off)"
        elif not res.get("calendar_dates"):
            skipped_note = f"{len(res['filtered']['skipped'])} (no calendar)"
        else:
            skipped_note = str(len(res["filtered"]["skipped"]))
        lines.append(
            f"| {ticker} | {plain['trades']} | "
            f"${plain['total_pnl']:+,.2f} | {filt['trades']} | "
            f"${filt['total_pnl']:+,.2f} | {skipped_note} | {gate} |")
    for ticker, res in payload["tickers"].items():
        filt = res["filtered"]
        if not filt["skipped"]:
            continue
        lines.append("")
        lines.append(f"## {ticker} — skipped entries")
        lines.append("")
        lines.append("| Entry day | DTE | Earnings date |")
        lines.append("|-----------|-----|---------------|")
        for s in filt["skipped"]:
            lines.append(f"| day {s['entry_day']} | {s['dte']} | "
                         f"{s['earnings_date']} |")
        plain, fsum = res["plain"]["summary"], filt["summary"]
        lines.append("")
        lines.append(f"Skipping those {len(filt['skipped'])} entries moved "
                     f"model P&L from ${plain['total_pnl']:+,.2f} "
                     f"({plain['trades']} trades) to "
                     f"${fsum['total_pnl']:+,.2f} ({fsum['trades']} "
                     f"trades) on this period — a description of the "
                     f"past, not a promise about the next earnings season.")
    lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)


# Earnings-aware spread backtest (v11)
#
# v9's live gate already covers spreads (the spread shares the short
# leg's expiry, so an excluded CSP signal never produces a spread),
# but the v7/v8 spread backtests kept trading straight through
# earnings windows — their P&L described spread trades the live
# screener would never have opened. This mode closes that gap the
# same way v10 did for plain CSPs: run the spread simulation twice
# per skew (plain, then with the v9 gate applied) and report both
# side by side. Per-skew entry timing can still shift after a skip
# (exit timing differs by skew, as in v7), so the clean comparison
# is the per-skew plain-vs-filtered pair, not across skews.
# ---------------------------------------------------------------------------

def simulate_spreads_earnings_aware(prices, cfg, skew, width_pct,
                                    entry_dates, earnings_dates):
    """Run the spread simulation with the earnings gate applied.

    Mirrors `simulate_earnings_aware` for spreads: returns {"trades",
    "skipped", "summary", "gate_active"} where "summary" is
    `summarize_spreads` (so avg credit/width and the credit-rule pass
    rate are reported for the filtered run too). A gated entry is
    never opened; the next attempt moves up by entry_interval_days.
    When the config gate is inactive the run equals the plain spread
    simulation and "gate_active" is False. `entry_dates` must align
    with `prices` (same length); a mismatch raises ValueError.
    """
    ecfg = cfg.get("earnings", {}) or {}
    gate_active = bool(ecfg.get("enabled", True)
                       and ecfg.get("exclude", True))
    if not prices:
        return {"trades": [], "skipped": [],
                "summary": summarize_spreads([]),
                "gate_active": gate_active}
    if len(entry_dates) != len(prices):
        raise ValueError(
            f"entry_dates ({len(entry_dates)}) must align with prices "
            f"({len(prices)})")
    gate = (make_earnings_gate(entry_dates, earnings_dates)
            if gate_active else None)

    bt = cfg["backtest"]
    dte = bt.get("entry_dte") or (cfg["strategy"]["dte_min"]
                                  + cfg["strategy"]["dte_max"]) // 2
    window, interval = bt["iv_window"], bt["entry_interval_days"]
    n = len(prices)

    trades, skipped = [], []
    day = window
    while day + dte < n:
        if gate is not None:
            hit = gate(day, dte)
            if hit is not None:
                skipped.append({"entry_day": day, "dte": dte,
                                "earnings_date": hit.isoformat()})
                day += interval
                continue
        sigma = estimate_iv(prices, day, window,
                            bt["iv_multiplier"], bt["iv_fallback"])
        trade = simulate_spread_trade(prices, day, dte, sigma, cfg, skew,
                                      width_pct)
        trades.append(trade)
        day = max(trade["exit_day"] + 1, day + interval)
    return {"trades": trades, "skipped": skipped,
            "summary": summarize_spreads(trades),
            "gate_active": gate_active}


def build_spread_earnings_payload(results, cfg, skews, width_pct):
    """results: {ticker: {"skews": {skew: {"plain": {"trades", "summary"},
    "filtered": simulate_spreads_earnings_aware result}},
    "calendar_dates": int}} -> report payload."""
    return {
        "mode": "Earnings-aware spread backtest",
        "skews": skews,
        "width_pct": width_pct,
        "strategy": "bull put spreads over ~30-delta CSP strikes, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS + [
            "Earnings gate mirrors the live screener (v9): an entry is "
            "skipped when a KNOWN earnings date falls inside its option "
            "window (entry date through entry date + DTE calendar days, "
            "inclusive). A skipped spread entry is never opened; the "
            "next attempt moves up by entry_interval_days.",
            "Historical earnings dates are caller-supplied (no free "
            "source publishes them). A ticker with no calendar dates "
            "runs unfiltered and is flagged as such — an empty calendar "
            "is a data gap, never evidence of no earnings.",
            "This comparison is descriptive: the filtered run shows what "
            "the live gate would have done to past model spread trades. "
            "It does not prove skipping earnings helps or hurts next "
            "period, and per-skew entry timing may shift after a skip, "
            "so compare plain vs filtered within one skew.",
        ],
        "tickers": results,
    }


def render_spread_earnings_markdown(payload):
    """Human-readable plain-vs-filtered spread report, one table per ticker."""
    lines = ["# Options Screener — Earnings-Aware Spread Backtest "
             "(research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy — this is NOT historical "
             "option-chain data. The filtered run applies the live "
             "screener's earnings gate to spread entries; the plain run "
             "does not. Results are illustrative research, not expected "
             "returns. See Assumptions.",
             "",
             f"**Strategy:** {payload['strategy']}",
             f"**Spread width:** {payload['width_pct']:.0%} of the short "
             "strike. **Skews:** "
             + ", ".join(str(s) for s in payload["skews"]) + ".",
             ""]
    for ticker, res in payload["tickers"].items():
        lines.append(f"## {ticker}")
        lines.append("")
        lines.append("| Skew | Plain trades | Plain P&L | Filtered trades | "
                     "Filtered P&L | Skipped | Credit/width (plain) | Gate |")
        lines.append("|------|--------------|-----------|-----------------|"
                     "--------------|---------|----------------------|------|")
        for skew in payload["skews"]:
            pair = res["skews"][skew]
            plain, filt = pair["plain"]["summary"], pair["filtered"]["summary"]
            gate = "active" if pair["filtered"]["gate_active"] else "inactive"
            if not pair["filtered"]["gate_active"]:
                skipped_note = "0 (gate off)"
            elif not res.get("calendar_dates"):
                skipped_note = (f"{len(pair['filtered']['skipped'])} "
                                "(no calendar)")
            else:
                skipped_note = str(len(pair["filtered"]["skipped"]))
            cf = plain["avg_credit_fraction"]
            cf_cell = f"{cf:.3f}" if cf is not None else "—"
            lines.append(
                f"| {skew} | {plain['trades']} | "
                f"${plain['total_pnl']:+,.2f} | {filt['trades']} | "
                f"${filt['total_pnl']:+,.2f} | {skipped_note} | "
                f"{cf_cell} | {gate} |")
        for skew in payload["skews"]:
            filt = res["skews"][skew]["filtered"]
            if not filt["skipped"]:
                continue
            lines.append("")
            lines.append(f"### {ticker} — skipped spread entries (skew {skew})")
            lines.append("")
            lines.append("| Entry day | DTE | Earnings date |")
            lines.append("|-----------|-----|---------------|")
            for s in filt["skipped"]:
                lines.append(f"| day {s['entry_day']} | {s['dte']} | "
                             f"{s['earnings_date']} |")
        lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Regime-aware forward entry filter (v16)
#
# v6 split past trades at the median entry IV and v15 showed the
# high-IV regime made more on 13 of 15 tickers — but both reports
# carried the same warning: the median is only known AFTER the whole
# period, so neither split is a rule anyone could have traded. This
# mode tests the tradable version of the idea, and says plainly
# whether it survives contact with causality.
#
# The rule: at each entry attempt, compare today's entry IV (the
# same trailing-realized-vol proxy the rest of the backtester uses)
# with the MEDIAN of that same proxy over the trailing lookback_days
# trading days (default 252, about one year), computed only from data
# available at entry time — the threshold day range ends the day
# BEFORE the entry, so today's IV never leaks into its own threshold.
# Enter only when entry IV is strictly ABOVE the trailing median.
# The first iv_window + lookback_days days are warmup and never trade,
# so the threshold is always defined and always causal.
#
# Two things the report does NOT do: it does not grid-search the
# lookback (one rule, chosen once, tested once — a sweep would turn
# this into an optimizer, and v5 already showed why that matters),
# and it does not recommend the filter. The plain run shares the
# filtered run's warmup start, so the P&L delta isolates the filter
# itself — not the later start — and the report prints the skipped
# count and the trade share alongside the delta, because a filter
# that earns more by trading half as often is a different claim from
# one that earns more by trading better.
#
# Honest limits, carried in the payload and the report: entry IV is
# the trailing-realized-vol proxy, not traded implied vol; in a
# stationary market the filter passes about half of all attempts by
# construction (a median splits attempts in half), so a positive P&L
# delta with a ~50% trade share means the skipped half lost money
# on this period — it does not promise the next period's low-IV
# entries lose. A sustained high-vol regime adapts the threshold
# upward until entries stop passing, which is the correct behavior
# of a trailing median and also its weakness: the filter chases
# relative richness, not absolute richness.
# ---------------------------------------------------------------------------

DEFAULT_REGIME_FILTER = {
    "lookback_days": 252,  # trailing trading days for the entry-IV median
    "mode": "above",       # v16 tests only "above": enter when entry IV is
    # above its trailing median. "below" is an untested variant, not a rule.
}

MIN_FILTER_TRADES = 5   # filtered side needs this many trades to be readable
MIN_PLAIN_TRADES = 10   # plain side needs this many trades to be readable


def trailing_iv_median(prices, day, lookback_days, window, multiplier,
                       fallback):
    """Median of the daily IV proxy over the trailing `lookback_days` days.

    The window is [day - lookback_days, day): it ends the day BEFORE the
    entry, so today's IV never leaks into its own threshold. Returns
    None when there is not enough history (day < lookback_days).
    Raises ValueError when lookback_days < 1.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if day < lookback_days:
        return None
    ivs = [estimate_iv(prices, d, window, multiplier, fallback)
           for d in range(day - lookback_days, day)]
    return _median(ivs)


def simulate_regime_filtered(prices, cfg, lookback_days=None, exits=None):
    """Run the simulation entering only above the trailing IV median.

    Returns {"trades", "skipped", "summary", "lookback_days",
    "warmup_days", "mode"}. A skipped entry is recorded with its entry
    IV and the threshold it failed, and the next attempt moves up by
    `entry_interval_days` only. No trade is ever opened before
    warmup (iv_window + lookback_days), so the threshold is always
    defined and always causal. `lookback_days` overrides
    cfg["regime_filter"]["lookback_days"]; raises ValueError for a
    lookback below 1. An empty or too-short series returns zero trades
    and zero skips, never an error.
    """
    rf = cfg.get("regime_filter") or DEFAULT_REGIME_FILTER
    lookback = (lookback_days if lookback_days is not None
                else rf.get("lookback_days",
                            DEFAULT_REGIME_FILTER["lookback_days"]))
    if lookback < 1:
        raise ValueError("lookback_days must be >= 1")
    bt = cfg["backtest"]
    dte = bt.get("entry_dte") or (cfg["strategy"]["dte_min"]
                                  + cfg["strategy"]["dte_max"]) // 2
    window, interval = bt["iv_window"], bt["entry_interval_days"]
    n = len(prices)

    warmup = window + lookback
    trades, skipped = [], []
    day = warmup
    while day + dte < n:
        sigma = estimate_iv(prices, day, window,
                            bt["iv_multiplier"], bt["iv_fallback"])
        threshold = trailing_iv_median(prices, day, lookback, window,
                                       bt["iv_multiplier"],
                                       bt["iv_fallback"])
        if threshold is None or sigma <= threshold:
            skipped.append({"entry_day": day, "dte": dte,
                            "entry_iv": round(sigma, 4),
                            "threshold_iv": (round(threshold, 4)
                                             if threshold is not None
                                             else None)})
            day += interval
            continue
        trade = simulate_trade(prices, day, dte, sigma, cfg, exits=exits)
        trades.append(trade)
        day = max(trade["exit_day"] + 1, day + interval)
    return {"trades": trades, "skipped": skipped,
            "summary": summarize(trades), "lookback_days": lookback,
            "warmup_days": warmup, "mode": "above"}


def compare_regime_filter(results):
    """Compare plain vs regime-filtered runs, per ticker + cross-ticker.

    `results`: {ticker: {"plain": {"trades", "summary"},
    "filtered": simulate_regime_filtered result}} -> {"rows": [...],
    "summary": {...}}. Rows are ranked by P&L delta (filtered minus
    plain), highest first. A row is "readable" when the plain side has
    at least MIN_PLAIN_TRADES trades and the filtered side at least
    MIN_FILTER_TRADES; thin rows are printed but never count toward
    the summary. A filtered run with zero trades still reports its
    P&L delta (0 minus the plain total) — the filter skipped
    everything, which the skipped count makes visible — but its
    win-rate and drawdown gaps are None, never fake zeros.
    """
    rows = []
    for ticker, res in results.items():
        plain, filt = res["plain"]["summary"], res["filtered"]["summary"]
        p_trades, f_trades = plain["trades"], filt["trades"]
        readable = (p_trades >= MIN_PLAIN_TRADES
                    and f_trades >= MIN_FILTER_TRADES)
        win_gap = (round(filt["win_rate"] - plain["win_rate"], 4)
                   if p_trades and f_trades
                   and filt["win_rate"] is not None
                   and plain["win_rate"] is not None else None)
        dd_gap = (round(filt["max_drawdown"] - plain["max_drawdown"], 2)
                  if p_trades and f_trades else None)
        rows.append({
            "ticker": ticker,
            "plain_trades": p_trades,
            "plain_win_rate": plain["win_rate"],
            "plain_total_pnl": plain["total_pnl"],
            "plain_max_drawdown": plain["max_drawdown"],
            "filtered_trades": f_trades,
            "filtered_win_rate": filt["win_rate"],
            "filtered_total_pnl": filt["total_pnl"],
            "filtered_max_drawdown": filt["max_drawdown"],
            "skipped_entries": len(res["filtered"]["skipped"]),
            "trade_share": (round(f_trades / p_trades, 4)
                            if p_trades else None),
            "pnl_delta": (round(filt["total_pnl"] - plain["total_pnl"], 2)
                          if p_trades else None),
            "win_gap": win_gap,
            "dd_gap": dd_gap,
            "thin": not readable,
        })
    rows.sort(key=lambda r: (r["pnl_delta"] is None,
                             -(r["pnl_delta"] or 0)))
    readable_rows = [r for r in rows if not r["thin"]]
    summary = {
        "tickers_compared": len(rows),
        "readable_tickers": len(readable_rows),
        "filtered_wins_pnl": sum(1 for r in readable_rows
                                 if r["pnl_delta"] is not None
                                 and r["pnl_delta"] > 0),
        "filtered_wins_win_rate": sum(1 for r in readable_rows
                                      if r["win_gap"] is not None
                                      and r["win_gap"] > 0),
        "median_pnl_delta": _median([r["pnl_delta"] for r in readable_rows
                                     if r["pnl_delta"] is not None]),
        "median_win_gap": _median([r["win_gap"] for r in readable_rows
                                   if r["win_gap"] is not None]),
        "median_dd_gap": _median([r["dd_gap"] for r in readable_rows
                                  if r["dd_gap"] is not None]),
        "median_trade_share": _median([r["trade_share"] for r in readable_rows
                                       if r["trade_share"] is not None]),
    }
    return {"rows": rows, "summary": summary}


def build_regime_filter_payload(results, cfg, lookback_days):
    """results: {ticker: {"plain": {...}, "filtered": {...}}} -> payload."""
    compared = compare_regime_filter(results)
    return {
        "mode": "Regime-aware forward entry filter",
        "lookback_days": lookback_days,
        "strategy": "cash-secured puts: ~30 delta, "
                    f"{cfg['backtest'].get('entry_dte') or 'mid-band'} DTE",
        "assumptions": ASSUMPTIONS + [
            "The filter enters only when the entry IV is strictly above "
            f"the median of the same IV proxy over the trailing "
            f"{lookback_days} trading days, computed only from data "
            "available at entry time (the threshold window ends the day "
            "before the entry). The first iv_window + lookback_days days "
            "are warmup and never trade.",
            "The plain run shares the filtered run's warmup start, so "
            "both runs attempt entries over the same days and the P&L "
            "delta isolates the filter itself, not the later start.",
            "Entry IV is the trailing-realized-vol proxy (x 1.15), not "
            "traded implied vol; a high-IV entry also means larger "
            "premiums AND larger moves. In a stationary market the "
            "filter passes about half of all attempts by construction "
            "(a median splits attempts in half).",
            "One rule, chosen once, tested once: the lookback was not "
            "grid-searched, so this is a forward test of a hypothesis, "
            "not an optimizer result. Re-test on a different period "
            "before trusting any delta here.",
            "A row is readable when the plain side has at least "
            f"{MIN_PLAIN_TRADES} trades and the filtered side at least "
            f"{MIN_FILTER_TRADES}; thin rows are printed but never "
            "count toward the cross-ticker summary.",
        ],
        "rows": compared["rows"],
        "summary": compared["summary"],
    }


def render_regime_filter_markdown(payload):
    """Human-readable plain-vs-filtered regime-filter report."""
    lines = ["# Options Screener — Regime-Aware Forward Entry Filter "
             "(research only)",
             "",
             "> **Model-based simulation.** Premiums are priced with "
             "Black-Scholes against an IV proxy (trailing realized vol x "
             "1.15) — this is NOT historical option-chain data. v6/v15 "
             "showed high-IV entries beat low-IV entries *descriptively*, "
             "but their median is only known after the whole period. "
             "This mode tests the tradable version: enter only when "
             "today's entry IV is above the median of the same proxy "
             "over the trailing "
             f"{payload['lookback_days']} trading days, computed only "
             "from data available at entry time. No look-ahead: the "
             "first warmup days never trade.",
             "",
             f"**Strategy:** {payload['strategy']}",
             f"**Filter:** enter only when entry IV > trailing "
             f"{payload['lookback_days']}-day median of the entry-IV "
             "proxy. **Plain run:** same warmup start, no filter — the "
             "P&L delta isolates the filter, not the later start.",
             ""]
    s = payload["summary"]
    lines.append("## Cross-ticker summary")
    lines.append("")
    lines.append(f"- {s['tickers_compared']} tickers compared, "
                 f"{s['readable_tickers']} with enough trades on both "
                 "sides to read.")
    if s["readable_tickers"]:
        lines.append(f"- The filtered run made more than the plain run on "
                     f"{s['filtered_wins_pnl']} of {s['readable_tickers']} "
                     "readable tickers, and won more often on "
                     f"{s['filtered_wins_win_rate']} of "
                     f"{s['readable_tickers']}.")
        lines.append(f"- Median P&L delta (filtered minus plain): "
                     f"${s['median_pnl_delta']:+,.2f} per contract; "
                     f"median win-rate gap: {s['median_win_gap']:+.1%}; "
                     f"median max-drawdown gap: "
                     f"${s['median_dd_gap']:+,.2f} (positive means the "
                     "filtered run drew down further).")
        lines.append(f"- Median trade share: {s['median_trade_share']:.0%} "
                     "of the plain run's trades — in a stationary market "
                     "the filter passes about half of all attempts by "
                     "construction, so a positive delta with a ~50% share "
                     "means the skipped half lost money on this period.")
    else:
        lines.append("- No ticker has enough trades on both sides to "
                     "summarize — every row is thin. Run more years or "
                     "more tickers before reading anything into this.")
    lines.append("")
    lines.append("## Per-ticker results (ranked by P&L delta)")
    lines.append("")
    lines.append("| Ticker | Plain trades | Plain P&L | Filtered trades | "
                 "Filtered P&L | Skipped | Trade share | P&L Δ | Win Δ | "
                 "Max-DD Δ | Read |")
    lines.append("|--------|--------------|-----------|-----------------|"
                 "--------------|---------|-------------|-------|-------|"
                 "----------|------|")
    for r in payload["rows"]:
        lines.append(
            f"| {r['ticker']} | {r['plain_trades']} | "
            f"${r['plain_total_pnl']:+,.2f} | {r['filtered_trades']} | "
            f"${r['filtered_total_pnl']:+,.2f} | {r['skipped_entries']} | "
            f"{_compare_cell(r['trade_share'], '{:.0%}')} | "
            f"{_compare_cell(r['pnl_delta'], '${:+,.2f}')} | "
            f"{_compare_cell(r['win_gap'], '{:+.1%}')} | "
            f"{_compare_cell(r['dd_gap'], '${:+,.2f}')} | "
            f"{'thin ⚠' if r['thin'] else 'readable'} |")
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- Read the trade share with the delta: a positive delta "
                 "at a 50% share means the filter earned more while "
                 "trading half as often — the skipped entries were the "
                 "losers on this period. A negative delta at a low share "
                 "means it skipped winners too, and sat out while the "
                 "plain run worked.")
    lines.append("- A sustained high-vol regime adapts the threshold "
                 "upward until entries stop passing: that is the filter "
                 "working as designed (relative richness, not absolute), "
                 "and also its weakness — check the skipped count before "
                 "concluding anything from a 0-trade filtered row.")
    lines.append("- One period, one lookback, no grid search: a positive "
                 "median delta is a hypothesis that survived one forward "
                 "test, not a rule. Re-test on a different period before "
                 "touching the live screener — and the live screener "
                 "takes no entry filter from this mode until that "
                 "re-test happens.")
    lines.append("")
    lines.append("## Assumptions & limitations")
    lines.append("")
    for a in payload["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("_Educational research tool, not financial advice. Options "
                 "involve substantial risk of loss; paper-trade before "
                 "risking capital._")
    lines.append("")
    return "\n".join(lines)
