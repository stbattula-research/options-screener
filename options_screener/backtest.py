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

from .vol import bs_put_delta, bs_put_value, realized_vol


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


def simulate(prices, cfg, exits=None):
    """Run the strategy over a price series. Returns a list of trade dicts.

    Entries are attempted every `entry_interval_days` trading days (one
    position at a time), each needing `entry_dte` of future data to resolve.
    Trades that cannot complete are never opened. `exits` overrides the exit
    rules from config (sensitivity sweep), without mutating cfg.
    """
    bt = cfg["backtest"]
    dte = bt.get("entry_dte") or (cfg["strategy"]["dte_min"]
                                  + cfg["strategy"]["dte_max"]) // 2
    window, interval = bt["iv_window"], bt["entry_interval_days"]
    n = len(prices)

    trades = []
    day = window  # first entry needs trailing history for the IV proxy
    while day + dte < n:
        sigma = estimate_iv(prices, day, window,
                            bt["iv_multiplier"], bt["iv_fallback"])
        trade = simulate_trade(prices, day, dte, sigma, cfg, exits=exits)
        trades.append(trade)
        day = max(trade["exit_day"] + 1, day + interval)
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
