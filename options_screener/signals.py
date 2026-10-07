"""Signal selection: pick the best cash-secured-put candidate per ticker."""

import math

from .filters import liquidity_ok, quote_stats, strategy_ok
from .vol import annualized_yield, bs_put_delta, iv_vs_hv_rank


def find_candidates(puts, underlying: float, expiry: str, dte: int,
                    hv_stats, cfg: dict) -> tuple[list, list]:
    """Scan a put chain; return (signals, rejected).

    signals: list of dicts, best first (ranked by annualized yield).
    rejected: list of (strike, reason) for the first few near-misses, for the
    report's transparency section.
    """
    strat = cfg["strategy"]
    r = cfg["risk_free_rate"]
    years = dte / 365.0
    signals, rejected = [], []

    for _, row in puts.iterrows():
        try:
            strike = float(row["strike"])
            iv = float(row["impliedVolatility"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(strike) and math.isfinite(iv)) or strike <= 0:
            continue

        ok, reason = liquidity_ok(row, cfg)
        if not ok:
            if len(rejected) < 3:
                rejected.append((strike, reason))
            continue
        q = quote_stats(row)
        premium_pct = q["mid"] / strike

        try:
            delta = bs_put_delta(underlying, strike, years, r, iv)
        except ValueError:
            continue
        ok, reason = strategy_ok(delta, premium_pct, cfg)
        if not ok:
            if len(rejected) < 3:
                rejected.append((strike, reason))
            continue

        iv_rank = iv_vs_hv_rank(iv, hv_stats[0], hv_stats[1]) if hv_stats else None
        ann = annualized_yield(premium_pct, dte)
        signals.append({
            "strike": round(strike, 2),
            "expiry": expiry,
            "dte": dte,
            "delta": round(delta, 3),
            "iv": round(iv, 4),
            "iv_vs_hv_rank": round(iv_rank, 3) if iv_rank is not None else None,
            "bid": round(q["bid"], 2),
            "ask": round(q["ask"], 2),
            "mid": round(q["mid"], 2),
            "spread_pct": round(q["spread_pct"], 4),
            "volume": int(row.get("volume", 0) or 0),
            "open_interest": int(row.get("openInterest", 0) or 0),
            "premium_pct": round(premium_pct, 4),
            "annualized_yield": round(ann, 4) if ann is not None else None,
            "max_profit_per_share": round(q["mid"], 2),
            "breakeven": round(strike - q["mid"], 2),
        })

    signals.sort(key=lambda s: s["annualized_yield"] or 0, reverse=True)
    return signals, rejected


def exit_plan(signal: dict, cfg: dict) -> dict:
    """Mechanical exit rules for a signal (guidance, not orders)."""
    exits = cfg["exits"]
    return {
        "profit_target": (f"Close at {exits['profit_target_pct']:.0%} of max profit "
                          f"(~${signal['max_profit_per_share'] * exits['profit_target_pct']:.2f}/share)"),
        "time_exit": f"Exit or roll at {exits['manage_dte']} DTE",
        "loss_manage": (f"Manage the trade if the loss reaches "
                        f"{exits['loss_manage_multiple']:.0f}x premium received "
                        f"(~${signal['max_profit_per_share'] * exits['loss_manage_multiple']:.2f}/share)"),
    }
