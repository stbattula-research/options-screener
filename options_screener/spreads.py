"""Bull put spread construction: defined-risk credit spreads over CSP candidates.

For each cash-secured-put candidate (the short leg), pair it with a cheaper
long put on the same expiry. The long leg caps the worst case: max loss
becomes (spread width - net credit) instead of the full strike.

Entry rules (all configurable in config.yaml under `spread:`):
- the long strike must sit between min_width_pct and max_width_pct of the
  short strike below it (keeps the wing sane relative to the underlying)
- the long leg must pass the same liquidity filters as the short leg
- net credit received must be >= min_credit_fraction of the spread width
  (the classic "collect at least a third of the width" rule of thumb)

Spreads are ranked by return on risk (max profit / max loss), best first.
Signals only — this builds candidates, never trades.
"""

import math

from .filters import liquidity_ok, quote_stats
from .vol import annualized_yield, bs_put_delta


def find_spreads(puts, short_candidates, underlying: float, expiry: str,
                 dte: int, cfg: dict) -> tuple[list, list]:
    """Pair each short-put candidate with the best long leg.

    puts: chain DataFrame (all rows; long legs are read from it).
    short_candidates: output of signals.find_candidates (needs strike, mid).
    Returns (spreads, rejected): spreads ranked by ROI desc; rejected is
    (short strike, reason) for strikes with no qualifying long leg.
    """
    sp = cfg["spread"]
    r = cfg["risk_free_rate"]
    years = dte / 365.0
    spreads, rejected = [], []

    # Index chain rows by strike for long-leg lookup.
    rows = {}
    for _, row in puts.iterrows():
        try:
            strike = float(row["strike"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(strike) and strike > 0:
            rows[strike] = row

    for short in short_candidates:
        short_strike = short["strike"]
        short_mid = short["mid"]
        min_w = short_strike * sp["min_width_pct"]
        max_w = short_strike * sp["max_width_pct"]
        best = None

        for long_strike, lrow in rows.items():
            width = short_strike - long_strike
            if width < min_w or width > max_w:
                continue
            ok, _ = liquidity_ok(lrow, cfg)
            if not ok:
                continue
            q = quote_stats(lrow)
            if q is None:
                continue
            net_credit = short_mid - q["mid"]
            if net_credit <= 0:
                continue
            if net_credit < sp["min_credit_fraction"] * width:
                continue

            max_loss = width - net_credit
            if max_loss <= 0:
                continue
            roi = net_credit / max_loss
            ann_roi = annualized_yield(roi, dte)

            try:
                liv = float(lrow["impliedVolatility"])
                long_delta = (bs_put_delta(underlying, long_strike, years, r, liv)
                              if math.isfinite(liv) and liv > 0 else None)
            except (KeyError, TypeError, ValueError):
                long_delta = None

            if best is None or roi > best["roi"]:
                best = {
                    "short_strike": round(short_strike, 2),
                    "short_mid": round(short_mid, 2),
                    "short_delta": short.get("delta"),
                    "long_strike": round(long_strike, 2),
                    "long_delta": round(long_delta, 3) if long_delta is not None else None,
                    "long_bid": round(q["bid"], 2),
                    "long_ask": round(q["ask"], 2),
                    "long_mid": round(q["mid"], 2),
                    "width": round(width, 2),
                    "net_credit": round(net_credit, 2),
                    "max_loss": round(max_loss, 2),
                    "max_profit": round(net_credit, 2),
                    "breakeven": round(short_strike - net_credit, 2),
                    "roi": round(roi, 4),
                    "annualized_roi": round(ann_roi, 4) if ann_roi is not None else None,
                    "expiry": expiry,
                    "dte": dte,
                }

        if best is None:
            if len(rejected) < 3:
                rejected.append((short_strike, "no long leg met spread rules"))
            continue
        spreads.append(best)

    spreads.sort(key=lambda s: s["roi"], reverse=True)
    return spreads, rejected


def spread_exit_plan(spread: dict, cfg: dict) -> dict:
    """Mechanical exit rules for a bull put spread (guidance, not orders)."""
    exits = cfg["exits"]
    return {
        "profit_target": (f"Close at {exits['profit_target_pct']:.0%} of max profit "
                          f"(~${spread['max_profit'] * exits['profit_target_pct']:.2f}/share)"),
        "time_exit": f"Exit or roll at {exits['manage_dte']} DTE",
        "loss_manage": (f"Manage if the cost to close reaches "
                        f"{exits['loss_manage_multiple']:.0f}x net credit received "
                        f"(~${spread['net_credit'] * exits['loss_manage_multiple']:.2f}/share); "
                        f"max loss is capped at ${spread['max_loss']:.2f}/share by construction"),
    }
