"""Liquidity and strategy filters for candidate short puts.

A row passes only if every gate passes; the first failing gate's reason is
recorded so the report can say *why* a ticker produced no signal.
"""

import math


def quote_stats(row) -> dict | None:
    """Mid price and spread width from a chain row. None if unquotable."""
    try:
        bid = float(row["bid"])
        ask = float(row["ask"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(bid) and math.isfinite(ask)) or ask <= 0 or bid < 0:
        return None
    if bid > ask:  # crossed market: bad quote
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return {"bid": bid, "ask": ask, "mid": mid,
            "spread_pct": (ask - bid) / mid}


def liquidity_ok(row, cfg: dict) -> tuple[bool, str]:
    """Check open interest, volume and bid-ask spread. Returns (ok, reason)."""
    liq = cfg["liquidity"]
    try:
        oi = float(row.get("openInterest", 0) or 0)
        vol = float(row.get("volume", 0) or 0)
    except (TypeError, ValueError):
        return False, "missing OI/volume data"
    if oi < liq["min_open_interest"]:
        return False, f"open interest {oi:.0f} < {liq['min_open_interest']}"
    if vol < liq["min_volume"]:
        return False, f"volume {vol:.0f} < {liq['min_volume']}"
    q = quote_stats(row)
    if q is None:
        return False, "no usable bid/ask quote"
    if q["spread_pct"] > liq["max_spread_pct"]:
        return False, (f"spread {q['spread_pct']:.1%} > "
                       f"{liq['max_spread_pct']:.0%} of mid")
    return True, ""


def strategy_ok(delta: float, premium_pct: float, cfg: dict) -> tuple[bool, str]:
    """Check delta band and minimum premium for the CSP strategy."""
    strat = cfg["strategy"]
    if abs(delta - strat["target_delta"]) > strat["delta_tolerance"]:
        return False, (f"delta {delta:.2f} outside "
                       f"{strat['target_delta']}{strat['delta_tolerance']:+.2f}")
    if premium_pct < strat["min_premium_pct"]:
        return False, (f"premium {premium_pct:.2%} of strike < "
                       f"{strat['min_premium_pct']:.2%} minimum")
    return True, ""
