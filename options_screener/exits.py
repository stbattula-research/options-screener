"""Paper position tracking: the signal -> alert -> exit loop as research.

Signals and alerts answer "what looks good today"; this module answers
"what happened to the ones we paper-tracked". It opens paper positions from
newly fired alerts, marks every open position against each day's option
chain, and evaluates them against the SAME mechanical exit rules the
screener publishes (`exits:` in config.yaml): close at 50% of max profit,
at 21 DTE, or when the loss reaches 2x the premium received.

Paper positions only. This module never places orders, holds no broker
credentials, and writes state to a local git-ignored directory. It is a
research simulation, not a trade recommendation engine.

Position keys reuse the alert dedupe keys (ticker|expiry|strike), so an
alert can only ever open one paper position, and a closed position is
never silently re-opened by a later alert for the same candidate.
"""

import json
import math
import os
from datetime import date

from .filters import quote_stats

STATE_FILENAME = "positions.json"

# Marking: a leg is only markable when today's chain still quotes it.
MARK_STALE = "stale"


def csp_key(ticker: str, expiry: str, strike: float) -> str:
    return f"{ticker}|{expiry}|{strike}"


def spread_key(ticker: str, expiry: str,
               short_strike: float, long_strike: float) -> str:
    return f"{ticker}|{expiry}|{short_strike}/{long_strike}"


def open_from_signal(sig: dict, opened: str | None = None) -> dict:
    """Build a paper CSP position from a screener signal dict."""
    opened = opened or date.today().isoformat()
    credit = float(sig["mid"])
    return {
        "key": csp_key(sig["ticker"], sig["expiry"], sig["strike"]),
        "kind": "csp",
        "ticker": sig["ticker"],
        "strike": float(sig["strike"]),
        "expiry": sig["expiry"],
        "contracts": 1,
        "opened": opened,
        "entry_credit": round(credit, 2),      # per-share premium received
        "max_profit": round(credit, 2),        # per-share max profit
        "entry_underlying": sig.get("underlying"),
        "entry_dte": sig["dte"],
        "entry_delta": sig.get("delta"),
        "breakeven": sig.get("breakeven"),
        "status": "open",
    }


def open_from_spread(sp: dict, opened: str | None = None) -> dict:
    """Build a paper bull-put-spread position from a spread dict."""
    opened = opened or date.today().isoformat()
    credit = float(sp["net_credit"])
    return {
        "key": spread_key(sp["ticker"], sp["expiry"],
                          sp["short_strike"], sp["long_strike"]),
        "kind": "spread",
        "ticker": sp["ticker"],
        "short_strike": float(sp["short_strike"]),
        "long_strike": float(sp["long_strike"]),
        "expiry": sp["expiry"],
        "contracts": 1,
        "opened": opened,
        "entry_credit": round(credit, 2),      # per-share net credit received
        "max_profit": round(credit, 2),
        "max_loss": float(sp["max_loss"]),     # per-share worst case
        "entry_underlying": sp.get("underlying"),
        "entry_dte": sp["dte"],
        "breakeven": sp.get("breakeven"),
        "status": "open",
    }


def _per_contract(value_per_share: float, contracts: int) -> float:
    return round(value_per_share * 100 * contracts, 2)


def evaluate(position: dict, snapshot: dict, exits_cfg: dict) -> dict:
    """Evaluate one open paper position against the exit rules.

    snapshot: {"date", "underlying", "mark", "dte"}. `mark` is today's
    mid of the position (short-put mid for a CSP; short-mid minus long-mid
    for a spread); None when today's chain doesn't quote the leg(s).
    Returns a status dict with action "hold" or "close" (+ reason, pnl).
    """
    profit_frac = exits_cfg.get("profit_target_pct", 0.50)
    manage_dte = exits_cfg.get("manage_dte", 21)
    loss_mult = exits_cfg.get("loss_manage_multiple", 2.0)

    credit = position["entry_credit"]
    contracts = position.get("contracts", 1)
    dte = snapshot["dte"]
    underlying = snapshot["underlying"]
    mark = snapshot["mark"]

    status = {
        "key": position["key"],
        "date": snapshot["date"],
        "underlying": underlying,
        "mark": mark,
        "dte": dte,
        "days_held": _days_between(position["opened"], snapshot["date"]),
        "action": "hold",
        "reason": None,
        "unrealized": None,
        "realized": None,
    }

    if mark is None:  # chain doesn't quote the leg(s) today — hold, honestly
        status["reason"] = MARK_STALE
        return status

    if dte is None:  # can't compute days-to-expiry — hold, honestly
        status["reason"] = MARK_STALE
        return status

    profit_trigger = credit * (1 - profit_frac)
    # loss reaches loss_mult x premium <-> mark is premium + loss_mult x premium
    loss_trigger = credit * (1 + loss_mult)

    if dte <= 0:  # expiry: settle at intrinsic
        status["action"] = "close"
        if position["kind"] == "csp":
            itm = underlying < position["strike"]
            intrinsic = max(position["strike"] - underlying, 0.0)
            status["reason"] = "expiry"
            status["assigned"] = itm
        else:  # spread
            ss, ls = position["short_strike"], position["long_strike"]
            if underlying >= ss:
                intrinsic = 0.0                       # full profit
            elif underlying <= ls:
                intrinsic = position["max_loss"] + credit  # worst case
            else:  # short leg in the money, long leg out
                intrinsic = ss - underlying
            status["reason"] = "expiry"
            status["assigned"] = underlying < ss
        status["realized"] = _per_contract(credit - intrinsic, contracts)
        return status

    if mark <= profit_trigger:
        status["action"] = "close"
        status["reason"] = "profit"
        status["realized"] = _per_contract(credit - mark, contracts)
        return status

    if dte <= manage_dte:
        status["action"] = "close"
        status["reason"] = "time"
        status["realized"] = _per_contract(credit - mark, contracts)
        return status

    if mark >= loss_trigger:
        status["action"] = "close"
        status["reason"] = "loss"
        status["realized"] = _per_contract(credit - mark, contracts)
        return status

    status["unrealized"] = _per_contract(credit - mark, contracts)
    return status


def _days_between(opened: str, today: str) -> int:
    try:
        return (date.fromisoformat(today)
                - date.fromisoformat(opened)).days
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Chain indexing: what the caller (screener.py) feeds into the mark/update
# step each run. Kept here so the indexing logic is tested with the rest.
# ---------------------------------------------------------------------------


def chain_index(puts) -> dict:
    """Build a {strike: mid} index over a put chain for position marking.

    Only rows with a finite strike and a usable quote are indexed. Strikes
    are rounded to 2dp — the same rounding signal/spread dicts carry.
    """
    index = {}
    for _, row in puts.iterrows():
        try:
            strike = round(float(row["strike"]), 2)
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(strike) and strike > 0):
            continue
        q = quote_stats(row)
        if q is None or not math.isfinite(q["mid"]):
            continue
        index[strike] = round(q["mid"], 2)
    return index


def build_snapshot(position: dict, market: dict, as_of: str) -> dict | None:
    """Build an evaluation snapshot for one position from market data.

    market: {"marks": {strike: mid}, "underlying": float}.
    Returns None when the expiry can't be parsed (stale hold downstream).
    """
    marks = market.get("marks") or {}
    if position["kind"] == "csp":
        mark = marks.get(round(float(position["strike"]), 2))
    else:
        short = marks.get(round(float(position["short_strike"]), 2))
        long = marks.get(round(float(position["long_strike"]), 2))
        mark = round(short - long, 2) if short is not None and long is not None else None
    try:
        dte = (date.fromisoformat(position["expiry"])
               - date.fromisoformat(as_of)).days
    except (ValueError, TypeError):
        return None
    return {"date": as_of, "underlying": market.get("underlying"),
            "mark": mark, "dte": dte}


# ---------------------------------------------------------------------------
# Local state: open + closed paper positions (git-ignored; research only).
# ---------------------------------------------------------------------------

def store_path(state_dir: str) -> str:
    return os.path.join(state_dir, STATE_FILENAME)


def load_store(state_dir: str) -> dict:
    """Load the position store; tolerate missing or corrupt files."""
    path = store_path(state_dir)
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("open", [])
            data.setdefault("closed", [])
            return data
    except (OSError, ValueError):
        pass
    return {"open": [], "closed": []}


def save_store(state_dir: str, store: dict) -> None:
    os.makedirs(state_dir, exist_ok=True)
    with open(store_path(state_dir), "w") as f:
        json.dump(store, f, indent=2)


def known_keys(store: dict) -> set:
    """Keys of every position ever tracked (open or closed) — prevents
    silently re-opening a closed position when the same alert fires later."""
    keys = set()
    for p in store.get("open", []) + store.get("closed", []):
        keys.add(p["key"])
    return keys


def open_positions(candidates: list, store: dict, max_open: int) -> list:
    """Open paper positions for candidate dicts (signal/spread dicts).

    Skips any candidate whose key is already tracked (open or closed) and
    stops at `max_open` simultaneous open positions. Returns the newly
    opened positions. The caller saves the store.
    """
    seen = known_keys(store)
    new = []
    for cand in candidates:
        if "short_strike" in cand:  # bull put spread dict
            pos = open_from_spread(cand)
        else:                       # CSP signal dict
            pos = open_from_signal(cand)
        pos["contracts"] = cand.get("contracts", 1)
        if pos["key"] in seen:
            continue
        if len(store["open"]) + len(new) >= max_open:
            continue
        seen.add(pos["key"])
        new.append(pos)
    store["open"].extend(new)
    return new


def update_positions(store: dict, marks: dict, exits_cfg: dict,
                     as_of: str | None = None) -> dict:
    """Mark every open position against today's chain and apply exit rules.

    marks: {position_key: snapshot dict} — the caller builds snapshots from
    the day's chains; a position with no entry holds as `stale`.
    Returns {"updated": [...statuses...], "closed_today": [...positions...]}.
    """
    as_of = as_of or date.today().isoformat()
    updated, closed = [], []
    still_open = []
    for pos in store["open"]:
        snap = marks.get(pos["key"])
        if snap is None:
            snap = {"date": as_of, "underlying": None,
                    "mark": None, "dte": None}
        status = evaluate(pos, snap, exits_cfg)
        pos["last_mark"] = snap["mark"]
        pos["last_dte"] = snap["dte"]
        pos["last_seen"] = as_of
        pos["last_unrealized"] = status["unrealized"]  # None when closed/stale
        pos["days_held"] = status["days_held"]
        updated.append(status)
        if status["action"] == "close":
            pos["status"] = "closed"
            pos["closed_date"] = as_of
            pos["exit_reason"] = status["reason"]
            pos["realized"] = status["realized"]
            if status.get("assigned"):
                pos["assigned"] = True
            closed.append(pos)
        else:
            still_open.append(pos)
    store["open"] = still_open
    store["closed"].extend(closed)
    return {"updated": updated, "closed_today": closed}
