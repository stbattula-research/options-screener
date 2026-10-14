"""True IV rank from an observed-IV history.

Free sources (yfinance) publish today's option-chain IVs but no historical
IV series, and no free source publishes one either. So this module builds
the history itself: every screener run snapshots the IV of the day's
~30-delta short-put candidate per ticker into an append-only JSONL file
(<history_dir>/<TICKER>.jsonl). Once enough daily snapshots have
accumulated, "IV rank" becomes a real measurement: where today's IV sits
inside the observed IV range for the same DTE band.

Honest cold start: until `min_observations` snapshots exist for a ticker,
the rank returns None and the report keeps showing the IV-vs-HV proxy,
labeled as a proxy. The observation count is always reported next to any
true rank so a 10-day rank is never mistaken for a 1-year one.
"""

import json
import os
from datetime import date


def history_path(history_dir: str, ticker: str) -> str:
    """Path of a ticker's observation file."""
    safe = "".join(c for c in ticker.upper() if c.isalnum())
    return os.path.join(history_dir, f"{safe}.jsonl")


def record_observation(history_dir: str, ticker: str, dte: int, iv: float,
                       observed: str | None = None) -> str:
    """Append (or replace) one (date, dte, iv) observation for a ticker.

    Idempotent per date: re-running the screener the same day replaces the
    earlier snapshot instead of double-counting it.
    """
    observed = observed or date.today().isoformat()
    path = history_path(history_dir, ticker)
    os.makedirs(history_dir, exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate hand-edits; don't let one bad line
                if row.get("date") != observed:  # keep everything but today
                    rows.append(row)
    rows.append({"date": observed, "dte": int(dte), "iv": float(iv)})
    rows.sort(key=lambda r: r["date"])
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def load_observations(history_dir: str, ticker: str,
                      dte_min: int | None = None,
                      dte_max: int | None = None) -> list:
    """Return [(date, dte, iv), ...] sorted by date, optionally DTE-filtered.

    The DTE filter keeps comparisons apples-to-apples: IVs from a 60-DTE
    snapshot belong to a different part of the term structure and must not
    mix into a 30-45 DTE rank. Corrupt lines are skipped, not fatal.
    """
    path = history_path(history_dir, ticker)
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                obs = (str(row["date"]), int(row["dte"]), float(row["iv"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if dte_min is not None and dte_max is not None:
                if not (dte_min <= obs[1] <= dte_max):
                    continue
            rows.append(obs)
    rows.sort(key=lambda r: r[0])
    return rows


def iv_rank(iv_now: float, observations: list,
            min_observations: int = 10) -> tuple[float | None, int]:
    """True IV rank: (iv_now - min) / (max - min) over the observed history.

    Returns (rank, n_observations). Rank is None when the history is too
    short (honest cold start) or flat (no range to rank against).
    """
    n = len(observations)
    if n < min_observations:
        return None, n
    ivs = [o[2] for o in observations]
    lo, hi = min(ivs), max(ivs)
    if hi <= lo:
        return None, n
    return max(0.0, min(1.0, (iv_now - lo) / (hi - lo))), n
