"""Dividend / ex-date awareness for premium-selling signals.

Why this exists: when a stock goes ex-dividend inside the option window,
the underlying drops by roughly the dividend amount on the ex-date. For a
short cash-secured put that means (a) a small mechanical drag on the
underlying toward the short strike and (b) a higher chance of early
assignment if the put is in the money near the ex-date (dividend capture).
The effect is real but modest for a ~30-delta OTM put — nothing like an
earnings gap — so the default here is the opposite of the v9 earnings
gate: FLAG, don't exclude. The signal prints with its ex-date in the
report, and exclusion is opt-in (`dividends.exclude: true`).

Design choices, stated plainly:

- The next ex-date comes from yfinance's calendar data, which is
  unofficial and sometimes missing or stale. A missing date is reported
  as "unknown" — it never blocks the run and never silently passes as
  "no dividend". Only a KNOWN date inside the window is flagged or
  excluded.
- The window is inclusive on both ends, like the earnings gate: an
  ex-date on the run date or on the expiry date counts as inside.
- The dividend amount is not shown: yfinance's calendar does not publish
  it reliably, and a stale amount would be worse than none.
"""

from datetime import date, datetime, timezone

# Epoch seconds below this are not plausible ex-dates (before 2001-09-09);
# anything smaller is a stray integer, not a timestamp.
_MIN_EPOCH = 1_000_000_000


def parse_dividend_date(value):
    """Normalize one yfinance-style ex-dividend date to a `date`, or None.

    Accepts `date`, `datetime`, pandas Timestamp-like objects (anything
    with a `.date()` method), "YYYY-MM-DD" strings (a trailing time part
    is tolerated), and Unix-epoch seconds as int/float (the shape
    `Ticker.info["exDividendDate"]` returns). Anything else — None,
    garbage, a list, a too-small integer — returns None instead of
    raising, because an unparseable calendar entry is an "unknown", not
    a crash.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date"):
        try:
            parsed = value.date()
        except Exception:
            return None
        return parsed if isinstance(parsed, date) else None
    if isinstance(value, (int, float)):
        if value < _MIN_EPOCH:
            return None
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).date()
        except (ValueError, OverflowError, OSError):
            return None
    if isinstance(value, str):
        text = value.strip()[:10]
        try:
            y, m, d = (int(p) for p in text.split("-"))
            return date(y, m, d)
        except (ValueError, AttributeError):
            return None
    return None


def dividend_status(next_ex_date, as_of: date, expiry: date):
    """Classify the next ex-dividend date against the option window.

    Returns (status, ex_date) where status is one of:
    - "inside":  as_of <= ex-date <= expiry (the underlying drops by
      roughly the dividend while the short put is live)
    - "outside": ex-date is known but falls outside the window
    - "unknown": no ex-date available
    """
    if next_ex_date is None:
        return "unknown", None
    if as_of <= next_ex_date <= expiry:
        return "inside", next_ex_date
    return "outside", next_ex_date


def apply_dividend_gate(signal: dict, next_ex_date, as_of: date,
                        cfg: dict) -> dict:
    """Attach dividend info to a signal and decide whether it survives.

    Returns a dict with:
    - "status": "inside" | "outside" | "unknown"
    - "date": the ex-dividend date as an ISO string, or None
    - "excluded": True only when the date is inside the window AND
      `dividends.exclude` is enabled in config (default: False — flag,
      don't exclude, unlike the earnings gate)
    - "reason": human-readable skip reason when excluded, else None

    The signal dict itself is updated with a "dividends" entry so the
    report can render it.
    """
    dcfg = cfg.get("dividends", {}) or {}
    enabled = dcfg.get("enabled", True)
    exclude = dcfg.get("exclude", False)
    if not enabled:
        signal["dividends"] = {"status": "disabled", "date": None}
        return {"status": "disabled", "date": None, "excluded": False,
                "reason": None}
    try:
        y, m, d = (int(p) for p in signal["expiry"].split("-"))
        expiry = date(y, m, d)
    except (KeyError, ValueError, AttributeError):
        signal["dividends"] = {"status": "unknown", "date": None}
        return {"status": "unknown", "date": None, "excluded": False,
                "reason": None}
    status, ex_date = dividend_status(next_ex_date, as_of, expiry)
    signal["dividends"] = {
        "status": status,
        "date": ex_date.isoformat() if ex_date else None,
    }
    if status == "inside" and exclude:
        return {
            "status": status,
            "date": ex_date.isoformat(),
            "excluded": True,
            "reason": (f"ex-dividend {ex_date.isoformat()} falls inside the "
                       f"option window ({as_of.isoformat()} to "
                       f"{signal['expiry']}) — skipped per dividends.exclude"),
        }
    return {"status": status,
            "date": ex_date.isoformat() if ex_date else None,
            "excluded": False, "reason": None}
