"""Earnings-event awareness for premium-selling signals.

The classic rule this implements: do not sell a put whose life spans an
earnings announcement. An earnings jump can gap the underlying through the
short strike overnight, and the extra premium collected rarely pays for
that gap risk.

Design choices, stated plainly:

- The earnings date comes from yfinance's calendar data, which is
  unofficial and sometimes missing or stale. A missing date is reported
  as "unknown" — it never blocks the run and never silently passes as
  "no earnings". Only a KNOWN date inside the window excludes a signal.
- The window is inclusive on both ends: an earnings date on the run date
  or on the expiry date counts as inside. Earnings announced after the
  close on expiry day still move the underlying before expiry.
- Exclusion is config-gated (`earnings.exclude`, default true). With
  `exclude: false` the signal still prints, flagged with its earnings
  date, so the user decides.
"""

from datetime import date, datetime


def parse_earnings_date(value):
    """Normalize one yfinance-style earnings date to a `date`, or None.

    Accepts `date`, `datetime`, pandas Timestamp-like objects (anything
    with a `.date()` method), and "YYYY-MM-DD" strings (a trailing time
    part is tolerated). Anything else — None, garbage, a list — returns
    None instead of raising, because an unparseable calendar entry is an
    "unknown", not a crash.
    """
    if value is None:
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
    if isinstance(value, str):
        text = value.strip()[:10]
        try:
            y, m, d = (int(p) for p in text.split("-"))
            return date(y, m, d)
        except (ValueError, AttributeError):
            return None
    return None


def earnings_status(next_earnings, as_of: date, expiry: date):
    """Classify the next earnings date against the option window.

    Returns (status, earnings_date) where status is one of:
    - "inside":  as_of <= earnings <= expiry (the announcement lands
      while the short put is live)
    - "outside": earnings is known but falls outside the window
    - "unknown": no earnings date available
    """
    if next_earnings is None:
        return "unknown", None
    if as_of <= next_earnings <= expiry:
        return "inside", next_earnings
    return "outside", next_earnings


def apply_earnings_gate(signal: dict, next_earnings, as_of: date,
                        cfg: dict) -> dict:
    """Attach earnings info to a signal and decide whether it survives.

    Returns a dict with:
    - "status": "inside" | "outside" | "unknown"
    - "date": the earnings date as an ISO string, or None
    - "excluded": True only when the date is inside the window AND
      `earnings.exclude` is enabled in config
    - "reason": human-readable skip reason when excluded, else None

    The signal dict itself is updated with an "earnings" entry so the
    report can render it.
    """
    ecfg = cfg.get("earnings", {}) or {}
    enabled = ecfg.get("enabled", True)
    exclude = ecfg.get("exclude", True)
    if not enabled:
        signal["earnings"] = {"status": "disabled", "date": None}
        return {"status": "disabled", "date": None, "excluded": False,
                "reason": None}
    try:
        y, m, d = (int(p) for p in signal["expiry"].split("-"))
        expiry = date(y, m, d)
    except (KeyError, ValueError, AttributeError):
        signal["earnings"] = {"status": "unknown", "date": None}
        return {"status": "unknown", "date": None, "excluded": False,
                "reason": None}
    status, earn_date = earnings_status(next_earnings, as_of, expiry)
    signal["earnings"] = {
        "status": status,
        "date": earn_date.isoformat() if earn_date else None,
    }
    if status == "inside" and exclude:
        return {
            "status": status,
            "date": earn_date.isoformat(),
            "excluded": True,
            "reason": (f"earnings {earn_date.isoformat()} falls inside the "
                       f"option window ({as_of.isoformat()} to "
                       f"{signal['expiry']}) — skipped per earnings.exclude"),
        }
    return {"status": status,
            "date": earn_date.isoformat() if earn_date else None,
            "excluded": False, "reason": None}
