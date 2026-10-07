"""Market-data access. yfinance only, with honest error reporting.

yfinance is an unofficial Yahoo Finance API: quotes are delayed (~15 min)
and the endpoint can break without notice. Every fetch failure is surfaced
to the caller instead of silently returning stale data.
"""

from datetime import date

import yfinance as yf


class DataError(Exception):
    """Raised when market data cannot be fetched or parsed."""


def get_closes(ticker: str, period: str = "1y") -> list:
    """Daily closes for `ticker`; raises DataError on failure."""
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception as exc:  # network / API breakage
        raise DataError(f"{ticker}: price history fetch failed: {exc}")
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise DataError(f"{ticker}: no price history returned")
    closes = [float(c) for c in hist["Close"].tolist()]
    if len(closes) < 2:
        raise DataError(f"{ticker}: price history too short")
    return closes


def get_expirations(ticker: str) -> list:
    """Option expiration dates (YYYY-MM-DD strings); raises DataError."""
    try:
        exps = yf.Ticker(ticker).options
    except Exception as exc:
        raise DataError(f"{ticker}: expirations fetch failed: {exc}")
    if not exps:
        raise DataError(f"{ticker}: no option expirations listed")
    return list(exps)


def pick_expiry(expirations: list, dte_min: int, dte_max: int,
                today: date | None = None):
    """Pick the expiration whose DTE is closest to the middle of the band.

    Returns (expiry_str, dte). Raises DataError if nothing lands in the band.
    """
    today = today or date.today()
    mid = (dte_min + dte_max) / 2.0
    best = None
    for exp in expirations:
        try:
            y, m, d = (int(p) for p in exp.split("-"))
            dte = (date(y, m, d) - today).days
        except ValueError:
            continue
        if dte_min <= dte <= dte_max:
            score = abs(dte - mid)
            if best is None or score < best[0]:
                best = (score, exp, dte)
    if best is None:
        raise DataError(
            f"no expiration inside {dte_min}-{dte_max} DTE "
            f"(nearest: {expirations[:3]}...)")
    return best[1], best[2]


def get_put_chain(ticker: str, expiry: str):
    """Put chain DataFrame for one expiration; raises DataError."""
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
    except Exception as exc:
        raise DataError(f"{ticker} {expiry}: chain fetch failed: {exc}")
    puts = getattr(chain, "puts", None)
    if puts is None or puts.empty:
        raise DataError(f"{ticker} {expiry}: empty put chain")
    return puts
