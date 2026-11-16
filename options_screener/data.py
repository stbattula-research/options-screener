"""Market-data access. yfinance only, with honest error reporting.

yfinance is an unofficial Yahoo Finance API: quotes are delayed (~15 min)
and the endpoint can break without notice. Every fetch failure is surfaced
to the caller instead of silently returning stale data.
"""

from datetime import date, timedelta

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


def get_closes_with_dates(ticker: str, period: str = "1y"):
    """Daily closes paired with their trading-day dates.

    Returns (closes, dates) with one `date` per close, in order — the
    pairing the earnings-aware backtest needs to map each simulated
    entry day to a calendar window. Raises DataError on failure, like
    `get_closes`.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception as exc:  # network / API breakage
        raise DataError(f"{ticker}: price history fetch failed: {exc}")
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise DataError(f"{ticker}: no price history returned")
    closes = [float(c) for c in hist["Close"].tolist()]
    dates = [idx.date() for idx in hist.index.tolist()]
    if len(closes) < 2 or len(dates) != len(closes):
        raise DataError(f"{ticker}: price history too short")
    return closes, dates


def slice_window(closes, dates, years, offset_years, today):
    """Carve an earlier `years`-long window out of a longer closes series.

    `closes`/`dates` are the full series (oldest first), fetched over
    `years + offset_years` years. The window is the CALENDAR span
    (cutoff - years*365 days, cutoff], where cutoff is `offset_years`
    calendar years before `today` — so a 2-year window holds however
    many trading days the market actually produced in those two years
    (typically 500-506, not exactly 2 * 252: a fixed 252/year count
    over-counts short years and once left a real window one close
    short). The re-test window is the same calendar span as the
    original run, only earlier, so the two runs stay directly
    comparable. Returns (window_closes, start_date, end_date).
    Raises ValueError for years < 1 or offset_years < 0; raises
    DataError when the window holds fewer than years * 240 closes
    (a data gap, never a silently shorter window).
    """
    if years < 1:
        raise ValueError("years must be >= 1")
    if offset_years < 0:
        raise ValueError("offset_years must be >= 0")
    if len(closes) != len(dates):
        raise DataError("closes/dates length mismatch")
    cutoff = today - timedelta(days=365 * offset_years)
    start = cutoff - timedelta(days=365 * years)
    window = [c for c, d in zip(closes, dates) if start < d <= cutoff]
    window_dates = [d for d in dates if start < d <= cutoff]
    floor = years * 240  # real years hold ~250-253 trading days
    if len(window) < floor:
        raise DataError(
            f"only {len(window)} closes in the {years}-year window "
            f"ending {cutoff.isoformat()}, need at least {floor}")
    return window, window_dates[0], window_dates[-1]


def get_closes_window(ticker: str, years: int, offset_years: int):
    """Closes for a `years`-long window ending `offset_years` years ago.

    Fetches `years + offset_years` years of history and carves the
    window out with `slice_window`. Returns (closes, start_date,
    end_date). Raises DataError like get_closes.
    """
    closes, dates = get_closes_with_dates(
        ticker, period=f"{years + offset_years}y")
    return slice_window(closes, dates, years, offset_years, date.today())


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


def get_next_earnings(ticker: str):
    """Next earnings date for `ticker` as a `date`, or None if unknown.

    Unlike the other fetchers here, this NEVER raises: yfinance's
    calendar data is patchy (ETFs have no earnings; some tickers return
    nothing), and a missing date is an "unknown" to flag, not a reason to
    kill the whole run. Returns None on any fetch/parse failure.
    """
    from options_screener.earnings import parse_earnings_date

    try:
        cal = yf.Ticker(ticker).calendar
    except Exception:
        return None
    if isinstance(cal, dict):
        dates = cal.get("Earnings Date")
        if dates:
            first = dates[0] if isinstance(dates, (list, tuple)) else dates
            return parse_earnings_date(first)
        return None
    # Some yfinance versions return a DataFrame with an "Earnings Date"
    # column instead of a dict.
    try:
        col = cal["Earnings Date"]
        values = col.tolist() if hasattr(col, "tolist") else list(col)
        if values:
            return parse_earnings_date(values[0])
    except Exception:
        return None
    return None


def get_next_dividend(ticker: str):
    """Next ex-dividend date for `ticker` as a `date`, or None if unknown.

    Like `get_next_earnings`, this NEVER raises: yfinance's calendar data
    is patchy (ETFs and non-payers have no ex-date; some tickers return
    nothing), and a missing date is an "unknown" to flag, not a reason to
    kill the whole run. Returns None on any fetch/parse failure.

    Sources, in order: the calendar's "Ex-Dividend Date" (dict or
    DataFrame shape), then `Ticker.info["exDividendDate"]` (epoch seconds)
    as a fallback for tickers whose calendar omits it.
    """
    from options_screener.dividends import parse_dividend_date

    ticker_obj = yf.Ticker(ticker)
    try:
        cal = ticker_obj.calendar
    except Exception:
        cal = None
    if cal is not None:
        if isinstance(cal, dict):
            dates = cal.get("Ex-Dividend Date")
            if dates:
                first = dates[0] if isinstance(dates, (list, tuple)) else dates
                parsed = parse_dividend_date(first)
                if parsed is not None:
                    return parsed
        else:
            # Some yfinance versions return a DataFrame with an
            # "Ex-Dividend Date" column instead of a dict.
            try:
                col = cal["Ex-Dividend Date"]
                values = col.tolist() if hasattr(col, "tolist") else list(col)
                if values:
                    parsed = parse_dividend_date(values[0])
                    if parsed is not None:
                        return parsed
            except Exception:
                pass
    try:
        info = ticker_obj.info
        if isinstance(info, dict):
            return parse_dividend_date(info.get("exDividendDate"))
    except Exception:
        return None
    return None
