"""Volatility and options-pricing math for the screener.

Everything here is computed from first principles (Black-Scholes) so the
screener does not depend on a broker's greeks feed.
"""

import math

SQRT_252 = math.sqrt(252)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_delta(price: float, strike: float, years_to_expiry: float,
                 risk_free_rate: float, sigma: float) -> float:
    """Black-Scholes delta of a European put. Returns a negative number."""
    if price <= 0 or strike <= 0 or years_to_expiry <= 0 or sigma <= 0:
        raise ValueError("price, strike, T and sigma must all be positive")
    d1 = (math.log(price / strike)
          + (risk_free_rate + 0.5 * sigma * sigma) * years_to_expiry) \
        / (sigma * math.sqrt(years_to_expiry))
    return norm_cdf(d1) - 1.0


def bs_put_value(price: float, strike: float, years_to_expiry: float,
                 risk_free_rate: float, sigma: float) -> float:
    """Black-Scholes value of a European put (per share).

    At expiry (years_to_expiry <= 0) returns intrinsic value.
    """
    if price <= 0 or strike <= 0 or sigma <= 0:
        raise ValueError("price, strike and sigma must all be positive")
    if years_to_expiry <= 0:
        return max(strike - price, 0.0)
    sqrt_t = math.sqrt(years_to_expiry)
    d1 = (math.log(price / strike)
          + (risk_free_rate + 0.5 * sigma * sigma) * years_to_expiry) \
        / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return (strike * math.exp(-risk_free_rate * years_to_expiry) * norm_cdf(-d2)
            - price * norm_cdf(-d1))


def log_returns(prices) -> list:
    """Simple list of log returns from a price series."""
    return [math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices)) if prices[i - 1] > 0]


def realized_vol(prices, window: int = 20) -> float | None:
    """Annualized realized volatility over the trailing `window` closes."""
    rets = log_returns(list(prices))
    if len(rets) < window:
        return None
    recent = rets[-window:]
    mean = sum(recent) / len(recent)
    var = sum((r - mean) ** 2 for r in recent) / (len(recent) - 1)
    return math.sqrt(var) * SQRT_252


def hv_range(prices, window: int = 20, lookback: int = 252):
    """Min/max of trailing-`window` realized vol over the past `lookback` days.

    Returns (hv_min, hv_max, hv_now). Returns None when history is too short.
    """
    prices = list(prices)
    if len(prices) < lookback + window:
        return None
    vols = []
    # rolling window ending at each day of the lookback period
    for end in range(window, lookback + window + 1):
        v = realized_vol(prices[:end], window)
        if v is not None:
            vols.append(v)
    if not vols:
        return None
    return min(vols), max(vols), vols[-1]


def iv_vs_hv_rank(iv: float, hv_min: float, hv_max: float) -> float | None:
    """Where current IV sits inside the 1-year realized-vol range, 0..1.

    This is an IV-vs-HV premium gauge, NOT a true IV rank: free data sources
    (yfinance) do not provide a historical IV series. v2 will use a paid
    historical-IV feed for a true IV rank. The distinction is documented in
    the README and the report.
    """
    if hv_max <= hv_min:
        return None
    return max(0.0, min(1.0, (iv - hv_min) / (hv_max - hv_min)))


def skewed_iv(price: float, strike: float, atm_iv: float, skew: float) -> float:
    """Strike-dependent IV under a one-parameter downside-skew model.

    Real equity option markets price downside puts richer than at-the-money
    puts: the further a put strike sits below spot, the higher its implied
    vol. This model captures that shape with a single parameter:

        iv(K) = atm_iv * (1 + skew * max(0, log(S / K)))

    So at-the-money (K == S) and above-spot strikes keep atm_iv, and a
    5%-OTM put (log(S/K) ~= 0.051) gets atm_iv * (1 + 0.051 * skew).
    With skew = 2 that is ~10% richer than ATM — in the ballpark of
    observed SPY put skew at ~30 DTE; skew = 0 recovers flat Black-Scholes.

    Honest limits: this is a shape, not a fitted smile — it has no
    term-structure, no upside wing, and no calibration to a real chain.
    It is used to ask "does skew change the answer?" (see the spread
    backtest), never to claim a market price. skew must be >= 0.
    """
    if price <= 0 or strike <= 0 or atm_iv <= 0:
        raise ValueError("price, strike and atm_iv must all be positive")
    if skew < 0:
        raise ValueError("skew must be >= 0")
    downside = max(0.0, math.log(price / strike))
    return atm_iv * (1.0 + skew * downside)


def bs_put_value_skewed(price: float, strike: float, years_to_expiry: float,
                         risk_free_rate: float, atm_iv: float,
                         skew: float) -> float:
    """Black-Scholes put value priced at the strike's skewed IV."""
    return bs_put_value(price, strike, years_to_expiry, risk_free_rate,
                        skewed_iv(price, strike, atm_iv, skew))


def annualized_yield(premium_pct: float, dte: int) -> float | None:
    """Simple annualized yield: (premium/strike) * (365/DTE)."""
    if dte <= 0:
        return None
    return premium_pct * (365.0 / dte)
