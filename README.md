# options-screener

A daily signal screener for **weekly options premium selling** — research only.
Each run scans a watchlist of liquid, optionable stocks/ETFs, pulls put chains,
and emits concrete cash-secured-put candidates: **~30 delta, 30–45 DTE**,
passing liquidity filters, with mechanical exit rules. Output is a readable
Markdown report plus machine-readable JSON.

> ⚠️ **Risk disclaimer — read first.** This is an educational research tool,
> **not financial advice**. Options involve substantial risk of loss, including
> losing the entire premium and, for cash-secured puts, being assigned the
> underlying at the strike price. Paper-trade every signal before risking real
> capital. No returns are guaranteed. Nothing in this repo places trades,
> connects to a broker, or should be interpreted as a recommendation to buy or
> sell any security.

## What it does

For each ticker in the watchlist (`config.yaml`):

1. Fetches 1 year of price history and the put chain for the expiration
   closest to the middle of the 30–45 DTE band.
2. Estimates put deltas with Black-Scholes (fixed risk-free-rate assumption).
3. Computes an **IV-vs-HV gauge**: current contract IV positioned inside the
   ticker's 1-year realized-volatility range. This is a *proxy*, not a true IV
   rank (see [data limitations](#data-limitations)).
4. Applies **liquidity filters**: minimum open interest, minimum daily option
   volume, maximum bid-ask spread width.
5. Keeps strikes near **−0.30 delta** with premium ≥ 1% of strike, ranked by
   annualized yield.
6. Attaches a mechanical **exit plan**: close at 50% of max profit, exit/roll
   at 21 DTE, manage at 2× premium loss.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python screener.py
# writes reports/report_YYYY-MM-DD.md and .json
```

Run with a custom config or output dir:

```bash
python screener.py --config config.yaml --out-dir reports/
```

Run the tests (no network needed):

```bash
pytest -q
```

## Configuration

Everything tunable lives in `config.yaml`: the watchlist, the DTE band,
delta target/tolerance, minimum premium, liquidity thresholds, the risk-free
rate assumption, and the exit rules. Edit it, re-run, done.

## Data limitations

- Quotes come from **yfinance**, an unofficial Yahoo Finance API: delayed
  ~15 minutes, and the endpoint can change or break without notice. The
  screener surfaces fetch failures instead of silently using stale data.
- **No true IV rank in v1.** Free sources don't provide a historical IV
  series, so the report shows IV positioned against the 1-year realized-vol
  range and labels it honestly. A true IV rank (current IV vs 52-week IV
  range) is planned for v2 with a paid historical-IV feed.
- Deltas are Black-Scholes estimates; your broker's greeks may differ
  slightly.

## Roadmap

- v2: true IV rank via historical-IV feed, backtesting on historical chains,
  bull-put-spread signals.
- v3: alerting (email/webhook) on new signals.
- Later, only with explicit approval at each stage: paper-trading broker
  integration. **Live auto-trading is never in scope without the owner's
  explicit go-ahead.**

## License

MIT — see [LICENSE](LICENSE).
