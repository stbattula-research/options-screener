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
6. Builds **bull put spreads** (v2): for each CSP candidate, pairs it with a
   cheaper same-expiry long put 2–6% of the short strike lower, passing the
   same liquidity filters and collecting ≥ 1/3 of the spread width in credit.
   Max loss is capped at (width − credit); spreads are ranked by return on
   risk. This is the defined-risk version of the same trade.
7. Attaches a mechanical **exit plan** to every signal and spread: close at
   50% of max profit, exit/roll at 21 DTE, manage at 2× premium/credit.

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

## Backtesting the strategy

`backtest.py` simulates the CSP strategy over historical daily closes and
writes a Markdown + JSON report (trade log, win rate, P&L per contract,
max drawdown, exit breakdown):

```bash
python backtest.py --ticker SPY --years 2
python backtest.py --all --years 5   # whole watchlist
# writes reports/backtest_YYYY-MM-DD.md and .json
```

How it works: every 5 trading days it sells a ~30-delta put (35 DTE by
default, `backtest.entry_dte` in `config.yaml`), prices premiums with
Black-Scholes against an IV proxy (trailing 20-day realized vol × 1.15),
and applies the same exit rules as the screener (50% of max profit,
21-DTE management, 2×-premium loss management).

> ⚠️ **Model-based, not historical-chain data.** No free source publishes
> historical option chains, so premiums are *simulated*, not observed. IV is
> held constant per trade, there are no commissions or slippage, and fills
> are at model mid. Treat results as illustrative research — a sanity check
> on the mechanics, never as expected returns.

## Configuration

Everything tunable lives in `config.yaml`: the watchlist, the DTE band,
delta target/tolerance, minimum premium, liquidity thresholds, the risk-free
rate assumption, and the exit rules. Edit it, re-run, done.

## Data limitations

- Quotes come from **yfinance**, an unofficial Yahoo Finance API: delayed
  ~15 minutes, and the endpoint can change or break without notice. The
  screener surfaces fetch failures instead of silently using stale data.
- **True IV rank, built not bought.** Free sources don't publish a historical
  IV series, so the screener builds its own: each run snapshots the day's
  ~30-delta short-put IV per ticker into `iv_history/<TICKER>.jsonl`
  (git-ignored). Once a ticker has 10 daily snapshots in the same DTE band
  (configurable), the report shows a true IV rank — today's IV inside the
  ticker's own observed range — with the snapshot count beside it, so a
  young rank is never mistaken for a 1-year one. Before that, the report
  shows the IV-vs-HV proxy and labels it `(proxy)`.
- Deltas are Black-Scholes estimates; your broker's greeks may differ
  slightly.

## Alerts (v3): email + webhook on new signals

Alerts are **opt-in** and fire only for *new* qualifying signals — a signal
re-runs never re-fires (deduped against `alert_state/alerts.json`, local and
git-ignored).

```bash
python screener.py --alerts            # send, per the alerts: config section
python screener.py --alerts-dry-run   # print what would fire, send nothing
```

Rules live in `config.yaml` under `alerts:` — fire when a cash-secured put's
annualized yield or true IV rank crosses a threshold, or a spread's return
on risk crosses one:

```yaml
alerts:
  enabled: true
  rules:
    min_annualized_yield: 0.12
    min_spread_roi: 0.30
    min_iv_rank_true: 0.75   # rich-IV regime
  email:   {enabled: true}   # SMTP config from env vars (see below)
  webhook: {enabled: true}   # URL from env var (Discord/Slack-style POST)
```

**Credentials are never in config.** Email needs `ALERT_EMAIL_TO`,
`ALERT_SMTP_HOST`, `ALERT_SMTP_USER`, `ALERT_SMTP_PASSWORD` (and optional
`ALERT_EMAIL_FROM`, `ALERT_SMTP_PORT`); the webhook needs
`ALERT_WEBHOOK_URL`. Every alert body carries the risk disclaimer. A failed
channel is reported and skipped — it never kills the run.

## Roadmap

- v2: ✅ bull-put-spread signals (shipped 2026-10-10); ✅ model-based
  backtesting (shipped 2026-10-12: Black-Scholes premium simulation with an
  IV proxy — not historical chains, which no free source publishes); ✅ true
  IV rank (shipped 2026-10-14: observed-IV history built from the tool's own
  daily chain snapshots, no paid feed; IV-vs-HV proxy stays as the labeled
  cold-start fallback).
- v3: ✅ alerting (shipped 2026-10-16: email + webhook on new qualifying
  signals, deduped via local state; credentials env-only; disclaimer in every
  alert).
- Later, only with explicit approval at each stage: paper-trading broker
  integration. **Live auto-trading is never in scope without the owner's
  explicit go-ahead.**

## License

MIT — see [LICENSE](LICENSE).
