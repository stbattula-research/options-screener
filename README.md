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
python backtest.py --sweep --ticker SPY --years 5  # exit-rule sensitivity sweep
# writes reports/backtest_YYYY-MM-DD.md and .json (or backtest-sweep-*)
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

## Paper positions: signal → alert → exit tracking

Signals and alerts answer "what looks good today"; paper tracking answers
"what happened to the ones we tracked". It is **opt-in** (`paper: enabled:
true`) and **paper only** — nothing here ever places an order or holds
broker credentials.

```bash
python screener.py --alerts --paper-track
```

What a run does:

1. **Opens** one paper position per *newly fired alert* (never from plain
   signals; at most `max_open` positions at once, `contracts` per position
   for research sizing).
2. **Marks** every open position against that day's option chain
   (short-put mid, or short-minus-long for spreads) and records days held.
3. **Exit-checks** against the same mechanical rules as the exit plans —
   close at 50% of max profit, at 21 DTE, or when the loss reaches 2x the
   premium received (settled at intrinsic on expiry, with assignment
   flagged). A missing quote or an unparseable expiry holds the position
   as *stale* rather than guessing.
4. Prints a **Paper positions** section in the daily report with today's
   opens/closes plus every open position's unrealized P&L, DTE and days
   held.

Position keys reuse the alert dedupe keys (`ticker|expiry|strike`), so an
alert can only ever open one position and a closed position is never
silently re-opened. State lives in `positions/positions.json` (local,
git-ignored). Closed P&L is summed over contracts × 100 shares.

## Exit-rule sensitivity sweep

The mechanical exit rules in `config.yaml` (50% of max profit, 21-DTE
management, 2×-premium loss management) are config values, not validated
truths. The sweep re-runs the same simulation over a grid of those rules
and ranks the outcomes by model total P&L:

```bash
python backtest.py --sweep --ticker SPY --years 5
```

The report marks the config default (★) and shows each combo's Δ vs the
default. The question it answers: is the default combo near the top with
smooth neighbors (robust), or does performance collapse one notch away
(fragile/overfit)? The grid lives in `config.yaml` under `backtest_sweep:`
— deliberately coarse, because this is a robustness check, not an
optimizer. Everything else (entries, strikes, IV proxy) is identical across
combos.

## Skew-aware spread backtest (v7)

The v5 sweep deferred a question: under flat Black-Scholes, no width in
the live spread range (2–6% of the short strike) passes the live
screener's "collect at least 1/3 of the width" entry rule. The hope was
that downside skew would rescue it. It does not — and v7 shows why:

```bash
python backtest.py --spreads --ticker SPY --years 2
```

Each skew value re-runs the same spread simulation (same entry/strike/
exit rules, only leg pricing changes, with
`iv(K) = atm_iv x (1 + skew x max(0, log(S/K)))`). Skew raises the long
leg's IV more than the short leg's, so the long leg costs more and net
credit *falls* as skew rises. The earlier note's direction was backwards;
v7 corrects it with a test that asserts the direction.

Live smoke (SPY, 2y, 2% width): avg credit/width falls 0.244 at skew 0
to 0.200 at skew 4, and the credit-rule pass rate falls 5% to 0% — while
model total P&L flips from +$1,615 to −$1,299. What moves the ratio
toward the 1/3 rule is narrower widths (around 1% of the short strike)
and higher vol, not steeper skew. That is a screener-design finding, not
a recommendation: narrowing the live width band would also shrink max
loss per trade and change which wings qualify, so it needs weighing
before any change. The grid lives in `config.yaml` under
`spread_backtest:`.

## Spread-width sweep (v8)

v7 fixed the width at 2% and swept skew; its note claimed narrower
widths move the credit/width ratio toward the live 1/3 rule, but that
was never measured across widths. This sweep measures it:

```bash
python backtest.py --spread-widths --ticker SPY --years 2
```

Skew is held fixed at 2.0 (about the observed SPY put skew) while the
wing width runs 1% through the live 2–6% band. Two columns pull
opposite ways and are reported together: the credit fraction (which
rises as the wing narrows) and total P&L per contract (which falls,
because a narrower wing collects less absolute credit). The report
names the narrowest width whose *average* credit/width reaches the
1/3 rule, and the width with the best total P&L — the tradeoff to
weigh before touching the live band. It does not change the live
band: that stays the owner's decision.

## Earnings-aware filter (v9)

A signal whose option window contains a known earnings announcement is
skipped: an earnings gap can run through the short strike overnight, and
the extra premium rarely pays for that jump. The filter is on by default
(`earnings:` in `config.yaml`); set `exclude: false` to keep the signal
and flag it instead, or `enabled: false` to turn the feature off.

Honesty built in: the earnings date comes from yfinance's calendar data,
which is unofficial and sometimes missing (ETFs have no earnings at all).
A missing date renders as `unknown` in the report's Earnings column — a
gap in the data, never evidence that no earnings are coming. Only a
*known* date inside the window excludes a signal, and the skip reason
names the date. A spread shares its short leg's expiry, so an excluded
signal never produces a spread either.

## Earnings-aware backtest (v10)

The plain backtest traded straight through earnings windows, but the
live screener (v9) would never take those trades — so its P&L described
a strategy the screener doesn't run. `python backtest.py
--earnings-calendar calendar.json` closes the gap: it re-runs the same
simulation with the same gate (entry date through entry date + DTE
calendar days, inclusive) and reports plain vs filtered side by side,
with every skipped entry listed.

Honesty built in: no free source publishes historical earnings
calendars, so the dates are caller-supplied (`{ticker: ["YYYY-MM-DD",
...]}`; garbage dates are dropped, a missing file fails loudly). A
ticker with no calendar dates runs unfiltered and is flagged as such in
the report — an empty calendar is a data gap, never evidence of no
earnings. With `earnings.exclude: false` or `enabled: false` in config,
the filtered run equals the plain run and the report says the gate was
inactive, matching the live screener's flag-only behavior.

## Earnings-aware spread backtest (v11)

The v9 live gate already covers spreads — a spread shares the short
leg's expiry, so an excluded CSP signal never produces a spread — but
the v7/v8 spread backtests kept trading straight through earnings
windows, so their P&L described spread trades the screener would never
open. `python backtest.py --spreads --earnings-calendar calendar.json`
closes that gap: it runs the spread simulation twice per skew (plain,
then with the v9 gate applied) and reports plain vs filtered side by
side, with every skipped spread entry listed per skew.

The same honesty rules apply as v10: caller-supplied calendars, empty
calendar flagged as a data gap, gate-off runs equal the plain run.
One extra caveat, carried from v7: per-skew entry timing can shift
after a skip (exit timing differs by skew), so the clean comparison is
plain vs filtered *within* one skew, not across skews. Live smoke
(AAPL 1y, one calendar date 2026-05-05): 5 entries skipped at every
skew — the same 35-day-window / 5-day-cadence effect v10 measured —
and filtered P&L ran below plain on that period at all four skews.
Descriptive, not a reason to change the live gate.

## Dividend / ex-date awareness (v12)

A known ex-dividend date inside the option window is now shown in a
Dividend column on the signals table — and, unlike the v9 earnings
gate, the signal still prints. The reason is the size of the effect:
on the ex-date the stock drops by roughly the dividend, which nudges
the underlying toward the short strike and raises early-assignment
odds if the put is in the money near the date — a real but modest
effect on a ~30-delta OTM put, not an overnight gap. Exclusion is
opt-in (`dividends.exclude: true` in config.yaml); the default is flag,
don't exclude. Spreads inherit the flag via the shared expiry.

The same honesty rules apply as v9: the ex-date comes from yfinance's
calendar (with `Ticker.info["exDividendDate"]` as a fallback), a
missing date renders as `unknown` — a gap in the data, never evidence
of no dividend — and the dividend amount is not shown, because
yfinance does not publish it reliably. A stale amount would be worse
than none.

## Earnings-aware paper-position marking (v13)

The v9 gate only checks earnings at signal time — but a paper position
lives 30–45 days, and an earnings date can be announced *after* it
opens, or was simply unknown when it opened. Every `--paper-track`
mark now re-checks each open position's ticker against its *remaining*
window (today to expiry) and records the exposure on the position:
`inside` with the date (shown ⚠ in the Paper positions tables), `outside`,
or `unknown` when the calendar has nothing. Positions also carry what
was known at entry, copied from the signal's v9 gate result.

The flag is informational only, stated in the report itself: a position
never exits early on earnings — exits stay mechanical (50% profit /
21 DTE / 2x-premium loss). The same honesty rules apply as v9: an
`unknown` calendar is a data gap, never evidence of no earnings, and
the check is config-gated (`paper.earnings_check`, default on).

## Alerting on newly-exposed paper positions (v14)

v13 only *flagged* exposure in the report — a position that became
earnings-exposed after opening sat there silently until someone read
the daily report. v14 detects the flip at the mark (from outside or
unknown to inside, or inside to a *different* announced date) and
alerts on it through the same email/webhook channels as signal alerts,
with the same rules: credentials env-only, disclaimer in every alert,
dedupe so a re-run never re-fires, once per position+earnings-date, and
a failing channel is reported, never fatal. A position that closes at
the same mark is never alerted on — there is nothing left to warn
about.

Two deliberate limits: the alert is informational only (it never exits
a position — exits stay mechanical), and it only sends on runs invoked
with `--alerts` (or `--alerts-dry-run`). A `--paper-track`-only run
records the exposure in the report's new "Newly earnings-exposed
today" section but sends nothing, so a quiet daily mark never surprises
anyone with mail. Config-gated with `paper.exposure_alerts`
(default on).

## Multi-ticker regime comparison (v15)

v6 split one ticker at a time, and on SPY the high-IV regime won more
often and made more, at the cost of a bigger drawdown. v15 asks whether
that pattern generalizes: `python backtest.py --regime-compare --all
--years 2` runs the same per-ticker median split across the watchlist
and compares the *gaps* (high minus low — P&L, win rate, max drawdown)
ticker by ticker, ranked by P&L gap, with a cross-ticker summary
(how many tickers the high-IV regime won on, median gaps).

The honest limit is the one v6 already named, now load-bearing: each
ticker's regimes are defined by *that* ticker's median entry IV, and
medians differ wildly per ticker (SPY 0.15, NVDA 0.42, AMD 0.58 on the
2y smoke run), so absolute IV levels are never compared across
tickers — only the within-ticker gaps are, and the median-IV column is
printed per ticker so the difference is visible. A row is flagged
"thin" when either regime has fewer than 10 trades, and thin rows
never count toward the summary. Descriptive only, like v6: the median
is known only after the whole period, so nothing here is a live entry
filter.

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
- v4: ✅ paper position tracking (shipped 2026-10-18: positions open from
  newly fired alerts, marked against each day's chain, exit-checked on the
  50%-profit / 21-DTE / 2x-premium rules; stale holds when quotes are
  missing; paper only, no orders, no broker credentials).
- v5: ✅ exit-rule sensitivity sweep (shipped 2026-10-21: `python backtest.py
  --sweep` re-runs the simulation over a grid of profit-target / loss-multiple
  / manage-DTE rules and ranks combos by model total P&L, marking the config
  default — a robustness check on the mechanical exit rules, not an optimizer).
- v7: ✅ skew-aware spread backtest (shipped 2026-10-26: `python backtest.py
  --spreads` prices both spread legs at their own skewed IVs and sweeps
  skew 0→4, closing the v5 deferred question — skew *lowers* net credit
  (long leg's IV rises faster), so it cannot rescue the 1/3-width credit
  rule; SPY 2y smoke: credit/width 0.244→0.200, pass rate 5%→0% as skew
  rises. Narrower widths, not steeper skew, move the ratio toward the rule).
- v8: ✅ spread-width sweep (shipped 2026-10-28: `python backtest.py
  --spread-widths` sweeps wing widths 1–6% at fixed skew 2.0 and reports
  where the 1/3-width credit rule starts passing on average and what
  passing costs in total P&L — measuring v7's narrower-width claim
  rather than changing the live 2–6% band).
- v9: ✅ earnings-aware filter (shipped 2026-10-31: signals whose option
  window contains a known earnings date are skipped, with the date named
  in the skip reason; unknown calendar dates render as `unknown`, never
  as safe; spreads inherit the gate via the shared expiry).
- v10: ✅ earnings-aware backtest (shipped 2026-11-02: `python backtest.py
  --earnings-calendar calendar.json` re-runs the simulation with the v9
  gate applied and reports plain vs filtered P&L side by side with every
  skipped entry listed; calendars are caller-supplied JSON because no
  free source publishes historical earnings dates, and a ticker with no
  calendar dates is flagged as unfiltered, not assumed earnings-free).
- v11: ✅ earnings-aware spread backtest (shipped 2026-11-04: `python
  backtest.py --spreads --earnings-calendar calendar.json` runs the
  spread simulation twice per skew — plain, then with the v9 gate
  applied — and reports plain vs filtered side by side with every
  skipped spread entry listed per skew; closes the gap the v7/v8 spread
  backtests left, since the live v9 gate already covers spreads).
- v12: ✅ dividend / ex-date awareness (shipped 2026-11-06: a known
  ex-dividend date inside the option window is flagged in a Dividend
  column — ⚠ when inside — and the signal still prints by default,
  because the effect on a ~30-delta OTM put is a nudge, not an earnings
  gap; exclusion is opt-in via `dividends.exclude`; unknown calendar
  dates render as `unknown`, and no dividend amount is shown since
  yfinance does not publish it reliably).
- v13: ✅ earnings-aware paper-position marking (shipped 2026-11-08: every
  `--paper-track` mark re-checks each open position's next earnings date
  against its remaining window and flags it in the Paper positions
  tables — informational only, exits stay mechanical; a position can
  become exposed after opening when the date is announced later or was
  unknown at entry).
- v14: ✅ alerting on newly-exposed paper positions (shipped 2026-11-10:
  when a still-open position becomes earnings-exposed at a mark, the
  report names it in a "Newly earnings-exposed today" section and — on
  runs invoked with `--alerts` — the same email/webhook channels send
  one alert per position+earnings-date, deduped so re-runs never re-fire;
  a `--paper-track`-only run records the exposure but sends nothing.
  Informational only, exits stay mechanical).
- v15: ✅ multi-ticker regime comparison (shipped 2026-11-12: `python
  backtest.py --regime-compare` runs the v6 median split per ticker and
  compares the high-minus-low gaps across the watchlist, ranked by P&L
  gap with a cross-ticker summary; medians differ per ticker so only
  the gaps are compared, never absolute IV levels; thin rows (<10
  trades per regime) never count toward the summary. On a 2y, 15-ticker
  smoke run the high-IV regime made more on 13 of 15 tickers (median
  gap +$2,547.74) and won more often on 10 of 15 — but the median
  max-drawdown gap was only +$7.66, so across the watchlist the extra
  premium did *not* come with systematically bigger drawdowns the way
  SPY alone suggested in v6. MSFT and NVDA were the exceptions:
  high-IV entries lost more and won less often there. AMD's +$18,280
  gap is an outlier row, not the story — the median gap is).
- v6: ✅ IV-regime split (shipped 2026-10-24: `python backtest.py --regimes`
  splits each ticker's backtest trades into low/high entry-IV regimes at that
  ticker's median entry IV and reports win rate, P&L, and drawdown per regime
  — descriptive only, since the median is only known after the whole period;
  it is not a live entry filter. On a 2y SPY smoke run the high-IV regime
  won 86% vs 64% low-IV, but with the larger max drawdown — extra premium
  paid for extra risk on that period).
- Later, only with explicit approval at each stage: paper-trading broker
  integration. **Live auto-trading is never in scope without the owner's
  explicit go-ahead.**

## License

MIT — see [LICENSE](LICENSE).
