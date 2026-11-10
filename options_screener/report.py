"""Report rendering: human-readable Markdown + machine-readable JSON."""

import json
from datetime import date

DISCLAIMER = (
    "Disclaimer: this report is for education and research only — not financial "
    "advice. Options involve substantial risk of loss, including the loss of the "
    "entire premium and, for cash-secured puts, assignment risk on the underlying. "
    "Paper-trade any signal before risking capital. No returns are guaranteed. "
    "Quotes are delayed and indicative; verify with your broker before acting."
)


def _iv_rank_cell(s: dict) -> str:
    """Render the IV-rank cell: true rank with observation count when the
    observed-IV history is deep enough, otherwise the labeled IV-vs-HV proxy."""
    true = s.get("iv_rank_true")
    if true is not None:
        return f"{true:.2f} (true, {s.get('iv_rank_obs', 0)}d)"
    proxy = s.get("iv_vs_hv_rank")
    if proxy is not None:
        return f"{proxy:.2f} (proxy)"
    return "n/a"


def _earnings_cell(s: dict) -> str:
    """Render the earnings cell: the date when known, 'unknown' when the
    calendar had nothing, '—' when the feature is disabled. A date flagged
    with ⚠ means earnings fall inside the window but exclusion is off —
    the signal prints, and the user decides."""
    earn = s.get("earnings") or {}
    status = earn.get("status")
    if status == "disabled":
        return "—"
    if status == "unknown" or not earn.get("date"):
        return "unknown"
    if status == "inside":
        return f"{earn['date']} ⚠"
    return earn["date"]


def _dividend_cell(s: dict) -> str:
    """Render the dividend cell: the ex-date when known, 'unknown' when the
    calendar had nothing, '—' when the feature is disabled. A date flagged
    with ⚠ means the ex-date falls inside the window but exclusion is off
    (the default) — the signal prints, and the user decides."""
    div = s.get("dividends") or {}
    status = div.get("status")
    if status == "disabled":
        return "—"
    if status == "unknown" or not div.get("date"):
        return "unknown"
    if status == "inside":
        return f"{div['date']} ⚠"
    return div["date"]


def build_report_payload(results: dict, cfg: dict, as_of: str | None = None) -> dict:
    """Assemble the JSON-serializable report payload."""
    spread_cfg = cfg.get("spread", {})
    return {
        "as_of": as_of or date.today().isoformat(),
        "strategy": ("cash-secured puts + bull put spreads "
                     "(~30 delta short leg, 30-45 DTE)"),
        "data_source": "yfinance (unofficial, delayed ~15 min)",
        "config": {
            "dte_band": [cfg["strategy"]["dte_min"], cfg["strategy"]["dte_max"]],
            "target_delta": cfg["strategy"]["target_delta"],
            "liquidity": cfg["liquidity"],
            "spread": {k: spread_cfg.get(k) for k in
                       ("enabled", "min_width_pct", "max_width_pct",
                        "min_credit_fraction")} if spread_cfg else None,
            "earnings": cfg.get("earnings", {}),
            "dividends": cfg.get("dividends", {}),
            "paper": cfg.get("paper", {}).get("enabled", False),
        },
        "disclaimer": DISCLAIMER,
        "signals": results["signals"],
        "spreads": results.get("spreads", []),
        "paper": results.get("paper"),
        "skipped": results["skipped"],
        "errors": results["errors"],
    }


def _position_label(p: dict) -> str:
    if p["kind"] == "spread":
        return (f"${p['short_strike']:.0f}/${p['long_strike']:.0f} "
                f"spread {p['expiry']}")
    return f"${p['strike']:.0f} put {p['expiry']}"


def _position_earnings_cell(p: dict) -> str:
    """Render a paper position's earnings-exposure cell. '—' when the
    position was never earnings-checked (opened before v13, or the check
    is off in config); otherwise the shared signal rendering — the date
    with ⚠ when earnings fall inside the remaining window, 'unknown'
    when the calendar had nothing."""
    if "earnings" not in p:
        return "—"
    return _earnings_cell(p)


def render_paper_section(paper: dict) -> list:
    """Markdown for the paper position-tracking section."""
    lines = ["## Paper positions (research)", "",
             "Paper-trade only — these positions were never opened at a "
             "broker. Exits follow the same mechanical rules as the exit "
             "plans above (50% of max profit, 21 DTE, 2x-premium loss).",
             ""]
    opened, open_, closed = (paper["opened"], paper["open"],
                             paper["closed_today"])
    lines.append(f"Opened this run: {len(opened)} · "
                 f"Closed this run: {len(closed)} · "
                 f"Still open: {len(open_)}.")
    lines.append("")
    newly = paper.get("newly_exposed") or []
    if newly:
        lines.append("### Newly earnings-exposed today")
        lines.append("")
        for p in newly:
            earn_date = (p.get("earnings") or {}).get("date")
            lines.append(f"- **{p['ticker']}** {_position_label(p)} "
                         f"(opened {p['opened']}): earnings {earn_date} "
                         f"now falls inside the remaining window. "
                         f"Informational only — the position stays open "
                         f"and exits stay mechanical.")
        alerts = paper.get("exposure_alerts") or {}
        if alerts.get("fired"):
            lines.append(f"Alert sent via {', '.join(alerts['channels'])}.")
        elif alerts.get("would_fire"):
            lines.append("Alert would fire (dry run — nothing sent).")
        elif alerts.get("status") == "skipped":
            lines.append("Alert not sent — run with --alerts to send "
                         "exposure alerts.")
        elif alerts.get("channel_errors"):
            lines.append("Alert attempted but the channel failed: "
                         + "; ".join(f"{ch}: {err}" for ch, err
                                     in alerts["channel_errors"].items())
                         + ".")
        lines.append("")
    if closed:
        lines.append("### Closed today")
        lines.append("")
        lines.append("| Ticker | Position | Opened | Exit | P&L/contract | Earnings |")
        lines.append("|---|---|---|---|---|---|")
        for c in closed:
            lines.append(
                f"| {c['ticker']} | {_position_label(c)} | {c['opened']} | "
                f"{c['exit_reason']} | ${c['realized']:+.2f} | "
                f"{_position_earnings_cell(c)} |")
        lines.append("")
    if open_:
        lines.append("### Open")
        lines.append("")
        lines.append("| Ticker | Position | Entry credit | Mark | "
                     "Unrealized | DTE | Days held | Earnings |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in open_:
            unreal = (f"${p['last_unrealized']:+.2f}"
                      if p.get("last_unrealized") is not None else "stale")
            mark = (f"${p['last_mark']:.2f}"
                    if p.get("last_mark") is not None else "stale")
            dte = p.get("last_dte", "—")
            held = p.get("days_held", "—")
            lines.append(
                f"| {p['ticker']} | {_position_label(p)} | "
                f"${p['entry_credit']:.2f} | {mark} | {unreal} | "
                f"{dte} | {held} | {_position_earnings_cell(p)} |")
        lines.append("")
    lines.append("Earnings exposure is re-checked at every mark against the "
                 "position's remaining window (today to expiry) and shown "
                 "with ⚠ when a known date falls inside it. It is "
                 "informational only — a position never exits early on "
                 "earnings; exits stay mechanical. A position can become "
                 "exposed after opening, when the date is announced later "
                 "or was unknown at entry; `unknown` is a gap in the "
                 "calendar data, not evidence of no earnings.")
    lines.append("")
    return lines


def render_markdown(payload: dict) -> str:
    """Render the daily Markdown report."""
    lines = [
        f"# Options premium-selling signals — {payload['as_of']}",
        "",
        f"Strategy: {payload['strategy']} · Source: {payload['data_source']}",
        "",
        f"> {payload['disclaimer']}",
        "",
    ]
    signals = payload["signals"]
    if not signals:
        lines.append("No signals passed all filters today.")
    else:
        lines.append(f"## Signals ({len(signals)})")
        lines.append("")
        lines.append("| Ticker | Underlying | Strike | Expiry | DTE | Δ | IV | "
                     "IV rank | Premium | Ann. yield | Bid/Ask | OI | Breakeven | Earnings | Dividend |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in signals:
            anny = f"{s['annualized_yield']:.1%}" if s["annualized_yield"] is not None else "n/a"
            lines.append(
                f"| {s['ticker']} | ${s['underlying']:.2f} | ${s['strike']:.2f} | "
                f"{s['expiry']} | {s['dte']} | {s['delta']:.2f} | {s['iv']:.2%} | "
                f"{_iv_rank_cell(s)} | {s['premium_pct']:.2%} | {anny} | "
                f"${s['bid']:.2f}/${s['ask']:.2f} | {s['open_interest']} | "
                f"${s['breakeven']:.2f} | {_earnings_cell(s)} | {_dividend_cell(s)} |")
        lines.append("")
        lines.append("## Exit plan (per signal)")
        lines.append("")
        for s in signals:
            e = s["exit_plan"]
            lines.append(f"- **{s['ticker']} ${s['strike']:.0f} put {s['expiry']}**: "
                         f"{e['profit_target']}; {e['time_exit']}; {e['loss_manage']}.")
        lines.append("")

    spreads = payload.get("spreads", [])
    if spreads:
        lines.append(f"## Bull put spreads ({len(spreads)}) — defined risk")
        lines.append("")
        lines.append("Short leg = the CSP candidate above; long leg = a cheaper "
                     "put on the same expiry, capping max loss at "
                     "(width − credit).")
        lines.append("")
        lines.append("| Ticker | Short/Long | Width | Net credit | Max loss | ROI | "
                     "Ann. ROI | Breakeven | Expiry | DTE |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for sp in spreads:
            anny = f"{sp['annualized_roi']:.1%}" if sp["annualized_roi"] is not None else "n/a"
            lines.append(
                f"| {sp['ticker']} | ${sp['short_strike']:.2f}/${sp['long_strike']:.2f} | "
                f"${sp['width']:.2f} | ${sp['net_credit']:.2f} | "
                f"${sp['max_loss']:.2f} | {sp['roi']:.1%} | {anny} | "
                f"${sp['breakeven']:.2f} | {sp['expiry']} | {sp['dte']} |")
        lines.append("")
        lines.append("## Exit plan (per spread)")
        lines.append("")
        for sp in spreads:
            e = sp["exit_plan"]
            lines.append(f"- **{sp['ticker']} ${sp['short_strike']:.0f}/"
                         f"${sp['long_strike']:.0f} {sp['expiry']}**: "
                         f"{e['profit_target']}; {e['time_exit']}; {e['loss_manage']}.")
        lines.append("")

    paper = payload.get("paper")
    if paper is not None:
        lines.extend(render_paper_section(paper))

    if payload["skipped"]:
        lines.append("## No signal (why)")
        lines.append("")
        for item in payload["skipped"]:
            lines.append(f"- **{item['ticker']}**: {item['reason']}")
        lines.append("")

    if payload["errors"]:
        lines.append("## Data errors")
        lines.append("")
        for err in payload["errors"]:
            lines.append(f"- {err}")
        lines.append("")

    lines += [
        "## Method notes",
        "",
        "- **IV rank (true)**: once the screener has collected at least "
        "10 daily IV snapshots for a ticker (configurable), the IV-rank column "
        "shows the true rank — today's contract IV positioned inside the "
        "ticker's own observed IV range for the same DTE band — with the "
        "snapshot count beside it. Only same-band snapshots are compared, so "
        "the term structure doesn't contaminate the rank.",
        "- **IV rank (proxy)**: before the history is deep enough, the "
        "column shows a proxy — current IV inside the ticker's 1-year "
        "realized-vol range — labeled `(proxy)`. No paid historical-IV feed "
        "is needed; the history is built from this tool's own daily "
        "snapshots (stored in `iv_history/`, git-ignored).",
        "- Bull put spreads: the short leg is the CSP candidate from the table "
        "above; the long leg is the cheaper same-expiry put, 2–6% of the short "
        "strike lower, that maximizes return on risk while passing liquidity "
        "filters and collecting at least 1/3 of the spread width in credit.",
        "- Earnings gate: a signal whose option window contains a known "
        "earnings date is skipped (config `earnings.exclude`, default on) "
        "and the skip reason names the date — an earnings gap can run "
        "through the short strike overnight. When the calendar has no date "
        "for a ticker, the Earnings column says `unknown`: that is a gap in "
        "the data, not evidence of no earnings. With `exclude: false` the "
        "signal still prints, flagged with its earnings date (⚠).",
        "- Dividend flag: a known ex-dividend date inside the option window "
        "is shown in the Dividend column (⚠) but the signal still prints — "
        "the stock drops by roughly the dividend on the ex-date, nudging "
        "the underlying toward the short strike and raising early-assignment "
        "odds if the put is in the money near the date, but for a ~30-delta "
        "OTM put that is a nudge, not an overnight gap, so exclusion is "
        "opt-in (`dividends.exclude`, default off) rather than the default. "
        "When the calendar has no ex-date, the column says `unknown`: a gap "
        "in the data, not evidence of no dividend. The dividend amount is "
        "not shown — yfinance does not publish it reliably.",
        "- Deltas are Black-Scholes estimates using a fixed risk-free rate "
        "assumption; broker greeks may differ slightly.",
        "- Signals are candidates for *research*, not orders. Nothing here "
        "places or suggests placing a trade automatically.",
        "- Paper positions: positions opened automatically from *newly fired* "
        "alerts (never from plain signals) and marked against the next "
        "day's chain. A position only exits via the mechanical rules above; "
        "a missing quote or an unparseable expiry holds the position as "
        "stale rather than guessing. Closed positions are never re-opened "
        "by later alerts. State lives in `positions/` (git-ignored). "
        "Earnings exposure is re-checked at every mark against the "
        "position's remaining window and shown in the Earnings column — "
        "informational only, it never triggers an exit. When a still-open "
        "position becomes earnings-exposed at a mark (v14), the report "
        "names it and — on runs invoked with `--alerts` — the same "
        "email/webhook channels send one alert per position+earnings-date, "
        "deduped so a re-run never re-fires; a `--paper-track`-only run "
        "records the exposure but sends nothing.",
    ]
    return "\n".join(lines) + "\n"


def write_reports(payload: dict, out_dir: str) -> tuple[str, str]:
    """Write report_<date>.md and report_<date>.json; return the two paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"report_{payload['as_of']}")
    md_path, json_path = base + ".md", base + ".json"
    with open(md_path, "w") as f:
        f.write(render_markdown(payload))
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return md_path, json_path
