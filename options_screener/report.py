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
        },
        "disclaimer": DISCLAIMER,
        "signals": results["signals"],
        "spreads": results.get("spreads", []),
        "skipped": results["skipped"],
        "errors": results["errors"],
    }


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
                     "IV-vs-HV | Premium | Ann. yield | Bid/Ask | OI | Breakeven |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in signals:
            ivh = f"{s['iv_vs_hv_rank']:.2f}" if s["iv_vs_hv_rank"] is not None else "n/a"
            anny = f"{s['annualized_yield']:.1%}" if s["annualized_yield"] is not None else "n/a"
            lines.append(
                f"| {s['ticker']} | ${s['underlying']:.2f} | ${s['strike']:.2f} | "
                f"{s['expiry']} | {s['dte']} | {s['delta']:.2f} | {s['iv']:.2%} | "
                f"{ivh} | {s['premium_pct']:.2%} | {anny} | "
                f"${s['bid']:.2f}/${s['ask']:.2f} | {s['open_interest']} | "
                f"${s['breakeven']:.2f} |")
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
        "- IV-vs-HV rank is a proxy: current contract IV positioned inside the "
        "ticker's 1-year realized-vol range. True IV rank needs a historical-IV "
        "feed (planned for v2).",
        "- Bull put spreads: the short leg is the CSP candidate from the table "
        "above; the long leg is the cheaper same-expiry put, 2–6% of the short "
        "strike lower, that maximizes return on risk while passing liquidity "
        "filters and collecting at least 1/3 of the spread width in credit.",
        "- Deltas are Black-Scholes estimates using a fixed risk-free rate "
        "assumption; broker greeks may differ slightly.",
        "- Signals are candidates for *research*, not orders. Nothing here "
        "places or suggests placing a trade automatically.",
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
