"""Tests for v22: alert on newly dividend-exposed paper positions.

v21 only *flagged* dividend exposure in the report — a position that
becomes dividend-exposed after opening sat there silently until someone
read the daily report. v22 detects the flip at the mark (outside/unknown
-> inside, or inside to a *different* ex-date) and alerts on the same
email/webhook channels as signal alerts, once per position+ex-date. The
alert is informational only: an ex-date is a nudge toward the short
strike, not an overnight gap — it never exits a position, and a position
that closes at the same mark is never alerted on. Network-independent.
"""

from datetime import date, timedelta

import options_screener.alerts as alerts_mod
from options_screener.alerts import (dividend_exposure_alert_items,
                                     dividend_exposure_key,
                                     render_dividend_exposure_subject,
                                     run_dividend_exposure_alerts)
from options_screener.exits import (open_from_signal, open_from_spread,
                                    update_positions)
from options_screener.report import render_paper_section

EXITS = {"profit_target_pct": 0.50, "manage_dte": 21,
         "loss_manage_multiple": 2.0}

TODAY = date.today()


def _expiry(days=30):
    return (TODAY + timedelta(days=days)).isoformat()


def _as_of(days=0):
    return (TODAY + timedelta(days=days)).isoformat()


def csp_signal(**kw):
    sig = {"ticker": "NVDA", "strike": 140.0, "expiry": _expiry(),
           "dte": 30, "delta": -0.29, "mid": 4.00, "underlying": 150.0,
           "breakeven": 136.0}
    sig.update(kw)
    return sig


def bull_spread(**kw):
    sp = {"ticker": "NVDA", "short_strike": 140.0, "long_strike": 135.0,
          "expiry": _expiry(), "dte": 30, "net_credit": 3.00,
          "max_loss": 2.00, "underlying": 150.0, "breakeven": 137.0}
    sp.update(kw)
    return sp


def _snap(mark=3.00, dte=30, underlying=150.0):
    # mark 3.00 vs entry credit 4.00: above the 50% profit trigger (2.00),
    # below the 2x loss trigger (12.00), DTE 30 > 21 -> hold.
    return {"date": _as_of(), "underlying": underlying,
            "mark": mark, "dte": dte}


def _store(pos):
    return {"open": [pos], "closed": []}


def _cfg(tmp_path, webhook=False, email=False):
    return {"alerts": {"enabled": True, "state_dir": str(tmp_path),
                       "email": {"enabled": email},
                       "webhook": {"enabled": webhook}}}


# --- detection in update_positions -----------------------------------------

def test_outside_to_inside_is_newly_dividend_exposed():
    pos = open_from_signal(csp_signal(
        dividends={"status": "outside", "date": "2027-02-10"}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap()}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=5)})
    assert len(res["newly_dividend_exposed"]) == 1
    assert res["newly_dividend_exposed"][0]["key"] == pos["key"]
    assert res["newly_dividend_exposed"][0]["dividends"]["status"] == "inside"
    assert res["newly_dividend_exposed"][0]["dividends"]["date"] == (
        TODAY + timedelta(days=5)).isoformat()


def test_unknown_to_inside_is_newly_dividend_exposed():
    pos = open_from_signal(csp_signal(
        dividends={"status": "unknown", "date": None}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap()}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=5)})
    assert len(res["newly_dividend_exposed"]) == 1


def test_no_prior_dividend_info_to_inside_is_newly_exposed():
    pos = open_from_signal(csp_signal())  # no "dividends" key at all
    assert "dividends" not in pos
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap()}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=5)})
    assert len(res["newly_dividend_exposed"]) == 1


def test_inside_same_date_is_not_new():
    ex = (TODAY + timedelta(days=5)).isoformat()
    pos = open_from_signal(csp_signal(
        dividends={"status": "inside", "date": ex}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap()}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=5)})
    assert res["newly_dividend_exposed"] == []


def test_inside_moved_date_is_new():
    old = (TODAY + timedelta(days=5)).isoformat()
    pos = open_from_signal(csp_signal(
        dividends={"status": "inside", "date": old}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap()}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=9)})
    assert len(res["newly_dividend_exposed"]) == 1
    assert res["newly_dividend_exposed"][0]["dividends"]["date"] == (
        TODAY + timedelta(days=9)).isoformat()


def test_outside_stays_outside_is_not_new():
    pos = open_from_signal(csp_signal(
        dividends={"status": "outside", "date": "2027-02-10"}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap()}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=40)})
    assert res["newly_dividend_exposed"] == []


def test_no_dividends_map_means_no_detection():
    pos = open_from_signal(csp_signal(
        dividends={"status": "outside", "date": "2027-02-10"}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap()}, EXITS,
                           as_of=_as_of(), dividends_map=None)
    assert res["newly_dividend_exposed"] == []
    # Exposure is preserved untouched when no map is supplied.
    assert pos["dividends"]["status"] == "outside"


def test_position_closing_today_is_never_newly_dividend_exposed():
    # mark 1.50 <= 50% profit trigger (2.00): closes for profit at this
    # mark — even though the ex-date moved inside, there is nothing left
    # to warn about.
    pos = open_from_signal(csp_signal(
        dividends={"status": "outside", "date": "2027-02-10"}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap(mark=1.50)}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=5)})
    assert res["newly_dividend_exposed"] == []
    assert len(res["closed_today"]) == 1


def test_spread_outside_to_inside_is_newly_dividend_exposed():
    pos = open_from_spread(bull_spread(
        dividends={"status": "outside", "date": "2027-02-10"}))
    store = _store(pos)
    # Spread mark is short-mid minus long-mid; 2.50 holds (profit trigger
    # 1.50, loss trigger 9.00).
    res = update_positions(store, {pos["key"]: _snap(mark=2.50)}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=5)})
    assert len(res["newly_dividend_exposed"]) == 1


def test_stale_mark_still_detects_dividend_exposure():
    # No chain quote today (mark None) -> hold as stale, but the dividend
    # refresh still ran, so the flip is still detected.
    pos = open_from_signal(csp_signal(
        dividends={"status": "unknown", "date": None}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap(mark=None)}, EXITS,
                           as_of=_as_of(),
                           dividends_map={"NVDA": TODAY + timedelta(days=5)})
    assert len(res["newly_dividend_exposed"]) == 1


def test_earnings_and_dividend_flips_are_independent():
    # Both flips at the same mark are reported on their own lists — the
    # earnings alert must not swallow the dividend one, or vice versa.
    pos = open_from_signal(csp_signal(
        earnings={"status": "outside", "date": "2027-01-28"},
        dividends={"status": "outside", "date": "2027-02-10"}))
    store = _store(pos)
    res = update_positions(store, {pos["key"]: _snap()}, EXITS,
                           as_of=_as_of(),
                           earnings_map={"NVDA": TODAY + timedelta(days=7)},
                           dividends_map={"NVDA": TODAY + timedelta(days=5)})
    assert len(res["newly_exposed"]) == 1
    assert len(res["newly_dividend_exposed"]) == 1
    assert res["newly_exposed"][0] is res["newly_dividend_exposed"][0]


# --- alert items ------------------------------------------------------------

def test_dividend_exposure_key_includes_position_and_date():
    pos = open_from_signal(csp_signal())
    assert dividend_exposure_key(pos, "2026-12-02") == (
        f"NVDA|{_expiry()}|140.0|dividends:2026-12-02")


def test_dividend_exposure_alert_items_shape():
    ex_date = (TODAY + timedelta(days=5)).isoformat()
    pos = open_from_signal(csp_signal(
        dividends={"status": "inside", "date": ex_date}))
    items = dividend_exposure_alert_items([pos])
    assert len(items) == 1
    it = items[0]
    assert it["kind"] == "dividend_exposure"
    assert it["ticker"] == "NVDA"
    assert it["key"] == dividend_exposure_key(pos, ex_date)
    assert ex_date in it["reason"]
    assert "informational only" in it["detail"]
    assert "$140.00 put" in it["detail"]


def test_dividend_exposure_alert_items_skip_unknown_date():
    pos = open_from_signal(csp_signal(
        dividends={"status": "unknown", "date": None}))
    assert dividend_exposure_alert_items([pos]) == []


def test_dividend_exposure_subject_names_positions_and_tickers():
    ex_date = (TODAY + timedelta(days=5)).isoformat()
    pos = open_from_signal(csp_signal(
        dividends={"status": "inside", "date": ex_date}))
    items = dividend_exposure_alert_items([pos])
    subject = render_dividend_exposure_subject(_as_of(), items)
    assert subject.startswith(
        "[options-screener] 1 paper position(s) newly dividend-exposed")
    assert "NVDA" in subject


# --- delivery (run_dividend_exposure_alerts) --------------------------------

def _item():
    ex_date = (TODAY + timedelta(days=5)).isoformat()
    pos = open_from_signal(csp_signal(
        dividends={"status": "inside", "date": ex_date}))
    return dividend_exposure_alert_items([pos])[0]


def test_dividend_exposure_dry_run_would_fire_without_state(tmp_path):
    cfg = _cfg(tmp_path, webhook=True)
    res = run_dividend_exposure_alerts([_item()], cfg, as_of=_as_of(),
                                       dry_run=True)
    assert len(res["would_fire"]) == 1
    assert res["fired"] == []
    assert not (tmp_path / "alerts.json").exists()


def test_dividend_exposure_webhook_success_marks_fired_and_dedupes(
        tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/hook")
    sent = []
    monkeypatch.setattr(alerts_mod, "send_webhook",
                        lambda url, payload: sent.append((url, payload)))
    cfg = _cfg(tmp_path, webhook=True)
    res = run_dividend_exposure_alerts([_item()], cfg, as_of=_as_of())
    assert len(res["fired"]) == 1
    assert res["channels"] == ["webhook"]
    assert len(sent) == 1
    assert "[DIVIDEND_EXPOSURE] NVDA" in sent[0][1]["text"]
    # the fixture date is TODAY + 5 days, so it moves with the run date
    assert (TODAY + timedelta(days=5)).isoformat() in sent[0][1]["text"]
    # Second run: same position+date is deduped, nothing re-fires.
    res2 = run_dividend_exposure_alerts([_item()], cfg, as_of=_as_of())
    assert res2["fired"] == [] and res2["would_fire"] == []
    assert len(sent) == 1


def test_dividend_exposure_webhook_failure_does_not_kill_run(
        tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/hook")

    def boom(url, payload):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(alerts_mod, "send_webhook", boom)
    cfg = _cfg(tmp_path, webhook=True)
    res = run_dividend_exposure_alerts([_item()], cfg, as_of=_as_of())
    assert res["fired"] == []
    assert "webhook" in res["channel_errors"]
    assert not (tmp_path / "alerts.json").exists()


def test_dividend_exposure_no_channels_enabled_sends_nothing(tmp_path):
    cfg = _cfg(tmp_path)  # both channels off
    res = run_dividend_exposure_alerts([_item()], cfg, as_of=_as_of())
    assert res["fired"] == [] and res["would_fire"] == []
    assert res["channels"] == []
    assert not (tmp_path / "alerts.json").exists()


def test_dividend_exposure_email_misconfigured_reports_error(tmp_path):
    cfg = _cfg(tmp_path, email=True)  # no SMTP env vars set
    res = run_dividend_exposure_alerts([_item()], cfg, as_of=_as_of())
    assert res["fired"] == []
    assert "email" in res["channel_errors"]


# --- report -----------------------------------------------------------------

def _paper_dict(newly_div, div_alerts=None):
    return {"opened": [], "open": newly_div, "closed_today": [],
            "newly_exposed": [], "exposure_alerts": None,
            "newly_dividend_exposed": newly_div,
            "dividend_exposure_alerts": div_alerts}


def test_report_names_newly_dividend_exposed_positions():
    ex_date = (TODAY + timedelta(days=5)).isoformat()
    pos = open_from_signal(csp_signal(
        dividends={"status": "inside", "date": ex_date}))
    md = "\n".join(render_paper_section(_paper_dict(
        [pos], {"fired": [_item()], "channels": ["webhook"],
                "would_fire": [], "channel_errors": {}})))
    assert "Newly dividend-exposed today" in md
    assert "NVDA" in md and ex_date in md
    assert "Alert sent via webhook" in md
    assert "stays open" in md


def test_report_skips_block_when_nothing_newly_dividend_exposed():
    md = "\n".join(render_paper_section(_paper_dict([])))
    assert "Newly dividend-exposed" not in md


def test_report_notes_skipped_dividend_alerts():
    ex_date = (TODAY + timedelta(days=5)).isoformat()
    pos = open_from_signal(csp_signal(
        dividends={"status": "inside", "date": ex_date}))
    md = "\n".join(render_paper_section(_paper_dict(
        [pos], {"status": "skipped", "reason": "needs --alerts"})))
    assert "Alert not sent" in md


# --- wiring: run_paper_tracking ---------------------------------------------

def test_run_paper_tracking_dry_run_would_fire_dividend_alert(
        tmp_path, monkeypatch):
    import screener
    pos = open_from_signal(csp_signal(
        dividends={"status": "outside", "date": "2027-02-10"}))
    state_dir = tmp_path / "positions"
    from options_screener.exits import save_store
    save_store(str(state_dir), {"open": [pos], "closed": []})
    monkeypatch.setattr(screener, "get_next_dividend",
                        lambda ticker: TODAY + timedelta(days=5))
    cfg = {"alerts": {"enabled": True,
                      "state_dir": str(tmp_path / "alerts"),
                      "email": {"enabled": False},
                      "webhook": {"enabled": False}},
           "paper": {"enabled": True, "state_dir": str(state_dir),
                     "open_from_alerts": False, "dividend_check": True,
                     "dividend_exposure_alerts": True,
                     "earnings_check": False},
           "exits": EXITS}
    payload = {"as_of": _as_of(), "signals": [], "spreads": []}
    market = {"NVDA": {"marks": {140.0: 3.00}, "underlying": 150.0}}
    # alerts_result non-None = the run was invoked with --alerts.
    res = screener.run_paper_tracking(payload, cfg, market,
                                      alerts_result={"fired": []},
                                      dry_run=True)
    assert len(res["newly_dividend_exposed"]) == 1
    assert len(res["dividend_exposure_alerts"]["would_fire"]) == 1
    assert res["dividend_exposure_alerts"]["fired"] == []


def test_run_paper_tracking_without_alerts_run_skips_dividend_sending(
        tmp_path, monkeypatch):
    import screener
    pos = open_from_signal(csp_signal(
        dividends={"status": "outside", "date": "2027-02-10"}))
    state_dir = tmp_path / "positions"
    from options_screener.exits import save_store
    save_store(str(state_dir), {"open": [pos], "closed": []})
    monkeypatch.setattr(screener, "get_next_dividend",
                        lambda ticker: TODAY + timedelta(days=5))
    cfg = {"alerts": {"enabled": True,
                      "state_dir": str(tmp_path / "alerts"),
                      "email": {"enabled": False},
                      "webhook": {"enabled": False}},
           "paper": {"enabled": True, "state_dir": str(state_dir),
                     "open_from_alerts": False, "dividend_check": True,
                     "dividend_exposure_alerts": True,
                     "earnings_check": False},
           "exits": EXITS}
    payload = {"as_of": _as_of(), "signals": [], "spreads": []}
    market = {"NVDA": {"marks": {140.0: 3.00}, "underlying": 150.0}}
    # alerts_result None = --paper-track without --alerts: record, don't send.
    res = screener.run_paper_tracking(payload, cfg, market,
                                      alerts_result=None)
    assert len(res["newly_dividend_exposed"]) == 1
    assert res["dividend_exposure_alerts"]["status"] == "skipped"
