"""Tests for the v3 alerting module (email/webhook on new signals)."""

import json
import os

import pytest

from options_screener.alerts import (dedupe, evaluate, load_state,
                                     render_email_body,
                                     render_email_subject,
                                     render_webhook_payload, run_alerts,
                                     save_state, signal_key, spread_key)

CSP = {
    "ticker": "SPY", "expiry": "2026-11-20", "strike": 640.0, "dte": 35,
    "delta": -0.30, "premium_pct": 0.012, "annualized_yield": 0.12,
    "breakeven": 632.32, "iv": 0.20, "iv_rank_true": None, "iv_rank_obs": 0,
    "open_interest": 5000, "bid": 7.60, "ask": 7.75,
}

SPREAD = {
    "ticker": "SPY", "expiry": "2026-11-20", "short_strike": 640.0,
    "long_strike": 620.0, "width": 20.0, "net_credit": 7.00,
    "max_loss": 13.0, "roi": 0.538, "breakeven": 633.0,
}

RULES = {"min_annualized_yield": 0.10, "min_spread_roi": 0.30,
         "min_iv_rank_true": 0.75}


def _payload(signals=None, spreads=None):
    return {"as_of": "2026-10-16", "signals": signals or [],
            "spreads": spreads or []}


def test_signal_key_stable():
    assert signal_key(CSP) == "SPY|2026-11-20|640.0"
    assert spread_key(SPREAD) == "SPY|2026-11-20|640.0/620.0"


def test_evaluate_yield_threshold():
    items = evaluate(_payload(signals=[CSP]), RULES)
    assert len(items) == 1
    assert items[0]["kind"] == "csp"
    assert "annualized yield 12.0% >= 10.0%" in items[0]["reason"]

    low = dict(CSP, annualized_yield=0.05)
    assert evaluate(_payload(signals=[low]), RULES) == []


def test_evaluate_iv_rank_rule():
    rich = dict(CSP, annualized_yield=0.05, iv_rank_true=0.82, iv_rank_obs=12)
    items = evaluate(_payload(signals=[rich]), RULES)
    assert len(items) == 1
    assert "true IV rank 0.82 >= 0.75" in items[0]["reason"]

    # None rank never triggers the IV rule (only the yield rule can fire it)
    proxy = dict(CSP, annualized_yield=0.05, iv_rank_true=None)
    no_iv = evaluate(_payload(signals=[proxy]),
                     {"min_annualized_yield": 0.10, "min_iv_rank_true": 0.75})
    assert no_iv == []


def test_evaluate_spread_roi():
    items = evaluate(_payload(spreads=[SPREAD]), RULES)
    assert len(items) == 1
    assert items[0]["kind"] == "spread"
    assert "return on risk 53.8% >= 30.0%" in items[0]["reason"]

    low = dict(SPREAD, roi=0.10)
    assert evaluate(_payload(spreads=[low]), RULES) == []


def test_evaluate_one_item_per_signal_combined_reasons():
    rich = dict(CSP, iv_rank_true=0.90, iv_rank_obs=15)
    items = evaluate(_payload(signals=[rich]), RULES)
    assert len(items) == 1
    assert "annualized yield" in items[0]["reason"]
    assert "true IV rank" in items[0]["reason"]


def test_dedupe_only_new_fire():
    state = {"fired_keys": [signal_key(CSP)]}
    items = evaluate(_payload(signals=[CSP]), RULES)
    assert dedupe(items, state) == []

    other = dict(CSP, strike=635.0)
    items = evaluate(_payload(signals=[other]), RULES)
    assert [it["key"] for it in dedupe(items, state)] == [signal_key(other)]


def test_state_roundtrip_and_corrupt(tmp_path):
    d = str(tmp_path)
    assert load_state(d) == {}
    save_state(d, {"fired_keys": ["a|b|c"]})
    assert load_state(d)["fired_keys"] == ["a|b|c"]
    # corrupt file tolerated, not fatal
    with open(os.path.join(d, "alerts.json"), "w") as f:
        f.write("{not json")
    assert load_state(d) == {}
    # non-dict json tolerated
    with open(os.path.join(d, "alerts.json"), "w") as f:
        f.write("[1,2]")
    assert load_state(d) == {}


def test_render_includes_disclaimer_and_fields():
    items = evaluate(_payload(signals=[CSP], spreads=[SPREAD]), RULES)
    body = render_email_body("2026-10-16", items)
    assert "not financial advice" in body
    assert "$640.00 put exp 2026-11-20" in body
    assert "bull put $640.00/$620.00" in body
    subj = render_email_subject("2026-10-16", items)
    assert "2 new signal alert(s)" in subj and "SPY" in subj

    wh = render_webhook_payload("2026-10-16", items)
    assert wh["as_of"] == "2026-10-16"
    assert len(wh["alerts"]) == 2
    assert set(wh["alerts"][0]) == {"kind", "ticker", "reason", "detail"}


def test_run_alerts_disabled_channels_fires_nothing(tmp_path):
    cfg = {"alerts": {"enabled": True, "state_dir": str(tmp_path),
                      "rules": RULES,
                      "email": {"enabled": False},
                      "webhook": {"enabled": False}}}
    res = run_alerts(_payload(signals=[CSP]), cfg)
    assert res["fired"] == [] and res["channels"] == []
    # no channels -> state untouched (nothing marked fired)
    assert load_state(str(tmp_path)) == {}


def test_run_alerts_dry_run_never_sends_or_persists(tmp_path):
    cfg = {"alerts": {"enabled": True, "state_dir": str(tmp_path),
                      "rules": RULES,
                      "email": {"enabled": True},
                      "webhook": {"enabled": True}}}
    res = run_alerts(_payload(signals=[CSP]), cfg, dry_run=True)
    assert len(res["would_fire"]) == 1 and res["fired"] == []
    assert load_state(str(tmp_path)) == {}


def test_run_alerts_misconfigured_email_reports_clean_error(tmp_path):
    cfg = {"alerts": {"enabled": True, "state_dir": str(tmp_path),
                      "rules": RULES, "email": {"enabled": True}}}
    res = run_alerts(_payload(signals=[CSP]), cfg)
    assert "email" in res["channels"]
    assert "misconfigured" in res["channel_errors"]["email"]
    assert res["fired"] == []


def test_run_alerts_missing_webhook_url_reports_clean_error(tmp_path):
    cfg = {"alerts": {"enabled": True, "state_dir": str(tmp_path),
                      "rules": RULES,
                      "webhook": {"enabled": True}},
           "alerts_env_missing": True}
    res = run_alerts(_payload(signals=[CSP]), cfg)
    assert "webhook" in res["channels"]
    assert "ALERT_WEBHOOK_URL is not set" in res["channel_errors"]["webhook"]
    assert res["fired"] == []


def test_run_alerts_webhook_success_marks_fired(tmp_path, monkeypatch):
    import options_screener.alerts as alerts_mod

    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/hook")

    captured = {}

    def fake_send(url, payload):
        captured["url"] = url
        captured["n"] = len(payload["alerts"])
    monkeypatch.setattr(alerts_mod, "send_webhook", fake_send)

    d = str(tmp_path)
    cfg = {"alerts": {"enabled": True, "state_dir": d, "rules": RULES,
                      "webhook": {"enabled": True}}}
    res = run_alerts(_payload(signals=[CSP]), cfg)
    assert len(res["fired"]) == 1
    assert captured["url"] == "https://example.com/hook"
    assert captured["n"] == 1
    assert signal_key(CSP) in load_state(d)["fired_keys"]

    # second run: same signal must not fire again
    res2 = run_alerts(_payload(signals=[CSP]), cfg)
    assert res2["fired"] == []


def test_run_alerts_webhook_failure_does_not_kill_run(tmp_path, monkeypatch):
    import options_screener.alerts as alerts_mod

    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/hook")

    def boom(url, payload):
        raise OSError("connection refused")
    monkeypatch.setattr(alerts_mod, "send_webhook", boom)

    d = str(tmp_path)
    cfg = {"alerts": {"enabled": True, "state_dir": d, "rules": RULES,
                      "webhook": {"enabled": True}}}
    res = run_alerts(_payload(signals=[CSP]), cfg)  # must not raise
    assert res["fired"] == []
    assert "connection refused" in res["channel_errors"]["webhook"]
    assert load_state(d) == {}  # undelivered items stay unmarked


def test_credentials_never_in_config_defaults():
    # The shipped config must stay credential-free: every sensitive field
    # ships empty and reads from env.
    import yaml
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = yaml.safe_load(open(os.path.join(here, "..", "config.yaml")))
    email = cfg["alerts"]["email"]
    assert email["to"] == "" and email["smtp_host"] == ""
    assert email["username"] == "" and email["from_"] == ""
    assert cfg["alerts"]["webhook"]["url"] == ""
    assert cfg["alerts"]["enabled"] is False  # opt-in
