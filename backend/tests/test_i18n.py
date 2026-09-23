"""B2-1: multilingual alerts. `?lang=hi` must translate every alert type, unknown languages fall
back to English, and English output is unchanged apart from the additive `lang` / `voice_text`."""
from __future__ import annotations

import asyncio
import re

from test_live_pipeline import Site, msg, person

from app.decision.hub import CabHub
from app.i18n import localize, render_explanation, resolve_lang, t
from app.i18n.catalog import EN, HI
from app.models.anomaly import explain_items
from app.rules import safety

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _alert_keys() -> list[tuple[str, str | None]]:
    keys = []
    for k in EN:
        if k.endswith(".msg"):
            base = k[:-4]
            head, _, tail = base.rpartition(".")
            if tail in ("warning", "high", "critical"):
                keys.append((head, tail))
            else:
                keys.append((base, None))
    return keys


def _dummy_alert(alert_type: str, variant: str | None) -> dict:
    fields = set(PLACEHOLDER.findall(EN[f"{alert_type + '.' + variant if variant else alert_type}.msg"]))
    a = safety._alert(alert_type, "critical", {f: "7" for f in fields}, variant=variant,
                      note={"radius": "8", "reasons": ["rainy", "low visibility"]})
    a.pop("_key")
    return a


def test_hindi_catalogue_covers_every_english_key_with_same_placeholders():
    assert set(EN) == set(HI)
    for key, text in EN.items():
        assert set(PLACEHOLDER.findall(text)) == set(PLACEHOLDER.findall(HI[key])), key


def test_every_alert_type_translates_to_hindi():
    keys = _alert_keys()
    types = {a for a, _ in keys}
    assert types >= {"proximity_person_blind_spot", "proximity_person_danger", "seatbelt_off_moving",
                     "operator_out_of_cab", "tilt_warning", "lightning_nearby", "refuel_engine_on",
                     "geofence_violation"}
    for alert_type, variant in keys:
        alert = _dummy_alert(alert_type, variant)
        hi = localize({"alerts": [alert]}, "hi")["alerts"][0]
        for field in ("message", "recommended_action", "voice_text"):
            assert DEVANAGARI.search(hi[field]), (alert_type, variant, field, hi[field])
        assert "{" not in hi["message"]
        assert DEVANAGARI.search(hi["adjusted_threshold_note"])
        assert hi["alert_id" if "alert_id" in hi else "alert_type"]


def test_english_text_is_unchanged_and_gains_only_additive_keys():
    site = Site(weather="Rainy")
    site.sec += 1
    site.feed(msg("proximity", site.sec, objects=[person(3.0)]))
    payload = site.assess().payload
    en = localize(payload, "en")
    before, after = payload["alerts"][0], en["alerts"][0]
    assert after["message"] == before["message"] == "Worker 3.0 m behind you, in blind spot"
    assert after["recommended_action"] == "Stop swing and sound horn"
    assert after["adjusted_threshold_note"] == "Danger zone widened to 8 m (rainy)"
    assert en["lang"] == "en" and after["voice_text"]
    assert set(before) <= set(after)          # nothing removed or renamed


def test_scenario_alert_in_hindi_with_numbers_and_voice():
    site = Site(weather="Rainy")
    site.sec += 1
    site.feed(msg("proximity", site.sec, objects=[person(3.0)]))
    hi = localize(site.assess().payload, "hi")
    a = hi["alerts"][0]
    assert hi["lang"] == "hi"
    assert "3.0" in a["message"] and DEVANAGARI.search(a["message"])
    assert "8" in a["adjusted_threshold_note"] and "बारिश" in a["adjusted_threshold_note"]
    assert DEVANAGARI.search(a["voice_text"]) and a["voice_text"].count("।") <= 2
    assert hi["zones"]["reason"] == "बारिश"


def test_unknown_language_falls_back_to_english():
    assert resolve_lang("xx") == "en" and resolve_lang(None) == "en" and resolve_lang("HI-in") == "hi"
    site = Site()
    site.sec += 1
    site.feed(msg("proximity", site.sec, objects=[person(3.0)]))
    payload = site.assess().payload
    xx = localize(payload, "xx")
    assert xx["lang"] == "en"
    assert xx["alerts"][0]["message"] == localize(payload, "en")["alerts"][0]["message"]


def test_anomaly_explanations_translate():
    row = {"idle_ratio": 0.8, "idle_streak_min": 25, "truck_wait_min": 5}
    items = explain_items(row, {"idle_ratio": 0.2}, "excessive_idling")
    en = render_explanation("en", items)
    hi = render_explanation("hi", items)
    assert en[0] == "Idle ratio: 0.80 vs your baseline 0.20" and "Likely waiting for haul truck" in en
    assert len(hi) == len(en) and all(DEVANAGARI.search(s) for s in hi)
    flagged = render_explanation("hi", explain_items({}, {}, "fuel_theft"))
    assert DEVANAGARI.search(flagged[0])
    assert t("hi", "no.such.key") == "no.such.key"


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


def test_hub_translates_per_connection():
    site = Site()
    site.sec += 1
    site.feed(msg("proximity", site.sec, objects=[person(3.0)]))
    payload = site.assess().payload
    hub = CabHub(site.sm, site.dl)
    en_ws, hi_ws, xx_ws = FakeSocket(), FakeSocket(), FakeSocket()

    async def run():
        await hub.connect("EXC001", en_ws, resolve_lang("en"))
        await hub.connect("EXC001", hi_ws, resolve_lang("hi"))
        await hub.connect("EXC001", xx_ws, resolve_lang("xx"))
        await hub.broadcast("EXC001", payload)
        hub.latest["EXC001"] = payload
        late = FakeSocket()
        await hub.connect("EXC001", late, "hi")          # late joiner gets the latest, translated
        return late

    late = asyncio.run(run())
    assert en_ws.sent[0]["lang"] == "en" and not DEVANAGARI.search(en_ws.sent[0]["alerts"][0]["message"])
    assert hi_ws.sent[0]["lang"] == "hi" and DEVANAGARI.search(hi_ws.sent[0]["alerts"][0]["message"])
    assert xx_ws.sent[0]["lang"] == "en"
    assert DEVANAGARI.search(late.sent[0]["alerts"][0]["voice_text"])
