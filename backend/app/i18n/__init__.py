"""Alert / explanation localisation for the cab UI.

The rules and the anomaly explainer build their English text through `t("en", ...)` and also
keep the structured pieces (`alert["params"]`, `anomaly["explanation_items"]`). The cab hub
re-renders those pieces per connection with `localize()`, so the wire format only gains
`lang`, `voice_text`, `params` and `explanation_items`; no existing key changes name or type.
"""
from __future__ import annotations

from app.i18n.catalog import CATALOGS, DEFAULT_LANG, EN

SUPPORTED = tuple(CATALOGS)


def resolve_lang(lang: str | None) -> str:
    """Normalise ?lang= (case, region suffix like hi-IN); unknown -> English."""
    if not lang:
        return DEFAULT_LANG
    base = str(lang).strip().lower().replace("_", "-").split("-")[0]
    return base if base in CATALOGS else DEFAULT_LANG


def t(lang: str, key: str, **params) -> str:
    template = CATALOGS.get(lang, EN).get(key)
    if template is None:
        template = EN.get(key, key)          # "" is a valid entry (empty unit), so test for None
    try:
        return template.format(**params)
    except (KeyError, IndexError):
        return template


def _reasons(lang: str, reasons: list[str]) -> str:
    return ", ".join(t(lang, f"reason.{r}") for r in reasons)


def alert_key(alert: dict) -> str:
    variant = (alert.get("params") or {}).get("variant")
    return f"{alert['alert_type']}.{variant}" if variant else alert["alert_type"]


def render_alert(lang: str, alert: dict) -> dict:
    """Copy of `alert` with message / action / note / voice_text in `lang`."""
    out = dict(alert)
    key = alert_key(alert)
    params = {k: v for k, v in (alert.get("params") or {}).items() if k not in ("variant", "note")}
    out["message"] = t(lang, f"{key}.msg", **params)
    out["recommended_action"] = t(lang, f"{key}.act", **params)
    out["voice_text"] = t(lang, f"{key}.voice", **params)
    note = (alert.get("params") or {}).get("note")
    if note:
        out["adjusted_threshold_note"] = t(lang, "note.zone_widened", radius=note["radius"],
                                           reason=_reasons(lang, note["reasons"]))
    return out


def render_explanation(lang: str, items: list[dict]) -> list[str]:
    out = []
    for item in items:
        params = dict(item.get("params") or {})
        if "feature" in params:
            params["feature"] = t(lang, f"feature.{params['feature']}")
        if "unit" in params:
            params["unit"] = t(lang, f"unit.{params['unit']}")
        if "anomaly" in params:
            params["anomaly"] = t(lang, f"anomaly.{params['anomaly']}") if f"anomaly.{params['anomaly']}" in EN \
                else params["anomaly"].replace("_", " ")
        out.append(t(lang, item["key"], **params))
    return out


def localize(payload: dict, lang: str) -> dict:
    """The `assessment` message in `lang` (English payloads still gain lang + voice_text)."""
    lang = resolve_lang(lang)
    out = dict(payload)
    out["lang"] = lang
    out["alerts"] = [render_alert(lang, a) for a in payload.get("alerts", [])]
    anomalies = []
    for a in payload.get("anomalies", []):
        a = dict(a)
        if a.get("explanation_items"):
            a["explanation"] = render_explanation(lang, a["explanation_items"])
        anomalies.append(a)
    out["anomalies"] = anomalies
    zones = payload.get("zones")
    if zones and zones.get("reason") and lang != DEFAULT_LANG:
        zones = dict(zones)
        zones["reason"] = _reasons(lang, [r.strip() for r in zones["reason"].split(",")])
        out["zones"] = zones
    recs = []
    for r in payload.get("training_recommendations", []):
        r = dict(r)
        translated = t(lang, f"module.{r.get('module_id')}")
        if translated != f"module.{r.get('module_id')}":
            r["title"] = translated
        recs.append(r)
    out["training_recommendations"] = recs
    return out
