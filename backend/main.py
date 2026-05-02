from pathlib import Path
import json
import uuid
import os
import tempfile
import logging
import threading
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger("farm_api")

BASE_DIR     = Path(__file__).resolve().parent
STORAGE_DIR  = BASE_DIR / "storage"
SETTINGS_PATH = STORAGE_DIR / "settings.json"
ALERTS_PATH   = STORAGE_DIR / "alerts.json"
UPDATES_PATH  = STORAGE_DIR / "updates.json"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH  = STORAGE_DIR / "history.json"
DLI_STATE_PATH = STORAGE_DIR / "dli_state.json"

MAX_ITEMS  = 100
MAX_HISTORY = 48  # keep last 48 records (24 hours at 30-min intervals)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "default_location": "Bekaa Valley",
    "locations": {
        "Bekaa Valley": {"solar": "High",   "wind": "Low",    "humidity": "Medium"},
        "Beirut Coast": {"solar": "Medium", "wind": "High",   "humidity": "High"},
        "Tripoli":      {"solar": "High",   "wind": "Medium", "humidity": "High"},
        "South Lebanon":{"solar": "High",   "wind": "Medium", "humidity": "Medium"},
        "Baabda":       {"solar": "Medium", "wind": "Low",    "humidity": "Medium"},
        "Zahle":        {"solar": "High",   "wind": "Low",    "humidity": "Low"},
    },
    "kpis": {"pv": 82, "comfort": 74, "water": 31},
}
DEFAULT_ALERTS:  Dict[str, Any] = {"items": []}
DEFAULT_UPDATES: Dict[str, Any] = {"items": []}


# ── storage helpers ───────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def load_json(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
    except Exception as exc:
        log.warning("Failed to load %s (%s) — using defaults", path.name, exc)
    return json.loads(json.dumps(fallback))

def save_json(path: Path, data: Dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def _migrate_ids(items: List[Dict[str, Any]]) -> bool:
    changed = False
    for item in items:
        if "id" not in item:
            item["id"] = str(uuid.uuid4())
            changed = True
    return changed

def ensure_storage() -> None:
    if not SETTINGS_PATH.exists():
        save_json(SETTINGS_PATH, DEFAULT_SETTINGS)
        log.info("Created default settings.json")
    for path, default in [(ALERTS_PATH, DEFAULT_ALERTS), (UPDATES_PATH, DEFAULT_UPDATES)]:
        if not path.exists():
            save_json(path, default)
            log.info("Created %s", path.name)
        else:
            data = load_json(path, default)
            if _migrate_ids(data.get("items", [])):
                save_json(path, data)
                log.info("Migrated IDs in %s", path.name)


# ── AI model setup ────────────────────────────────────────────────────────────
# These are loaded once at startup. If the model files are missing,
# the AI endpoints will return 503 instead of crashing the whole server.

_ai_ready   = False
_ai_error   = ""
_lock       = threading.Lock()
_payload: Dict[str, Any] = {}
_ws_list: List[WebSocket] = []
_dli        = None
_alpha      = 0.7
_get_full_payload  = None
_compare_treatments = None
_CROP_THRESHOLDS    = None

def _load_ai_model():
    global _ai_ready, _ai_error, _dli, _get_full_payload, _compare_treatments, _CROP_THRESHOLDS
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from inference  import get_full_payload, compare_treatments
        from dli_engine import DLIEngine, CROP_THRESHOLDS
        _get_full_payload   = get_full_payload
        _compare_treatments = compare_treatments
        _CROP_THRESHOLDS    = CROP_THRESHOLDS
        _dli = DLIEngine(crop="lettuce")
        # Restore DLI from previous session if same day
        try:
            dli_state = load_json(DLI_STATE_PATH, {})
            today = datetime.now().strftime("%Y-%m-%d")
            if dli_state.get("date") == today and dli_state.get("accumulated", 0) > 0:
                _dli.accumulated = float(dli_state["accumulated"])
                saved_crop = dli_state.get("crop", "lettuce")
                _dli.set_crop(saved_crop)
                log.info("Restored DLI: %.2f mol/m² for %s", _dli.accumulated, saved_crop)
        except Exception as re:
            log.warning("DLI restore failed: %s", re)
        _ai_ready = True
        log.info("AI model loaded successfully — BiLSTM PV R²=0.9085 PAR R²=0.8926")
    except Exception as e:
        _ai_error = str(e)
        log.warning("AI model not loaded: %s", e)

# Beirut coordinates
BEIRUT_LAT = 33.8938
BEIRUT_LON = 35.5018
OWM_API_KEY = "783c93efcb81b26150fb9368a37ce1cb"

def _fetch_owm_data() -> dict:
    """Fetch current weather from OpenWeatherMap for Beirut."""
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?lat={BEIRUT_LAT}&lon={BEIRUT_LON}&appid={OWM_API_KEY}&units=metric")
    import urllib.request
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.loads(r.read().decode())

def _cloud_to_ghi(cloud_pct: float, hour: float) -> float:
    """Estimate GHI from cloud cover % and hour of day."""
    # Clear-sky GHI based on solar angle
    if hour < 6 or hour > 20:
        return 0.0
    solar_angle = np.pi * (hour - 6) / 14
    clear_sky = max(0.0, 950 * np.sin(solar_angle))
    # Cloud cover reduces GHI
    clear_factor = 1.0 - (cloud_pct / 100.0) * 0.75
    return clear_sky * clear_factor

def _get_sensor_window() -> "pd.DataFrame":
    """Build sensor window from real OpenWeatherMap data for Beirut."""
    now = datetime.utcnow()
    beirut_hour = (now.hour + 3) % 24  # UTC+3

    # Try to fetch real weather data
    try:
        wx = _fetch_owm_data()
        clouds    = wx.get("clouds", {}).get("all", 50)       # cloud cover %
        temp_c    = wx.get("main", {}).get("temp", 22.0)      # air temp °C
        humidity  = wx.get("main", {}).get("humidity", 60)    # humidity %
        wind_spd  = wx.get("wind", {}).get("speed", 3.0)      # wind m/s
        # OWM free tier doesn't give direct GHI — derive from clouds
        current_ghi = _cloud_to_ghi(clouds, beirut_hour)
        log.info("[OWM] Beirut: clouds=%d%% temp=%.1f°C GHI_est=%.0fW/m²",
                 clouds, temp_c, current_ghi)
        use_real = True
    except Exception as e:
        log.warning("[OWM] Failed to fetch weather: %s — falling back to synthetic", e)
        clouds = 30; temp_c = 22.0; current_ghi = _cloud_to_ghi(30, beirut_hour)
        use_real = False

    # Build 24-step window (last 2 hours at 5-min intervals)
    n = 24
    # Each step is 5 min back from now
    hours = [(beirut_hour * 60 + now.minute - (n - 1 - i) * 5) / 60 % 24
             for i in range(n)]

    np.random.seed(int(now.timestamp()) % 999)
    noise_scale = 15 if use_real else 25

    ghi = []
    for i, h in enumerate(hours):
        base = _cloud_to_ghi(clouds, h)
        # Most recent value uses actual current conditions
        if i == n - 1:
            val = current_ghi + np.random.normal(0, noise_scale * 0.3)
        else:
            val = base + np.random.normal(0, noise_scale)
        ghi.append(max(0.0, val))

    dhi  = [max(0.0, g * (0.15 + clouds/100*0.1) + np.random.normal(0, 5)) for g in ghi]
    par  = [max(0.0, g * 1.8 + np.random.normal(0, 20)) for g in ghi]
    alb  = [max(0.0, g * 0.12) for g in ghi]
    temp = [max(5.0, temp_c - (ghi[-1] - g) / 200 + np.random.normal(0, 0.3)) for g in ghi]

    df = pd.DataFrame({
        "GHI (W.m-2)":        ghi,
        "DHI_SPN1 (W.m-2)":   dhi,
        "PAR (umol.s-1.m-2)": par,
        "Albedometer (W.m-2)": alb,
        "airtemp_underpanel":  temp,
        "sin_hour": [np.sin(2 * np.pi * h / 24) for h in hours],
        "cos_hour": [np.cos(2 * np.pi * h / 24) for h in hours],
    })
    df["GHI (W.m-2)_lag1"]         = df["GHI (W.m-2)"].shift(1).bfill()
    df["GHI (W.m-2)_lag2"]         = df["GHI (W.m-2)"].shift(2).bfill()
    df["GHI (W.m-2)_roll6"]        = df["GHI (W.m-2)"].rolling(6, min_periods=1).mean()
    df["DHI_SPN1 (W.m-2)_lag1"]    = df["DHI_SPN1 (W.m-2)"].shift(1).bfill()
    df["PAR (umol.s-1.m-2)_lag1"]  = df["PAR (umol.s-1.m-2)"].shift(1).bfill()
    df["PAR (umol.s-1.m-2)_roll6"] = df["PAR (umol.s-1.m-2)"].rolling(6, min_periods=1).mean()
    return df

def _run_inference():
    global _payload
    if not _ai_ready:
        return
    try:
        window = _get_sensor_window()
        result = _get_full_payload(window, _dli,
                                   current_hour=datetime.now().hour,
                                   alpha=_alpha)
        result["timestamp"] = datetime.now().isoformat()
        with _lock:
            _payload = result

        # Persist DLI state so it survives restarts
        try:
            save_json(DLI_STATE_PATH, {
                "accumulated": _dli.accumulated,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "crop": _dli.crop,
            })
        except Exception as de:
            log.warning("DLI state save failed: %s", de)

        # Save to history
        try:
            hist = load_json(HISTORY_PATH, {"records": []})
            record = {
                "timestamp":      result["timestamp"],
                "pv_peak_kw":     result["pv_peak_kw"],
                "pv_total_kwh":   result["pv_total_kwh"],
                "par_mean":       result["par_mean"],
                "dli_accumulated": result["dli_accumulated"],
                "dli_pct":        result["dli_pct"],
                "stress_alert":   result["stress_alert"],
                "irrigation_pct": result["irrigation_pct"],
                "recommended_config": result["recommended_config"],
            }
            records = hist.get("records", [])
            records.append(record)
            hist["records"] = records[-MAX_HISTORY:]
            save_json(HISTORY_PATH, hist)
        except Exception as he:
            log.warning("History save failed: %s", he)

        log.info("[AI] PV=%.1fkW PAR=%.0f DLI=%.2f(%d%%) Stress=%s Irr=%d%% Config=%s",
                 result["pv_peak_kw"], result["par_mean"],
                 result["dli_accumulated"], result["dli_pct"],
                 result["stress_alert"], result["irrigation_pct"],
                 result["recommended_config"])
    except Exception as e:
        log.error("[AI] inference error: %s", e)

def _inference_loop(interval_min: int = 30):
    while True:
        _run_inference()
        time.sleep(interval_min * 60)


# ── pydantic models ───────────────────────────────────────────────────────────

Level     = Literal["Low", "Medium", "High"]
AlertType = Literal["Info", "Warning", "Critical"]
CropName  = Literal["lettuce", "spinach", "wheat", "tomato", "cucumber", "pepper", "custom"]

class LocationModel(BaseModel):
    solar: Level; wind: Level; humidity: Level

class KpiModel(BaseModel):
    pv: int = Field(ge=0, le=100)
    comfort: int = Field(ge=0, le=100)
    water: int = Field(ge=0, le=100)

class AlertIn(BaseModel):
    type: AlertType
    message: str = Field(..., min_length=1, max_length=1000)

class UpdateIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body:  str = Field(..., min_length=1, max_length=2000)

class AlphaBody(BaseModel):
    alpha: float


# ── app ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage()
    _load_ai_model()
    if _ai_ready:
        _run_inference()  # first result immediately
        t = threading.Thread(target=_inference_loop, args=(30,), daemon=True)
        t.start()
        log.info("AI inference loop started (every 30 min)")
    log.info("Farm Dashboard Backend started")
    yield
    log.info("Farm Dashboard Backend shutting down")

app = FastAPI(
    title="Farm Dashboard Backend",
    version="3.0.0",
    description="Agrivoltaic farm management + BiLSTM AI forecast API",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── base routes ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Farm Dashboard Backend",
        "version": "3.0.0",
        "ai_ready": _ai_ready,
        "ai_error": _ai_error if not _ai_ready else None,
    }

@app.get("/health")
@app.head("/health")
def health():
    return {"ok": True, "timestamp": now_iso(), "ai_ready": _ai_ready}


# ── locations ─────────────────────────────────────────────────────────────────

@app.get("/locations")
def list_locations():
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    return {"locations": settings.get("locations", {}),
            "default_location": settings.get("default_location", "")}

@app.put("/locations/{name}")
def upsert_location(name: str, loc: LocationModel):
    name = name.strip()
    if not name: raise HTTPException(422, "Location name cannot be empty")
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    settings.setdefault("locations", {})[name] = loc.model_dump()
    if not settings.get("default_location"): settings["default_location"] = name
    save_json(SETTINGS_PATH, settings)
    return {"saved": True, "name": name}

@app.delete("/locations/{name}")
def delete_location(name: str):
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    locations = settings.get("locations", {})
    if name not in locations: raise HTTPException(404, "Location not found")
    locations.pop(name)
    settings["locations"] = locations
    if settings.get("default_location") == name:
        settings["default_location"] = next(iter(locations), "")
    save_json(SETTINGS_PATH, settings)
    return {"deleted": True}

@app.put("/default_location/{name}")
def set_default_location(name: str):
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    if name not in settings.get("locations", {}): raise HTTPException(404, "Location not found")
    settings["default_location"] = name
    save_json(SETTINGS_PATH, settings)
    return {"saved": True}


# ── kpis ──────────────────────────────────────────────────────────────────────

@app.get("/kpis")
def get_kpis():
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    return settings.get("kpis", DEFAULT_SETTINGS["kpis"])

@app.put("/kpis")
def put_kpis(kpis: KpiModel):
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    settings["kpis"] = kpis.model_dump()
    save_json(SETTINGS_PATH, settings)
    return {"saved": True}


# ── alerts ────────────────────────────────────────────────────────────────────

@app.get("/alerts")
def get_alerts():
    return load_json(ALERTS_PATH, DEFAULT_ALERTS)

@app.post("/alerts", status_code=201)
def add_alert(alert: AlertIn):
    data = load_json(ALERTS_PATH, DEFAULT_ALERTS)
    items: List[Dict[str, Any]] = data.get("items", [])
    record = {"id": str(uuid.uuid4()), "type": alert.type,
              "message": alert.message, "created_at": now_iso()}
    items.insert(0, record)
    data["items"] = items[:MAX_ITEMS]
    save_json(ALERTS_PATH, data)
    return {"saved": True, "id": record["id"], "count": len(data["items"])}

@app.delete("/alerts/{alert_id}")
def remove_alert(alert_id: str):
    data = load_json(ALERTS_PATH, DEFAULT_ALERTS)
    items = data.get("items", [])
    new_items = [a for a in items if a.get("id") != alert_id]
    if len(new_items) == len(items): raise HTTPException(404, "Alert not found")
    data["items"] = new_items
    save_json(ALERTS_PATH, data)
    return {"deleted": True, "count": len(new_items)}


# ── updates ───────────────────────────────────────────────────────────────────

@app.get("/updates")
def get_updates():
    return load_json(UPDATES_PATH, DEFAULT_UPDATES)

@app.post("/updates", status_code=201)
def add_update(update: UpdateIn):
    data = load_json(UPDATES_PATH, DEFAULT_UPDATES)
    items: List[Dict[str, Any]] = data.get("items", [])
    record = {"id": str(uuid.uuid4()), "title": update.title,
              "body": update.body, "created_at": now_iso()}
    items.insert(0, record)
    data["items"] = items[:MAX_ITEMS]
    save_json(UPDATES_PATH, data)
    return {"saved": True, "id": record["id"], "count": len(data["items"])}

@app.delete("/updates/{update_id}")
def remove_update(update_id: str):
    data = load_json(UPDATES_PATH, DEFAULT_UPDATES)
    items = data.get("items", [])
    new_items = [u for u in items if u.get("id") != update_id]
    if len(new_items) == len(items): raise HTTPException(404, "Update not found")
    data["items"] = new_items
    save_json(UPDATES_PATH, data)
    return {"deleted": True, "count": len(new_items)}


# ── AI forecast endpoints ─────────────────────────────────────────────────────

def _require_ai():
    if not _ai_ready:
        raise HTTPException(503, f"AI model not loaded: {_ai_error}")

@app.get("/forecast")
def get_forecast():
    """Latest BiLSTM prediction — all 5 dashboard panels. Refreshes every 30 min."""
    _require_ai()
    with _lock:
        p = dict(_payload)
    if not p:
        raise HTTPException(503, "Model warming up — retry in 5 seconds")
    return p

@app.get("/ai/status")
def ai_status():
    """Check if the AI model is loaded and working."""
    crop = _dli.crop if _dli else "lettuce"
    threshold = _CROP_THRESHOLDS.get(crop, 14.0) if _CROP_THRESHOLDS else 14.0
    return {"ai_ready": _ai_ready, "ai_error": _ai_error if not _ai_ready else None,
            "has_forecast": bool(_payload), "alpha": _alpha,
            "crop": crop, "dli_threshold": threshold}

@app.get("/dli/status")
def dli_status():
    """Current DLI accumulator state."""
    _require_ai()
    return _dli.get_status()

@app.get("/treatment/compare")
def treatment_compare(alpha: float = 0.7):
    """Live Fixed-tilt vs Vertical comparison with custom alpha."""
    _require_ai()
    if not 0 <= alpha <= 1: raise HTTPException(400, "alpha must be 0.0–1.0")
    return _compare_treatments(_get_sensor_window(), alpha=alpha)

@app.post("/treatment/alpha")
def set_alpha(body: AlphaBody):
    """Change crop vs energy priority (0=energy only, 1=crop only, default 0.7)."""
    global _alpha
    _require_ai()
    if not 0 <= body.alpha <= 1: raise HTTPException(400, "alpha must be 0.0–1.0")
    _alpha = body.alpha
    return {"alpha": _alpha,
            "meaning": f"Crop {int(_alpha*100)}%  Energy {int((1-_alpha)*100)}%"}

@app.post("/crop/{name}")
def set_crop(name: str):
    """Change active crop — adjusts DLI threshold immediately."""
    _require_ai()
    if name not in _CROP_THRESHOLDS:
        raise HTTPException(400, f"Options: {list(_CROP_THRESHOLDS.keys())}")
    _dli.set_crop(name)
    return {"crop": name, "threshold": f"{_CROP_THRESHOLDS[name]} mol/m²/day"}

@app.get("/history")
def get_history():
    """Last 48 forecast records (24 hours) for history charts."""
    hist = load_json(HISTORY_PATH, {"records": []})
    return hist

@app.websocket("/live")
async def ws_live(ws: WebSocket):
    """WebSocket push — receive every new prediction automatically."""
    await ws.accept()
    _ws_list.append(ws)
    with _lock:
        p = dict(_payload)
    if p:
        await ws.send_json(p)
    try:
        while True:
            await asyncio.sleep(30)
            with _lock:
                p = dict(_payload)
            if p:
                await ws.send_json(p)
    except WebSocketDisconnect:
        if ws in _ws_list:
            _ws_list.remove(ws)


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
