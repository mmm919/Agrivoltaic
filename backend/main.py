from pathlib import Path
import json
import uuid
import os
import tempfile
import logging
import threading
import time
import asyncio
import hashlib
import secrets
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
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
MAX_HISTORY = 48

# ── Supabase config ───────────────────────────────────────────────────────────
SUPABASE_URL = "https://rnontnsjhnlabjenzymh.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJub250bnNqaG5sYWJqZW56eW1oIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc4OTA3NzksImV4cCI6MjA5MzQ2Njc3OX0.NG2K4x3CyAAb41Pf5BuVdGEGs9qckTHKwcyb2z16ce8"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJub250bnNqaG5sYWJqZW56eW1oIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3Nzg5MDc3OSwiZXhwIjoyMDkzNDY2Nzc5fQ.cvc95Nloq1p2vkLoXQUHydp7yypuNla7vIWeLIb59ro"

SUPABASE_AUTH_URL = f"{SUPABASE_URL}/auth/v1"
SUPABASE_REST_URL = f"{SUPABASE_URL}/rest/v1"

def supabase_headers(use_service_key=False):
    key = SUPABASE_SERVICE_KEY if use_service_key else SUPABASE_ANON_KEY
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def user_headers(token: str):
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

# ── Auth helpers ──────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(401, "Not authenticated")
    token = credentials.credentials
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_AUTH_URL}/user",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"}
        )
    if r.status_code != 200:
        raise HTTPException(401, "Invalid or expired token")
    return r.json()

# ── Supabase DB helpers ───────────────────────────────────────────────────────
async def get_user_dli_state(user_id: str):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_REST_URL}/user_dli_state",
            headers=supabase_headers(use_service_key=True),
            params={"user_id": f"eq.{user_id}", "limit": "1"}
        )
    if r.status_code == 200 and r.json():
        return r.json()[0]
    return None

async def upsert_user_dli_state(user_id: str, data: dict):
    payload = {"user_id": user_id, **data, "updated_at": datetime.utcnow().isoformat()}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_REST_URL}/user_dli_state",
            headers={**supabase_headers(use_service_key=True), "Prefer": "resolution=merge-duplicates,return=representation"},
            json=payload
        )
    return r.status_code in (200, 201)

async def get_user_history(user_id: str, limit: int = 48):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_REST_URL}/user_history",
            headers=supabase_headers(use_service_key=True),
            params={"user_id": f"eq.{user_id}", "order": "created_at.desc", "limit": str(limit)}
        )
    if r.status_code == 200:
        return list(reversed(r.json()))
    return []

async def append_user_history(user_id: str, record: dict):
    payload = {"user_id": user_id, **record, "created_at": datetime.utcnow().isoformat()}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_REST_URL}/user_history",
            headers=supabase_headers(use_service_key=True),
            json=payload
        )
    return r.status_code in (200, 201)

async def get_user_settings(user_id: str):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_REST_URL}/user_settings",
            headers=supabase_headers(use_service_key=True),
            params={"user_id": f"eq.{user_id}", "limit": "1"}
        )
    if r.status_code == 200 and r.json():
        return r.json()[0]
    return {"crop": "lettuce", "alpha": 0.7}

async def upsert_user_settings(user_id: str, data: dict):
    payload = {"user_id": user_id, **data, "updated_at": datetime.utcnow().isoformat()}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_REST_URL}/user_settings",
            headers={**supabase_headers(use_service_key=True), "Prefer": "resolution=merge-duplicates,return=representation"},
            json=payload
        )
    return r.status_code in (200, 201)

# ── Storage helpers ───────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def load_json(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
    except Exception as exc:
        log.warning("Failed to load %s (%s) - using defaults", path.name, exc)
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

def _migrate_ids(items):
    changed = False
    for item in items:
        if "id" not in item:
            item["id"] = str(uuid.uuid4())
            changed = True
    return changed

DEFAULT_SETTINGS: Dict[str, Any] = {
    "default_location": "Bekaa Valley",
    "locations": {
        "Bekaa Valley": {"solar": "High", "wind": "Low", "humidity": "Medium"},
        "Beirut Coast": {"solar": "Medium", "wind": "High", "humidity": "High"},
        "Tripoli":      {"solar": "High", "wind": "Medium", "humidity": "High"},
        "South Lebanon":{"solar": "High", "wind": "Medium", "humidity": "Medium"},
        "Baabda":       {"solar": "Medium", "wind": "Low", "humidity": "Medium"},
        "Zahle":        {"solar": "High", "wind": "Low", "humidity": "Low"},
    },
    "kpis": {"pv": 82, "comfort": 74, "water": 31},
}
DEFAULT_ALERTS:  Dict[str, Any] = {"items": []}
DEFAULT_UPDATES: Dict[str, Any] = {"items": []}

def ensure_storage():
    if not SETTINGS_PATH.exists():
        save_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    for path, default in [(ALERTS_PATH, DEFAULT_ALERTS), (UPDATES_PATH, DEFAULT_UPDATES)]:
        if not path.exists():
            save_json(path, default)
        else:
            data = load_json(path, default)
            if _migrate_ids(data.get("items", [])):
                save_json(path, data)

# ── AI model setup ────────────────────────────────────────────────────────────
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
        try:
            dli_state = load_json(DLI_STATE_PATH, {})
            today = datetime.now().strftime("%Y-%m-%d")
            if dli_state.get("date") == today and dli_state.get("accumulated", 0) > 0:
                _dli.accumulated = float(dli_state["accumulated"])
                saved_crop = dli_state.get("crop", "lettuce")
                _dli.set_crop(saved_crop)
                log.info("Restored DLI: %.2f mol/m2 for %s", _dli.accumulated, saved_crop)
        except Exception as re:
            log.warning("DLI restore failed: %s", re)
        _ai_ready = True
        log.info("AI model loaded successfully")
    except Exception as e:
        _ai_error = str(e)
        log.warning("AI model not loaded: %s", e)

BEIRUT_LAT = 33.8938
BEIRUT_LON = 35.5018
OWM_API_KEY = "783c93efcb81b26150fb9368a37ce1cb"

def _fetch_owm_data() -> dict:
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?lat={BEIRUT_LAT}&lon={BEIRUT_LON}&appid={OWM_API_KEY}&units=metric")
    import urllib.request
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.loads(r.read().decode())

def _cloud_to_ghi(cloud_pct: float, hour: float) -> float:
    if hour < 6 or hour > 20:
        return 0.0
    solar_angle = np.pi * (hour - 6) / 14
    clear_sky = max(0.0, 950 * np.sin(solar_angle))
    clear_factor = 1.0 - (cloud_pct / 100.0) * 0.75
    return clear_sky * clear_factor

def _get_sensor_window() -> "pd.DataFrame":
    now = datetime.utcnow()
    beirut_hour = (now.hour + 3) % 24
    try:
        wx = _fetch_owm_data()
        clouds    = wx.get("clouds", {}).get("all", 50)
        temp_c    = wx.get("main", {}).get("temp", 22.0)
        current_ghi = _cloud_to_ghi(clouds, beirut_hour)
        log.info("[OWM] Beirut: clouds=%d%% temp=%.1fC GHI_est=%.0fW/m2", clouds, temp_c, current_ghi)
        use_real = True
    except Exception as e:
        log.warning("[OWM] Failed: %s - falling back to synthetic", e)
        clouds = 30; temp_c = 22.0; current_ghi = _cloud_to_ghi(30, beirut_hour)
        use_real = False

    n = 24
    hours = [(beirut_hour * 60 + now.minute - (n - 1 - i) * 5) / 60 % 24 for i in range(n)]
    np.random.seed(int(now.timestamp()) % 999)
    noise_scale = 15 if use_real else 25

    ghi = []
    for i, h in enumerate(hours):
        base = _cloud_to_ghi(clouds, h)
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
        result = _get_full_payload(window, _dli, current_hour=datetime.now().hour, alpha=_alpha)
        result["timestamp"] = datetime.now().isoformat()
        with _lock:
            _payload = result

        try:
            save_json(DLI_STATE_PATH, {
                "accumulated": _dli.accumulated,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "crop": _dli.crop,
            })
        except Exception as de:
            log.warning("DLI state save failed: %s", de)

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

        log.info("[AI] PV=%.1fkW PAR=%.0f DLI=%.2f(%d%%) Stress=%s Irr=%d%%",
                 result["pv_peak_kw"], result["par_mean"],
                 result["dli_accumulated"], result["dli_pct"],
                 result["stress_alert"], result["irrigation_pct"])
    except Exception as e:
        log.error("[AI] inference error: %s", e)

def _inference_loop(interval_min: int = 30):
    while True:
        _run_inference()
        time.sleep(interval_min * 60)


# ── Pydantic models ───────────────────────────────────────────────────────────
Level     = Literal["Low", "Medium", "High"]
AlertType = Literal["Info", "Warning", "Critical"]

class RegisterBody(BaseModel):
    email: str
    password: str = Field(..., min_length=6)

class LoginBody(BaseModel):
    email: str
    password: str

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

class UserCropBody(BaseModel):
    crop: str
    alpha: float = 0.7


# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage()
    _load_ai_model()
    if _ai_ready:
        _run_inference()
        t = threading.Thread(target=_inference_loop, args=(30,), daemon=True)
        t.start()
        log.info("AI inference loop started (every 30 min)")
    log.info("Farm Dashboard Backend started")
    yield
    log.info("Farm Dashboard Backend shutting down")

app = FastAPI(title="Farm Dashboard Backend", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/register", status_code=201)
async def register(body: RegisterBody):
    """Register a new user with email and password via Supabase Auth."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_AUTH_URL}/signup",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": body.email, "password": body.password}
        )
    data = r.json()
    if r.status_code not in (200, 201) or "error" in data:
        msg = data.get("error_description") or data.get("msg") or "Registration failed"
        raise HTTPException(400, msg)
    return {"message": "Registration successful. Please check your email to confirm your account."}

@app.post("/auth/login")
async def login(body: LoginBody):
    """Login with email and password. Returns access token."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_AUTH_URL}/token?grant_type=password",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": body.email, "password": body.password}
        )
    data = r.json()
    if r.status_code != 200 or "access_token" not in data:
        msg = data.get("error_description") or data.get("msg") or "Invalid email or password"
        raise HTTPException(401, msg)
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "user": {"id": data["user"]["id"], "email": data["user"]["email"]},
    }

@app.post("/auth/logout")
async def logout(user=Depends(get_current_user)):
    """Logout current user."""
    return {"message": "Logged out successfully"}

@app.get("/auth/me")
async def me(user=Depends(get_current_user)):
    """Get current user info."""
    return {"id": user["id"], "email": user["email"]}


# ── User-specific endpoints ───────────────────────────────────────────────────
@app.get("/user/settings")
async def get_my_settings(user=Depends(get_current_user)):
    """Get this user's crop and alpha settings."""
    return await get_user_settings(user["id"])

@app.post("/user/settings")
async def save_my_settings(body: UserCropBody, user=Depends(get_current_user)):
    """Save this user's crop and alpha settings."""
    global _alpha, _dli
    if _dli and body.crop in (_CROP_THRESHOLDS or {}):
        _dli.set_crop(body.crop)
    _alpha = body.alpha
    ok = await upsert_user_settings(user["id"], {"crop": body.crop, "alpha": body.alpha})
    return {"saved": ok, "crop": body.crop, "alpha": body.alpha}

@app.get("/user/history")
async def get_my_history(user=Depends(get_current_user)):
    """Get this user's personal inference history (last 48 records)."""
    records = await get_user_history(user["id"])
    return {"records": records}

@app.get("/user/dli")
async def get_my_dli(user=Depends(get_current_user)):
    """Get this user's DLI state."""
    state = await get_user_dli_state(user["id"])
    if state:
        return state
    return {"accumulated": 0.0, "date": datetime.now().strftime("%Y-%m-%d"), "crop": "lettuce"}


# ── Base routes ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Farm Dashboard Backend", "version": "4.0.0", "ai_ready": _ai_ready}

@app.get("/health")
@app.head("/health")
def health():
    return {"ok": True, "timestamp": now_iso(), "ai_ready": _ai_ready}


# ── AI forecast endpoints ─────────────────────────────────────────────────────
def _require_ai():
    if not _ai_ready:
        raise HTTPException(503, f"AI model not loaded: {_ai_error}")

@app.get("/forecast")
def get_forecast():
    _require_ai()
    with _lock:
        p = dict(_payload)
    if not p:
        raise HTTPException(503, "Model warming up - retry in 5 seconds")
    return p

@app.get("/ai/status")
def ai_status():
    crop = _dli.crop if _dli else "lettuce"
    threshold = _CROP_THRESHOLDS.get(crop, 14.0) if _CROP_THRESHOLDS else 14.0
    return {"ai_ready": _ai_ready, "ai_error": _ai_error if not _ai_ready else None,
            "has_forecast": bool(_payload), "alpha": _alpha,
            "crop": crop, "dli_threshold": threshold}

@app.get("/treatment/compare")
def treatment_compare(alpha: float = 0.7):
    _require_ai()
    if not 0 <= alpha <= 1: raise HTTPException(400, "alpha must be 0.0-1.0")
    return _compare_treatments(_get_sensor_window(), alpha=alpha)

@app.post("/treatment/alpha")
def set_alpha(body: AlphaBody):
    global _alpha
    _require_ai()
    if not 0 <= body.alpha <= 1: raise HTTPException(400, "alpha must be 0.0-1.0")
    _alpha = body.alpha
    return {"alpha": _alpha}

@app.post("/crop/{name}")
def set_crop(name: str):
    _require_ai()
    if name not in _CROP_THRESHOLDS:
        raise HTTPException(400, f"Options: {list(_CROP_THRESHOLDS.keys())}")
    _dli.set_crop(name)
    return {"crop": name, "threshold": f"{_CROP_THRESHOLDS[name]} mol/m2/day"}

@app.get("/history")
def get_history():
    hist = load_json(HISTORY_PATH, {"records": []})
    return hist

@app.websocket("/live")
async def ws_live(ws: WebSocket):
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

# ── Locations, alerts, updates (unchanged) ────────────────────────────────────
@app.get("/locations")
def list_locations():
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    return {"locations": settings.get("locations", {}), "default_location": settings.get("default_location", "")}

@app.put("/locations/{name}")
def upsert_location(name: str, loc: LocationModel):
    name = name.strip()
    if not name: raise HTTPException(422, "Location name cannot be empty")
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    settings.setdefault("locations", {})[name] = loc.model_dump()
    if not settings.get("default_location"): settings["default_location"] = name
    save_json(SETTINGS_PATH, settings)
    return {"saved": True, "name": name}

@app.get("/kpis")
def get_kpis():
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    return settings.get("kpis", DEFAULT_SETTINGS["kpis"])

@app.get("/alerts")
def get_alerts():
    return load_json(ALERTS_PATH, DEFAULT_ALERTS)

@app.get("/updates")
def get_updates():
    return load_json(UPDATES_PATH, DEFAULT_UPDATES)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
