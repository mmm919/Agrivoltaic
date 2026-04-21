from pathlib import Path
import json
import uuid
import os
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal

from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger("farm_api")

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
SETTINGS_PATH = STORAGE_DIR / "settings.json"
ALERTS_PATH = STORAGE_DIR / "alerts.json"
UPDATES_PATH = STORAGE_DIR / "updates.json"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

MAX_ITEMS = 100

DEFAULT_SETTINGS: Dict[str, Any] = {
    "default_location": "Bekaa Valley",
    "locations": {
        "Bekaa Valley": {"solar": "High", "wind": "Low", "humidity": "Medium"},
        "Beirut Coast": {"solar": "Medium", "wind": "High", "humidity": "High"},
        "Tripoli": {"solar": "High", "wind": "Medium", "humidity": "High"},
        "South Lebanon": {"solar": "High", "wind": "Medium", "humidity": "Medium"},
        "Baabda": {"solar": "Medium", "wind": "Low", "humidity": "Medium"},
        "Zahle": {"solar": "High", "wind": "Low", "humidity": "Low"},
    },
    "kpis": {"pv": 82, "comfort": 74, "water": 31},
}

DEFAULT_ALERTS: Dict[str, Any] = {"items": []}
DEFAULT_UPDATES: Dict[str, Any] = {"items": []}


# ── helpers ──────────────────────────────────────────────────────────────────

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
    return json.loads(json.dumps(fallback))  # deep copy of fallback


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomic write: temp file → rename so a crash never leaves a corrupt file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _migrate_ids(items: List[Dict[str, Any]]) -> bool:
    """Add 'id' to any item that lacks one. Returns True if changes were made."""
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


# ── models ───────────────────────────────────────────────────────────────────

Level = Literal["Low", "Medium", "High"]
AlertType = Literal["Info", "Warning", "Critical"]


class LocationModel(BaseModel):
    solar: Level
    wind: Level
    humidity: Level


class KpiModel(BaseModel):
    pv: int = Field(ge=0, le=100)
    comfort: int = Field(ge=0, le=100)
    water: int = Field(ge=0, le=100)


class AlertIn(BaseModel):
    type: AlertType
    message: str = Field(..., min_length=1, max_length=1000)


class UpdateIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=2000)


# ── app ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage()
    log.info("Farm Dashboard Backend started")
    yield
    log.info("Farm Dashboard Backend shutting down")


app = FastAPI(
    title="Farm Dashboard Backend",
    version="2.0.0",
    description="Agrivoltaic farm management API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Farm Dashboard Backend",
        "version": "2.0.0",
        "endpoints": ["/health", "/locations", "/default_location/{name}", "/kpis", "/alerts", "/updates"],
    }


@app.get("/health")
def health():
    return {"ok": True, "timestamp": now_iso()}


# ── locations ─────────────────────────────────────────────────────────────────

@app.get("/locations")
def list_locations():
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    return {
        "locations": settings.get("locations", {}),
        "default_location": settings.get("default_location", ""),
    }


@app.put("/locations/{name}")
def upsert_location(name: str, loc: LocationModel):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Location name cannot be empty")
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    locations = settings.setdefault("locations", {})
    locations[name] = loc.model_dump()
    if not settings.get("default_location"):
        settings["default_location"] = name
    save_json(SETTINGS_PATH, settings)
    log.info("Upserted location '%s'", name)
    return {"saved": True, "name": name}


@app.delete("/locations/{name}")
def delete_location(name: str):
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    locations = settings.get("locations", {})
    if name not in locations:
        raise HTTPException(status_code=404, detail="Location not found")
    locations.pop(name)
    settings["locations"] = locations
    if settings.get("default_location") == name:
        settings["default_location"] = next(iter(locations), "")
    save_json(SETTINGS_PATH, settings)
    log.info("Deleted location '%s'", name)
    return {"deleted": True}


@app.put("/default_location/{name}")
def set_default_location(name: str):
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    if name not in settings.get("locations", {}):
        raise HTTPException(status_code=404, detail="Location not found")
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
    record: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "type": alert.type,
        "message": alert.message,
        "created_at": now_iso(),
    }
    items.insert(0, record)
    data["items"] = items[:MAX_ITEMS]
    save_json(ALERTS_PATH, data)
    log.info("Alert added: [%s] %s", alert.type, alert.message[:60])
    return {"saved": True, "id": record["id"], "count": len(data["items"])}


@app.delete("/alerts/{alert_id}")
def remove_alert(alert_id: str):
    data = load_json(ALERTS_PATH, DEFAULT_ALERTS)
    items: List[Dict[str, Any]] = data.get("items", [])
    new_items = [a for a in items if a.get("id") != alert_id]
    if len(new_items) == len(items):
        raise HTTPException(status_code=404, detail="Alert not found")
    data["items"] = new_items
    save_json(ALERTS_PATH, data)
    log.info("Alert deleted: %s", alert_id)
    return {"deleted": True, "count": len(new_items)}


# ── updates ───────────────────────────────────────────────────────────────────

@app.get("/updates")
def get_updates():
    return load_json(UPDATES_PATH, DEFAULT_UPDATES)


@app.post("/updates", status_code=201)
def add_update(update: UpdateIn):
    data = load_json(UPDATES_PATH, DEFAULT_UPDATES)
    items: List[Dict[str, Any]] = data.get("items", [])
    record: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "title": update.title,
        "body": update.body,
        "created_at": now_iso(),
    }
    items.insert(0, record)
    data["items"] = items[:MAX_ITEMS]
    save_json(UPDATES_PATH, data)
    log.info("Update added: '%s'", update.title[:60])
    return {"saved": True, "id": record["id"], "count": len(data["items"])}


@app.delete("/updates/{update_id}")
def remove_update(update_id: str):
    data = load_json(UPDATES_PATH, DEFAULT_UPDATES)
    items: List[Dict[str, Any]] = data.get("items", [])
    new_items = [u for u in items if u.get("id") != update_id]
    if len(new_items) == len(items):
        raise HTTPException(status_code=404, detail="Update not found")
    data["items"] = new_items
    save_json(UPDATES_PATH, data)
    log.info("Update deleted: %s", update_id)
    return {"deleted": True, "count": len(new_items)}


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
