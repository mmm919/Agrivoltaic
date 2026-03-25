from pathlib import Path
import json
from typing import Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
SETTINGS_PATH = STORAGE_DIR / "settings.json"
ALERTS_PATH = STORAGE_DIR / "alerts.json"
UPDATES_PATH = STORAGE_DIR / "updates.json"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS = {
    "default_location": "Bekaa Valley",
    "locations": {
        "Bekaa Valley": {"solar": "High", "wind": "Low", "humidity": "Medium"},
        "Beirut Coast": {"solar": "Medium", "wind": "High", "humidity": "High"},
        "Tripoli": {"solar": "High", "wind": "Medium", "humidity": "High"},
        "South Lebanon": {"solar": "High", "wind": "Medium", "humidity": "Medium"},
    },
    "kpis": {"pv": 82, "comfort": 74, "water": 31},
}

DEFAULT_ALERTS = {"items": []}
DEFAULT_UPDATES = {"items": []}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return fallback


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ensure_storage():
    if not SETTINGS_PATH.exists():
        save_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    if not ALERTS_PATH.exists():
        save_json(ALERTS_PATH, DEFAULT_ALERTS)
    if not UPDATES_PATH.exists():
        save_json(UPDATES_PATH, DEFAULT_UPDATES)


class LocationModel(BaseModel):
    solar: str = Field(..., description="Low Medium High")
    wind: str = Field(..., description="Low Medium High")
    humidity: str = Field(..., description="Low Medium High")


class KpiModel(BaseModel):
    pv: int = Field(ge=0, le=100)
    comfort: int = Field(ge=0, le=100)
    water: int = Field(ge=0, le=100)


class AlertIn(BaseModel):
    type: str = Field(..., description="Info Warning Critical")
    message: str


class UpdateIn(BaseModel):
    title: str
    body: str


app = FastAPI(title="Farm Dashboard Backend", version="1.2.0")


@app.on_event("startup")
def startup():
    ensure_storage()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/locations")
def list_locations():
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    return {
        "locations": settings.get("locations", {}),
        "default_location": settings.get("default_location", ""),
    }


@app.put("/locations/{name}")
def upsert_location(name: str, loc: LocationModel):
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    locations = settings.get("locations", {})
    locations[name] = loc.model_dump()
    settings["locations"] = locations
    if not settings.get("default_location"):
        settings["default_location"] = name
    save_json(SETTINGS_PATH, settings)
    return {"saved": True}


@app.delete("/locations/{name}")
def delete_location(name: str):
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    locations = settings.get("locations", {})
    if name not in locations:
        raise HTTPException(status_code=404, detail="Location not found")
    locations.pop(name)
    settings["locations"] = locations
    if settings.get("default_location") == name:
        settings["default_location"] = next(iter(locations.keys()), "")
    save_json(SETTINGS_PATH, settings)
    return {"deleted": True}


@app.put("/default_location/{name}")
def set_default_location(name: str):
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    if name not in settings.get("locations", {}):
        raise HTTPException(status_code=404, detail="Location not found")
    settings["default_location"] = name
    save_json(SETTINGS_PATH, settings)
    return {"saved": True}


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


@app.get("/alerts")
def get_alerts():
    return load_json(ALERTS_PATH, DEFAULT_ALERTS)


@app.post("/alerts")
def add_alert(alert: AlertIn):
    data = load_json(ALERTS_PATH, DEFAULT_ALERTS)
    items: List[Dict[str, Any]] = data.get("items", [])
    record = {"type": alert.type, "message": alert.message, "created_at": now_iso()}
    items.insert(0, record)
    data["items"] = items
    save_json(ALERTS_PATH, data)
    return {"saved": True, "count": len(items)}


@app.delete("/alerts/{index}")
def remove_alert(index: int):
    data = load_json(ALERTS_PATH, DEFAULT_ALERTS)
    items: List[Dict[str, Any]] = data.get("items", [])
    if index < 0 or index >= len(items):
        raise HTTPException(status_code=404, detail="Alert index out of range")
    items.pop(index)
    data["items"] = items
    save_json(ALERTS_PATH, data)
    return {"deleted": True, "count": len(items)}


@app.get("/updates")
def get_updates():
    return load_json(UPDATES_PATH, DEFAULT_UPDATES)


@app.post("/updates")
def add_update(update: UpdateIn):
    data = load_json(UPDATES_PATH, DEFAULT_UPDATES)
    items: List[Dict[str, Any]] = data.get("items", [])
    record = {"title": update.title, "body": update.body, "created_at": now_iso()}
    items.insert(0, record)
    data["items"] = items
    save_json(UPDATES_PATH, data)
    return {"saved": True, "count": len(items)}


@app.delete("/updates/{index}")
def remove_update(index: int):
    data = load_json(UPDATES_PATH, DEFAULT_UPDATES)
    items: List[Dict[str, Any]] = data.get("items", [])
    if index < 0 or index >= len(items):
        raise HTTPException(status_code=404, detail="Update index out of range")
    items.pop(index)
    data["items"] = items
    save_json(UPDATES_PATH, data)
    return {"deleted": True, "count": len(items)}
