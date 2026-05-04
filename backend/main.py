@app.get("/ai/status")
def ai_status():
"""Check if the AI model is loaded and working."""
    crop = _dli.crop if _dli else "lettuce"
    threshold = _CROP_THRESHOLDS.get(crop, 14.0) if _CROP_THRESHOLDS else 14.0
return {"ai_ready": _ai_ready, "ai_error": _ai_error if not _ai_ready else None,
"has_forecast": bool(_payload), "alpha": _alpha,
            "crop": _dli.crop if _dli else None}
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
