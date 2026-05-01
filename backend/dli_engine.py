"""
dli_engine.py
Daily Light Integral accumulator + irrigation automation.

How to use:
    from dli_engine import DLIEngine

    engine = DLIEngine(crop="lettuce")           # create once at startup

    # call every 30 minutes with the PAR forecast from the model:
    status = engine.update(par_forecast, current_hour=14)

    # status contains everything the app needs for panels 2, 3, and 5.

Science references:
    DLI formula:    Faust & Logan (2018) HortScience 53(9)
    Crop targets:   Faust & Logan (2018), Gent (2003)
    Irrigation:     Jarvis (1976) stomatal conductance theory
"""
import numpy as np
from datetime import date
from typing import Optional, List

# ── Published crop minimum DLI requirements ────────────────────────────────────
CROP_THRESHOLDS = {
    "lettuce":  14.0,   # mol photons / m² / day
    "spinach":  16.0,
    "wheat":    22.0,
    "tomato":   25.0,
    "cucumber": 20.0,
    "pepper":   18.0,
    "custom":   17.0,
}

DAYLIGHT_HOURS = 14     # approximate growing-season daylight window (06:00–20:00)


class DLIEngine:
    """
    Tracks the crop's daily light budget.

    Every 30 minutes:
      1. Receives the 12 PAR forecast values from the BiLSTM
      2. Converts mean PAR to a mol/m² deposit  →  PAR × 300s × 1e-6
      3. Adds deposit to today's running total
      4. Projects end-of-day total from current rate
      5. If projected total < crop threshold  →  stress = True
      6. Computes irrigation factor proportional to deficit (Jarvis 1976):
             factor = 1 − (deficit / threshold × 0.5)
             minimum = 0.60  (never cuts more than 40%)
      7. Resets automatically at midnight

    Why irrigation reduces when DLI is low:
        Less light → less photosynthesis → stomata partially close →
        less water lost through leaves → less water needed through roots.
        Reducing irrigation is the biologically correct response.
    """

    def __init__(self, crop: str = "lettuce"):
        """
        crop: one of lettuce, spinach, wheat, tomato, cucumber, pepper, custom
        """
        if crop not in CROP_THRESHOLDS:
            raise ValueError(
                f"Unknown crop '{crop}'. "
                f"Options: {list(CROP_THRESHOLDS.keys())}"
            )
        self.crop         = crop
        self.threshold    = CROP_THRESHOLDS[crop]
        self._accumulated = 0.0
        self._n_updates   = 0
        self._today: Optional[date] = None
        self._stress_log  = []

    def update(self, par_forecast: List[float], current_hour: int) -> dict:
        """
        Call every 30 minutes with the BiLSTM's PAR forecast.

        Parameters
        ----------
        par_forecast  : list of 12 PAR values from the model (μmol/s/m²)
        current_hour  : current hour of day (0-23)

        Returns
        -------
        dict — complete DLI + irrigation status for panels 2, 3, and 5
        """
        # Auto-reset at start of new day
        self._check_reset()

        # ── DLI deposit ────────────────────────────────────────────────────
        # Formula (Faust & Logan 2018):
        #   DLI (mol/m²) = PAR (μmol/s/m²) × time (s) × 10⁻⁶
        # We use the mean of the 12 forecast values × 300 seconds (5 min)
        mean_par = float(np.mean(par_forecast))
        deposit  = mean_par * 300 * 1e-6
        self._accumulated += deposit
        self._n_updates   += 1

        # ── Project end-of-day DLI ─────────────────────────────────────────
        hours_elapsed = max(current_hour - 6, 0.5)   # since sunrise at 06:00
        rate          = self._accumulated / hours_elapsed
        hours_left    = max(0.0, (6 + DAYLIGHT_HOURS) - current_hour)
        projected     = self._accumulated + rate * hours_left

        # ── Stress decision ────────────────────────────────────────────────
        deficit  = max(0.0, self.threshold - projected)
        dli_pct  = min(100.0, self._accumulated / self.threshold * 100)
        stressed = projected < self.threshold

        # ── Irrigation factor (Jarvis 1976) ────────────────────────────────
        # A deficit of 50% of the target → 25% irrigation reduction
        # Maximum reduction is 40% (factor never below 0.60)
        if stressed:
            factor = round(max(0.60, 1.0 - deficit / self.threshold * 0.5), 3)
        else:
            factor = 1.0

        # ── Alert message ──────────────────────────────────────────────────
        if stressed:
            msg = (
                f"Crop light stress detected. "
                f"Projected DLI {projected:.1f} mol/m² is below "
                f"{self.crop} target {self.threshold:.0f} mol/m²/day "
                f"(deficit {deficit:.1f} mol/m²). "
                f"Irrigation reduced to {int(factor*100)}% of normal."
            )
        else:
            msg = (
                f"Crop light adequate — {dli_pct:.0f}% of daily target met. "
                f"Projected by sunset: {projected:.1f} mol/m². "
                f"Irrigation at 100%."
            )

        return {
            # Panel 2 — DLI gauge
            "dli_accumulated":    round(self._accumulated, 3),
            "dli_deposit":        round(deposit, 5),
            "dli_projected_eod":  round(projected, 2),
            "dli_threshold":      self.threshold,
            "dli_pct":            round(dli_pct, 1),
            "dli_deficit":        round(deficit, 3),
            "crop":               self.crop,
            # Panel 3 — Stress alert
            "stress_alert":       stressed,
            "alert_message":      msg,
            # Panel 5 — Irrigation
            "irrigation_factor":  factor,
            "irrigation_pct":     int(factor * 100),
            # Metadata
            "current_hour":       current_hour,
            "n_cycles_today":     self._n_updates,
        }

    def get_status(self) -> dict:
        """Return current state without updating — safe to call anytime."""
        return {
            "dli_accumulated": round(self._accumulated, 3),
            "dli_threshold":   self.threshold,
            "dli_pct":         round(min(100, self._accumulated / self.threshold * 100), 1),
            "crop":            self.crop,
            "stress_history":  self._stress_log,
        }

    def set_crop(self, crop: str):
        """Change crop type at runtime (e.g. seasonal rotation)."""
        if crop not in CROP_THRESHOLDS:
            raise ValueError(f"Unknown crop '{crop}'")
        self.crop      = crop
        self.threshold = CROP_THRESHOLDS[crop]

    def _check_reset(self):
        today = date.today()
        if self._today is None or today != self._today:
            if self._today is not None:
                self._stress_log.append({
                    "date":       str(self._today),
                    "final_dli":  round(self._accumulated, 3),
                    "threshold":  self.threshold,
                    "met_target": self._accumulated >= self.threshold,
                })
            self._accumulated = 0.0
            self._n_updates   = 0
            self._today       = today
