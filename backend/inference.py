"""
inference.py
Three functions your friend calls:

    predict(window_df, treatment)         → PV + PAR 1-hour forecast
    compare_treatments(window_df, alpha)  → which panel config is better
    get_full_payload(window_df, dli)      → everything in one dict for the app

SETUP:
    Place in same folder as: model.py  model_weights.pt  scaler_X.pkl  scaler_y.pkl
    pip install torch scikit-learn joblib pandas numpy
"""
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from model import BiLSTMModel, SCALABLE_FEATURES, LOOKBACK, HORIZON

# ── Load once at startup ────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[inference] Loading BiLSTM on {DEVICE}...")
_model = BiLSTMModel().to(DEVICE)
_model.load_state_dict(torch.load(BASE / "model_weights.pt", map_location=DEVICE))
_model.eval()

_scaler_X = joblib.load(BASE / "scaler_X.pkl")
_scaler_y = joblib.load(BASE / "scaler_y.pkl")
print(f"[inference] Ready — PV R²=0.9085  PAR R²=0.8926")


# ── Internal helpers ────────────────────────────────────────────────────────────
def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 4 engineered features from raw sensor columns."""
    df = df.copy()
    df["GHI_diff1"]  = df["GHI (W.m-2)"].diff().fillna(0)
    df["GHI_diff2"]  = df["GHI (W.m-2)"].diff(2).fillna(0)
    df["PAR_diff1"]  = df["PAR (umol.s-1.m-2)_lag1"].diff().fillna(0)
    df["cloud_flag"] = (df["DHI_SPN1 (W.m-2)"] /
                        df["GHI (W.m-2)"].clip(lower=1)).clip(0, 1)
    return df


def _build_tensor(df: pd.DataFrame, treatment: str) -> torch.Tensor:
    """
    Convert 24-row sensor DataFrame to model input tensor (1, 24, 17).
    treatment: "Fixedtilt" → 0   |   "Vertical" → 1
    """
    if len(df) != LOOKBACK:
        raise ValueError(f"Need exactly {LOOKBACK} rows (= 2 hours at 5-min intervals). Got {len(df)}.")

    df = _add_derived(df)

    scaled    = _scaler_X.transform(df[SCALABLE_FEATURES]).astype(np.float32)
    treat_val = 1.0 if treatment.lower() in ("vertical", "1") else 0.0
    treat_col = np.full((LOOKBACK, 1), treat_val, dtype=np.float32)
    X         = np.hstack([scaled, treat_col])          # (24, 17)
    return torch.tensor(X).unsqueeze(0).to(DEVICE)       # (1, 24, 17)


def _decode(pv_s: np.ndarray, par_s: np.ndarray):
    """Inverse-scale model outputs back to kW and μmol/s/m²."""
    dummy = np.stack([pv_s.ravel(), par_s.ravel()], axis=1)
    inv   = _scaler_y.inverse_transform(dummy)
    return inv[:, 0].clip(0).tolist(), inv[:, 1].clip(0).tolist()


# ── PUBLIC FUNCTION 1 ── predict ───────────────────────────────────────────────
def predict(window_df: pd.DataFrame, treatment: str = "Fixedtilt") -> dict:
    """
    Run the BiLSTM for ONE configuration. Returns 1-hour forecasts.

    Parameters
    ----------
    window_df : pd.DataFrame
        Last 24 rows of sensor data (2 hours, 5-min intervals).
        Required columns: GHI (W.m-2), GHI (W.m-2)_lag1, GHI (W.m-2)_lag2,
        GHI (W.m-2)_roll6, DHI_SPN1 (W.m-2), DHI_SPN1 (W.m-2)_lag1,
        PAR (umol.s-1.m-2)_lag1, PAR (umol.s-1.m-2)_roll6,
        Albedometer (W.m-2), airtemp_underpanel, sin_hour, cos_hour.
        (GHI_diff1, GHI_diff2, PAR_diff1, cloud_flag computed automatically.)

    treatment : str  →  "Fixedtilt"  or  "Vertical"

    Returns
    -------
    dict:
        pv_forecast_kw   list[12]  PV power next 60 min (kW), one per 5-min slot
        par_forecast     list[12]  Crop light next 60 min (μmol/s/m²)
        pv_peak_kw       float     Max predicted PV in the next hour
        pv_total_kwh     float     Total energy (kWh) over the next hour
        par_mean         float     Mean PAR over the next hour
        dli_deposit      float     mol/m² this window adds to today's DLI total
        treatment        str       Which config was used
    """
    X = _build_tensor(window_df, treatment)
    with torch.no_grad():
        pv_s, par_s = _model(X)

    pv, par = _decode(pv_s.cpu().numpy(), par_s.cpu().numpy())

    par_mean    = float(np.mean(par))
    dli_deposit = par_mean * 300 * 1e-6   # mol/m² for this 30-min window

    return {
        "pv_forecast_kw": [round(v, 2) for v in pv],
        "par_forecast":   [round(v, 1) for v in par],
        "pv_peak_kw":     round(max(pv), 2),
        "pv_total_kwh":   round(sum(pv) * 5 / 60, 3),
        "par_mean":       round(par_mean, 1),
        "dli_deposit":    round(dli_deposit, 5),
        "treatment":      treatment,
    }


# ── PUBLIC FUNCTION 2 ── compare_treatments ────────────────────────────────────
def compare_treatments(window_df: pd.DataFrame, alpha: float = 0.7) -> dict:
    """
    Run the model TWICE — Fixed-tilt and Vertical — and recommend the better one.

    The recommendation uses a weighted score that balances BOTH crop light and energy:
        score = alpha × (PAR / PAR_max)  +  (1-alpha) × (PV / PV_max)

    alpha = crop priority weight (default 0.7)
        alpha = 1.0  →  optimise purely for crop light (PAR)
        alpha = 0.0  →  optimise purely for energy (PV)
        alpha = 0.7  →  default — crop light matters more than energy
                        (Dupraz et al. 2011: food security > energy yield in APV)

    Why not use PAR alone?
        Fixed-tilt almost always generates more PV. If we compared only PV,
        the answer would always be Fixed-tilt. If we compared only PAR, we
        would ignore the energy tradeoff that is the whole point of APV.
        The weighted score lets the farmer adjust priorities by crop stage,
        electricity price, or weather conditions.

    Returns
    -------
    dict:
        recommended_config   "Fixedtilt" or "Vertical"
        alpha_crop_priority  the alpha value used
        fixed_score          weighted score for Fixed-tilt (0-1)
        vertical_score       weighted score for Vertical (0-1)
        fixed_par_mean       predicted mean PAR for Fixed-tilt (μmol/s/m²)
        vertical_par_mean    predicted mean PAR for Vertical (μmol/s/m²)
        fixed_pv_kwh         predicted energy for Fixed-tilt (kWh)
        vertical_pv_kwh      predicted energy for Vertical (kWh)
        par_advantage_pct    how much better winner is on PAR (%)
        pv_advantage_pct     how much better winner is on PV (%)
        reason               human-readable explanation string
        fixed_forecast       full predict() result for Fixed-tilt
        vertical_forecast    full predict() result for Vertical
    """
    fixed = predict(window_df, "Fixedtilt")
    vert  = predict(window_df, "Vertical")

    # Normalise to 0-1 so the two metrics are comparable
    par_max = max(fixed["par_mean"], vert["par_mean"], 1e-9)
    pv_max  = max(fixed["pv_total_kwh"], vert["pv_total_kwh"], 1e-9)

    score_fixed = alpha * fixed["par_mean"] / par_max + (1-alpha) * fixed["pv_total_kwh"] / pv_max
    score_vert  = alpha * vert["par_mean"]  / par_max + (1-alpha) * vert["pv_total_kwh"]  / pv_max

    if score_vert >= score_fixed:
        winner, loser  = "Vertical",  "Fixedtilt"
        win_pred, lose_pred = vert, fixed
    else:
        winner, loser  = "Fixedtilt", "Vertical"
        win_pred, lose_pred = fixed, vert

    par_adv = round((win_pred["par_mean"] - lose_pred["par_mean"])
                    / max(lose_pred["par_mean"], 1e-9) * 100, 1)
    pv_adv  = round((win_pred["pv_total_kwh"] - lose_pred["pv_total_kwh"])
                    / max(lose_pred["pv_total_kwh"], 1e-9) * 100, 1)

    return {
        "recommended_config":  winner,
        "alpha_crop_priority": alpha,
        "fixed_score":         round(score_fixed, 4),
        "vertical_score":      round(score_vert,  4),
        "fixed_par_mean":      fixed["par_mean"],
        "vertical_par_mean":   vert["par_mean"],
        "fixed_pv_kwh":        fixed["pv_total_kwh"],
        "vertical_pv_kwh":     vert["pv_total_kwh"],
        "par_advantage_pct":   par_adv,
        "pv_advantage_pct":    pv_adv,
        "reason": (
            f"{winner} recommended (α={alpha}): "
            f"PAR {'+' if par_adv>=0 else ''}{par_adv}% | "
            f"PV  {'+' if pv_adv>=0 else ''}{pv_adv}%"
        ),
        "fixed_forecast":    fixed,
        "vertical_forecast": vert,
    }


# ── PUBLIC FUNCTION 3 ── get_full_payload ──────────────────────────────────────
def get_full_payload(window_df: pd.DataFrame,
                     dli_engine,
                     current_hour: int,
                     alpha: float = 0.7) -> dict:
    """
    Run everything in one call. Returns the complete JSON for all 5 app panels.

    Parameters
    ----------
    window_df    : last 24 sensor rows
    dli_engine   : instance of DLIEngine from dli_engine.py
    current_hour : current hour of day (0-23)
    alpha        : crop priority weight (default 0.7)
    """
    # 1. Run model twice → treatment recommendation
    comp = compare_treatments(window_df, alpha=alpha)

    # 2. Use the recommended config's forecast as primary
    primary = (comp["fixed_forecast"]
               if comp["recommended_config"] == "Fixedtilt"
               else comp["vertical_forecast"])

    # 3. Update DLI engine with the PAR forecast
    dli = dli_engine.update(primary["par_forecast"], current_hour)

    return {
        # ── Panel 1: Dual chart ────────────────────────────────────────────
        "pv_forecast_kw":      primary["pv_forecast_kw"],   # list[12] kW
        "par_forecast":        primary["par_forecast"],      # list[12] μmol/s/m²
        "pv_peak_kw":          primary["pv_peak_kw"],
        "pv_total_kwh":        primary["pv_total_kwh"],
        "par_mean":            primary["par_mean"],

        # ── Panel 2: DLI gauge ─────────────────────────────────────────────
        "dli_accumulated":     dli["dli_accumulated"],      # mol/m² today so far
        "dli_projected_eod":   dli["dli_projected_eod"],    # projected by sunset
        "dli_threshold":       dli["dli_threshold"],        # crop minimum
        "dli_pct":             dli["dli_pct"],              # 0-100 %
        "dli_deficit":         dli["dli_deficit"],          # shortfall mol/m²
        "crop":                dli["crop"],

        # ── Panel 3: Stress alert ──────────────────────────────────────────
        "stress_alert":        dli["stress_alert"],         # True / False
        "alert_message":       dli["alert_message"],        # human text

        # ── Panel 4: Configuration recommendation ─────────────────────────
        "recommended_config":  comp["recommended_config"],  # "Fixedtilt" or "Vertical"
        "fixed_par_mean":      comp["fixed_par_mean"],
        "vertical_par_mean":   comp["vertical_par_mean"],
        "fixed_pv_kwh":        comp["fixed_pv_kwh"],
        "vertical_pv_kwh":     comp["vertical_pv_kwh"],
        "par_advantage_pct":   comp["par_advantage_pct"],
        "pv_advantage_pct":    comp["pv_advantage_pct"],
        "recommendation_reason": comp["reason"],
        "alpha_crop_priority": comp["alpha_crop_priority"],

        # ── Panel 5: Irrigation ────────────────────────────────────────────
        "irrigation_factor":   dli["irrigation_factor"],   # 0.60 – 1.00
        "irrigation_pct":      dli["irrigation_pct"],      # 60 – 100 %
    }
