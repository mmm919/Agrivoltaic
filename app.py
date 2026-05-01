from pathlib import Path
from typing import Any, Dict, List
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = "https://agrivoltaic.onrender.com"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:       #f5f4f0;
  --surface:  #ffffff;
  --surface2: #f0efe9;
  --border:   #e2e0d8;
  --text:     #1a1916;
  --muted:    #6b6860;
  --muted2:   #9e9b94;
  --accent:   #2d6a4f;
  --radius:   14px;
  --shadow:   0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
}

* { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1160px; background: var(--bg); }
header[data-testid="stHeader"], div[data-testid="stToolbar"], #MainMenu, footer { display: none !important; visibility: hidden !important; }
.stApp { background: var(--bg); }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px 22px; box-shadow: var(--shadow); }
.card-tinted { background: #f0f7f4; border: 1px solid #c3ddd3; border-radius: var(--radius); padding: 20px 22px; }
.card-warn { background: #fef9f0; border: 1px solid #f5d89a; border-radius: var(--radius); padding: 20px 22px; }
.card-danger { background: #fdf2f2; border: 1px solid #f5c0bc; border-radius: var(--radius); padding: 20px 22px; }

.page-header { padding: 22px 0 18px 0; border-bottom: 1px solid var(--border); margin-bottom: 22px; }
.page-title { font-size: 22px; font-weight: 700; color: var(--text); letter-spacing: -0.3px; }
.page-sub { font-size: 13px; color: var(--muted); margin-top: 3px; }

.label { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
.value-lg { font-size: 28px; font-weight: 700; color: var(--text); letter-spacing: -0.5px; line-height: 1.1; }
.value-md { font-size: 20px; font-weight: 600; color: var(--text); }
.value-sm { font-size: 14px; font-weight: 500; color: var(--text); }
.caption { font-size: 12px; color: var(--muted2); margin-top: 3px; }

.badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 999px; }
.badge-green { background: #dcfce7; color: #15803d; }
.badge-amber { background: #fef3c7; color: #92400e; }
.badge-red   { background: #fee2e2; color: #991b1b; }
.badge-rec   { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }

.progress-track { background: var(--border); border-radius: 999px; height: 6px; margin-top: 8px; overflow: hidden; }
.progress-fill  { height: 6px; border-radius: 999px; }

.stat-box { flex: 1; min-width: 110px; background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.divider { height: 1px; background: var(--border); margin: 16px 0; }

.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 2px solid var(--border) !important; gap: 0 !important; padding: 0 !important; }
.stTabs [data-baseweb="tab"] { font-weight: 600 !important; font-size: 14px !important; color: var(--muted) !important; padding: 12px 24px !important; border-bottom: 2px solid transparent !important; margin-bottom: -2px !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 22px !important; }

div[data-testid="stMetric"] { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: 12px 16px !important; }
div[data-testid="stMetricValue"] { color: var(--text) !important; }
div[data-testid="stMetricLabel"] { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#4b5563", family="DM Sans", size=11),
    margin=dict(l=8, r=8, t=8, b=8),
    xaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

def api_get(path, timeout=12):
    r = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(path, payload=None, timeout=10):
    r = requests.post(f"{BACKEND_URL}{path}", json=payload or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def clamp(x, lo, hi): return max(lo, min(hi, x))

def soil_factor(soil):
    return {"Dry": 0.0, "Medium": 0.5, "Wet": 1.0}.get(soil, 0.5)

def simulate_scenario(params):
    ph = float(params["panel_height_m"]); ps = float(params["panel_spacing_m"])
    tilt = float(params["tilt_deg"]); ch = float(params["canopy_height_m"])
    lai = float(params["lai"]); sf = soil_factor(params["soil_wetness"])
    tracking = bool(params["single_axis_tracking"])
    pv_gain  = 12.0 if tracking else 0.0
    pv_gain += 6.0 * math.exp(-((tilt - 25.0) / 18.0) ** 2)
    pv_gain -= 0.6 * max(0.0, ph - 3.0)
    pv_gain += 0.35 * (ps - 2.0)
    pv_gain  = clamp(pv_gain, 0.0, 20.0)
    shade = clamp((ph-1.0)/3.0, 0.0, 1.0); vent = clamp((ps-1.5)/3.0, 0.0, 1.0); lai_e = clamp(lai/4.0, 0.0, 1.0)
    lc = 25.0*(0.35*shade+0.35*sf+0.30*lai_e)*(0.75+0.25*vent); lc = clamp(lc, 0.0, 25.0)
    ws = clamp(10.0+35.0*(0.40*shade+0.40*sf+0.20*lai_e), 0.0, 50.0)
    hi = clamp(0.5*lc+4.0*sf, 0.0, 20.0)
    cs = clamp(45.0+2.0*lc+0.35*ws-0.6*abs(ch-1.4)*10.0, 0.0, 100.0)
    return {"pv_performance": clamp(65.0+pv_gain,0.0,100.0), "crop_comfort": cs,
            "water_savings_kpi": clamp(ws*2.0,0.0,100.0), "leaf_cooling_c": lc,
            "pv_gain_percent": pv_gain, "water_savings_percent": ws,
            "heat_index_reduction_c": hi, "comfort_score": cs}

def pct_color(v):
    if v >= 70: return "#15803d"
    if v >= 40: return "#92400e"
    return "#991b1b"

def fetch_forecast():
    try:
        return api_get("/forecast", timeout=15), None
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 503:
            return None, "warming_up"
        return None, str(e)
    except Exception as e:
        return None, str(e)

def show_backend_error(err):
    if err == "warming_up":
        st.warning("⏳ AI model warming up — refresh in ~30 seconds.")
    else:
        st.error("Backend not reachable.")
        st.caption(str(err))

# ── PAGE 1: OVERVIEW ─────────────────────────────────────────────────────────
def page_overview():
    st.markdown('<div class="page-header"><div class="page-title">🌱 Agrivoltaic Farm Overview</div><div class="page-sub">Live AI forecast · synthetic demo data</div></div>', unsafe_allow_html=True)

    d, err = fetch_forecast()
    if err: show_backend_error(err); return

    crop = d.get("crop","lettuce").capitalize()
    stressed = d.get("stress_alert", False)
    dli_pct  = d.get("dli_pct", 0)
    irr_pct  = d.get("irrigation_pct", 100)
    rec_cfg  = d.get("recommended_config","—")
    ts       = d.get("timestamp","")

    k1,k2,k3,k4 = st.columns(4, gap="medium")
    with k1:
        st.markdown(f'<div class="card"><div class="label">PV Peak Power</div><div class="value-lg">{d.get("pv_peak_kw",0):.1f} <span style="font-size:14px;color:var(--muted)">kW</span></div><div class="caption">Next hour forecast</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="card"><div class="label">Crop Light (PAR)</div><div class="value-lg">{d.get("par_mean",0):.0f} <span style="font-size:14px;color:var(--muted)">μmol/s/m²</span></div><div class="caption">Mean over next hour</div></div>', unsafe_allow_html=True)
    with k3:
        badge = "badge-green" if not stressed else "badge-red"
        status = "On track" if not stressed else "Stress detected"
        gc = "#15803d" if not stressed else "#c0392b"
        st.markdown(f'<div class="card"><div class="label">DLI Progress — {crop}</div><div class="value-lg">{dli_pct:.0f}<span style="font-size:14px;color:var(--muted)">%</span></div><div class="progress-track"><div class="progress-fill" style="width:{min(dli_pct,100):.0f}%;background:{gc}"></div></div><div style="margin-top:6px"><span class="badge {badge}">{status}</span></div></div>', unsafe_allow_html=True)
    with k4:
        ic = "#15803d" if irr_pct >= 90 else ("#d97706" if irr_pct >= 70 else "#c0392b")
        st.markdown(f'<div class="card"><div class="label">Irrigation Schedule</div><div class="value-lg" style="color:{ic}">{irr_pct}<span style="font-size:14px;color:var(--muted)">%</span></div><div class="progress-track"><div class="progress-fill" style="width:{irr_pct}%;background:{ic}"></div></div><div class="caption">of normal schedule</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col, rec_col = st.columns([1.6, 1.0], gap="large")

    with chart_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label" style="margin-bottom:12px">PV POWER + CROP LIGHT — NEXT 60 MINUTES</div>', unsafe_allow_html=True)
        pv_vals = d.get("pv_forecast_kw",[]); par_vals = d.get("par_forecast",[])
        mins = [f"+{(i+1)*5}m" for i in range(len(pv_vals))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mins, y=pv_vals, name="PV power (kW)", line=dict(color="#2d6a4f",width=2.5), fill="tozeroy", fillcolor="rgba(45,106,79,0.07)", yaxis="y1"))
        fig.add_trace(go.Scatter(x=mins, y=par_vals, name="PAR (μmol/s/m²)", line=dict(color="#b7791f",width=2,dash="dot"), yaxis="y2"))
        fig.update_layout(**CHART_LAYOUT, yaxis=dict(title="kW",gridcolor="#e5e7eb"), yaxis2=dict(title="μmol/s/m²",overlaying="y",side="right",gridcolor="rgba(0,0,0,0)"), legend=dict(orientation="h",y=-0.22,bgcolor="rgba(0,0,0,0)"), height=230, margin=dict(l=8,r=8,t=8,b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rec_col:
        fixed_par=d.get("fixed_par_mean",0); vert_par=d.get("vertical_par_mean",0)
        fixed_pv=d.get("fixed_pv_kwh",0); vert_pv=d.get("vertical_pv_kwh",0)
        reason=d.get("recommendation_reason","")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label" style="margin-bottom:12px">PANEL CONFIGURATION</div>', unsafe_allow_html=True)
        for cfg, par_v, pv_v in [("Fixed-tilt",fixed_par,fixed_pv),("Vertical",vert_par,vert_pv)]:
            is_rec = cfg == rec_cfg
            border = "2px solid #2d6a4f" if is_rec else "1px solid var(--border)"
            bg = "#f0f7f4" if is_rec else "var(--surface2)"
            rec_b = '<span class="badge badge-rec" style="margin-left:6px;font-size:10px">Recommended</span>' if is_rec else ""
            st.markdown(f'<div style="border:{border};background:{bg};border-radius:10px;padding:12px 14px;margin-bottom:8px"><div style="font-weight:600;font-size:13px;margin-bottom:6px">{cfg}{rec_b}</div><div style="display:flex;gap:20px;"><div><div class="label">PAR</div><div class="value-sm">{par_v:.0f}</div></div><div><div class="label">Energy</div><div class="value-sm">{pv_v:.1f} kWh</div></div></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="caption" style="margin-top:4px">{reason}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sc, dc = st.columns(2, gap="large")
    with sc:
        cls = "card-danger" if stressed else "card-tinted"
        icon = "⚠️" if stressed else "✅"
        msg = d.get("alert_message","")
        st.markdown(f'<div class="{cls}"><div class="label">{icon} CROP STRESS ALERT</div><div style="font-size:13px;color:var(--text);line-height:1.6;margin-top:6px">{msg}</div><div style="display:flex;gap:24px;margin-top:12px"><div><div class="label">Deficit</div><div class="value-md">{d.get("dli_deficit",0):.1f} <span style="font-size:12px">mol/m²</span></div></div><div><div class="label">DLI progress</div><div class="value-md">{dli_pct:.0f}<span style="font-size:12px">%</span></div></div></div></div>', unsafe_allow_html=True)
    with dc:
        gc2 = "#2d6a4f" if not stressed else "#c0392b"
        st.markdown(f'<div class="card"><div class="label">DAILY LIGHT INTEGRAL</div><div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap"><div><div class="label">Collected</div><div class="value-md">{d.get("dli_accumulated",0):.1f} <span style="font-size:12px">mol/m²</span></div></div><div><div class="label">Projected EOD</div><div class="value-md">{d.get("dli_projected_eod",0):.1f} <span style="font-size:12px">mol/m²</span></div></div><div><div class="label">Target ({crop})</div><div class="value-md">{d.get("dli_threshold",14):.0f} <span style="font-size:12px">mol/m²/day</span></div></div></div><div class="progress-track" style="margin-top:14px;height:8px"><div class="progress-fill" style="width:{min(dli_pct,100):.0f}%;height:8px;background:{gc2}"></div></div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="caption" style="margin-top:8px;text-align:right">Last updated: {ts[11:16] if len(ts)>15 else ts} · Refreshes every 30 min</div>', unsafe_allow_html=True)
    if st.button("🔄 Refresh", key="ov_refresh"): st.rerun()


# ── PAGE 2: AI FORECAST ───────────────────────────────────────────────────────
def page_forecast():
    st.markdown('<div class="page-header"><div class="page-title">🤖 AI Forecast</div><div class="page-sub">BiLSTM model — PV R²=0.9085 · PAR R²=0.8926</div></div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([1,1,1], gap="large")
    with ctrl1:
        crop_opts = ["lettuce","spinach","wheat","tomato","cucumber","pepper"]
        crop_sel  = st.selectbox("🌿 Crop type", crop_opts, key="fc_crop")
    with ctrl2:
        alpha_sel = st.slider("⚖️ Crop vs energy (α)", 0.0, 1.0, 0.7, 0.05, key="fc_alpha")
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Apply settings", key="fc_apply"):
            try:
                api_post(f"/crop/{crop_sel}"); api_post("/treatment/alpha", {"alpha": alpha_sel})
                st.success("Applied")
            except Exception as e: st.error(str(e))

    st.markdown("<br>", unsafe_allow_html=True)
    d, err = fetch_forecast()
    if err: show_backend_error(err); return

    pv_vals = d.get("pv_forecast_kw",[]); par_vals = d.get("par_forecast",[])
    mins = [f"+{(i+1)*5}m" for i in range(len(pv_vals))]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label" style="margin-bottom:12px">BILSTM FORECAST — PV POWER & CROP LIGHT (NEXT 60 MIN)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mins, y=pv_vals, name="PV power (kW)", line=dict(color="#2d6a4f",width=2.5), fill="tozeroy", fillcolor="rgba(45,106,79,0.08)", yaxis="y1", mode="lines+markers", marker=dict(size=5,color="#2d6a4f")))
    fig.add_trace(go.Scatter(x=mins, y=par_vals, name="Crop light PAR (μmol/s/m²)", line=dict(color="#b7791f",width=2.5), yaxis="y2", mode="lines+markers", marker=dict(size=5,color="#b7791f")))
    fig.update_layout(**CHART_LAYOUT, yaxis=dict(title="PV power (kW)",gridcolor="#e5e7eb"), yaxis2=dict(title="PAR (μmol/s/m²)",overlaying="y",side="right",gridcolor="rgba(0,0,0,0)"), legend=dict(orientation="h",y=-0.2,bgcolor="rgba(0,0,0,0)"), height=280, margin=dict(l=8,r=8,t=8,b=50))
    st.plotly_chart(fig, use_container_width=True)
    s1,s2,s3 = st.columns(3)
    s1.metric("PV peak", f'{d.get("pv_peak_kw",0):.1f} kW')
    s2.metric("Energy next hour", f'{d.get("pv_total_kwh",0):.1f} kWh')
    s3.metric("Crop light mean", f'{d.get("par_mean",0):.0f} μmol/s/m²')
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    rec_cfg=d.get("recommended_config","—"); fixed_par=d.get("fixed_par_mean",0); vert_par=d.get("vertical_par_mean",0)
    fixed_pv=d.get("fixed_pv_kwh",0); vert_pv=d.get("vertical_pv_kwh",0); reason=d.get("recommendation_reason",""); alpha_val=d.get("alpha_crop_priority",0.7)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label" style="margin-bottom:14px">FIXED-TILT VS VERTICAL — TREATMENT COMPARISON</div>', unsafe_allow_html=True)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="Fixed-tilt", x=["PAR (μmol/s/m²)","PV Energy (kWh×10)"], y=[fixed_par,fixed_pv*10], marker_color="#6b7280", opacity=0.85))
    fig_comp.add_trace(go.Bar(name="Vertical",   x=["PAR (μmol/s/m²)","PV Energy (kWh×10)"], y=[vert_par, vert_pv*10],  marker_color="#2d6a4f", opacity=0.85))
    fig_comp.update_layout(**CHART_LAYOUT, barmode="group", height=220, margin=dict(l=8,r=8,t=8,b=8))
    st.plotly_chart(fig_comp, use_container_width=True)
    c1,c2 = st.columns(2, gap="large")
    for col_el, cfg, par_v, pv_v in [(c1,"Fixed-tilt",fixed_par,fixed_pv),(c2,"Vertical",vert_par,vert_pv)]:
        is_rec = cfg == rec_cfg
        with col_el:
            border = "2px solid #2d6a4f" if is_rec else "1px solid var(--border)"
            bg = "#f0f7f4" if is_rec else "var(--surface2)"
            rec_b = '<span class="badge badge-rec">Recommended</span>' if is_rec else ""
            st.markdown(f'<div style="border:{border};background:{bg};border-radius:10px;padding:14px 16px"><div style="font-weight:600;margin-bottom:8px">{cfg} {rec_b}</div><div style="display:flex;gap:20px"><div><div class="label">PAR mean</div><div class="value-sm">{par_v:.0f}</div></div><div><div class="label">Energy</div><div class="value-sm">{pv_v:.1f} kWh</div></div></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="caption" style="margin-top:8px">{reason} · α={alpha_val}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh forecast", key="fc_refresh"): st.rerun()


# ── PAGE 3: DESIGN & COMPARE ──────────────────────────────────────────────────
def page_design():
    st.markdown('<div class="page-header"><div class="page-title">⚙️ Design & Compare</div><div class="page-sub">Simulate panel configurations and compare against AI forecast</div></div>', unsafe_allow_html=True)

    if "saved_scenarios" not in st.session_state: st.session_state["saved_scenarios"] = []
    if "design_result"   not in st.session_state: st.session_state["design_result"]   = None

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label" style="margin-bottom:14px">DESIGN PARAMETERS</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Agrivoltaic","Open cropland"], horizontal=True)
    open_cl = mode == "Open cropland"
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2, gap="large")
    with c1:
        ph   = st.slider("Panel height (m)",  0.5, 5.0, 2.0, 0.1, disabled=open_cl)
        ps   = st.slider("Panel spacing (m)", 0.5, 6.0, 3.0, 0.1, disabled=open_cl)
        tilt = st.slider("Tilt angle (°)",    0.0,60.0,25.0, 1.0, disabled=open_cl)
    with c2:
        ch   = st.slider("Canopy height (m)", 0.1, 3.0, 1.4, 0.1)
        lai  = st.slider("Leaf area index (LAI)", 0.0, 6.0, 3.0, 0.1)
        soil = st.radio("Soil wetness", ["Dry","Medium","Wet"], horizontal=True)
        trk  = st.toggle("Single axis tracking", disabled=open_cl)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    params = {"panel_height_m": ph if not open_cl else 0.8, "panel_spacing_m": ps if not open_cl else 1.2,
              "tilt_deg": tilt if not open_cl else 0.0, "canopy_height_m": ch, "lai": lai,
              "soil_wetness": soil, "single_axis_tracking": trk and not open_cl}

    if st.button("▶ Run simulation", use_container_width=False):
        st.session_state["design_result"] = {"params": params, "result": simulate_scenario(params), "mode": mode}

    if st.session_state["design_result"]:
        out = st.session_state["design_result"]["result"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label" style="margin-bottom:14px">SIMULATION RESULTS</div>', unsafe_allow_html=True)
        r1,r2,r3,r4 = st.columns(4, gap="medium")
        for col, label, val, unit in [(r1,"PV Performance",out["pv_performance"],"%"),(r2,"Crop Comfort",out["crop_comfort"],"%"),(r3,"Water Savings",out["water_savings_kpi"],"%"),(r4,"Leaf Cooling",out["leaf_cooling_c"],"°C")]:
            with col:
                color = pct_color(val) if unit=="%" else "var(--text)"
                st.markdown(f'<div class="stat-box"><div class="label">{label}</div><div class="value-md" style="color:{color}">{val:.1f}{unit}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="label" style="margin-bottom:10px">VS LIVE AI FORECAST</div>', unsafe_allow_html=True)
        try:
            d = api_get("/forecast", timeout=10)
            cmp1,cmp2,cmp3 = st.columns(3, gap="medium")
            with cmp1: st.markdown(f'<div class="stat-box"><div class="label">Your PV gain</div><div class="value-sm" style="color:#2d6a4f">{out["pv_gain_percent"]:.1f}%</div><div class="caption">AI peak: {d.get("pv_peak_kw",0):.1f} kW</div></div>', unsafe_allow_html=True)
            with cmp2: st.markdown(f'<div class="stat-box"><div class="label">AI PAR forecast</div><div class="value-sm">{d.get("par_mean",0):.0f} μmol/s/m²</div><div class="caption">live reading</div></div>', unsafe_allow_html=True)
            with cmp3: st.markdown(f'<div class="stat-box"><div class="label">AI recommendation</div><div class="value-sm">{d.get("recommended_config","—")}</div><div class="caption">Irrigation: {d.get("irrigation_pct",100)}%</div></div>', unsafe_allow_html=True)
        except: st.caption("Could not load live AI data.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        saved = st.session_state["saved_scenarios"]
        st.markdown(f'<div class="label" style="margin-bottom:8px">SAVE SCENARIO ({len(saved)}/3)</div>', unsafe_allow_html=True)
        if len(saved) < 3:
            sc_name = st.text_input("Name", value=mode, key="sc_name_inp", label_visibility="collapsed")
            if st.button("Save scenario"):
                name_clean = sc_name.strip() or "Scenario"
                idx = next((i for i,s in enumerate(saved) if s["name"]==name_clean), None)
                if idx is not None:
                    st.session_state["saved_scenarios"][idx] = {"name": name_clean, "params": params, "result": out}
                    st.success(f"Updated '{name_clean}'")
                else:
                    st.session_state["saved_scenarios"].append({"name": name_clean, "params": params, "result": out})
                    st.success(f"Saved ({len(saved)+1}/3)")
        else: st.warning("3 scenarios saved. Remove one below.")
        st.markdown("</div>", unsafe_allow_html=True)

    saved = st.session_state["saved_scenarios"]
    if not saved: return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="label" style="margin-bottom:14px">SAVED SCENARIOS ({len(saved)}/3)</div>', unsafe_allow_html=True)
    COLORS  = ["#2d6a4f","#b7791f","#1d4ed8"]
    METRICS = [("pv_gain_percent","PV Gain (%)",20.0),("leaf_cooling_c","Leaf Cooling (°C)",25.0),("water_savings_percent","Water Savings (%)",50.0),("comfort_score","Comfort Score",100.0),("heat_index_reduction_c","Heat Index Red.",20.0)]
    for i,s in enumerate(saved):
        c1,c2,c3 = st.columns([0.45,0.45,0.10])
        with c1: st.markdown(f'<span style="font-weight:700;color:{COLORS[i]}">{s["name"]}</span>', unsafe_allow_html=True)
        with c2: st.caption(f'PV {s["result"]["pv_performance"]:.0f}% · Comfort {s["result"]["crop_comfort"]:.0f}% · Water {s["result"]["water_savings_kpi"]:.0f}%')
        with c3:
            if st.button("✕", key=f"rm_{i}"): st.session_state["saved_scenarios"].pop(i); st.rerun()

    if len(saved) >= 2:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        names=[ s["name"] for s in saved]; results=[s["result"] for s in saved]; colors=COLORS[:len(saved)]
        metric_labels=[m[1] for m in METRICS]
        fig_radar = go.Figure()
        for ci,(name,res) in enumerate(zip(names,results)):
            nv=[res[k]/mv for k,_,mv in METRICS]; nv_c=nv+nv[:1]; th_c=metric_labels+metric_labels[:1]
            fig_radar.add_trace(go.Scatterpolar(r=nv_c,theta=th_c,fill='toself',name=name,line=dict(color=colors[ci],width=2),opacity=0.25))
            fig_radar.add_trace(go.Scatterpolar(r=nv_c,theta=th_c,fill=None,showlegend=False,line=dict(color=colors[ci],width=2)))
        fig_radar.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,1],gridcolor="#e5e7eb",tickfont=dict(color="#9ca3af",size=8)),angularaxis=dict(gridcolor="#e5e7eb",tickfont=dict(color="#4b5563",size=10))),paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#4b5563",family="DM Sans"),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)),margin=dict(l=30,r=30,t=20,b=20),height=300)
        st.plotly_chart(fig_radar, use_container_width=True)
        win_cols = st.columns(len(METRICS), gap="small")
        for i,(key,label,_) in enumerate(METRICS):
            vals=[r[key] for r in results]; best_i=vals.index(max(vals))
            with win_cols[i]:
                st.markdown(f'<div style="text-align:center;border:1px solid var(--border);border-radius:10px;padding:10px 6px;background:var(--surface2)"><div class="label">{label}</div><div style="font-weight:700;color:{colors[best_i]};font-size:13px;margin-top:4px">{names[best_i]}</div><div class="caption">{results[best_i][key]:.1f}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Agrivoltaic Dashboard", layout="wide")
    tabs = st.tabs(["Overview", "AI Forecast", "Design & Compare"])
    with tabs[0]: page_overview()
    with tabs[1]: page_forecast()
    with tabs[2]: page_design()

if __name__ == "__main__":
    main()
