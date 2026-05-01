from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = "https://agrivoltaic.onrender.com"

st.markdown(
    """
<style>
:root{
  --panel: rgba(255,255,255,0.03);
  --panel2: rgba(255,255,255,0.02);
  --border2: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
  --muted2: rgba(255,255,255,0.55);
  --accent: #22c55e;
  --good: #22c55e;
  --mid:  #f2c94c;
  --bad:  #ff6b6b;
}
.block-container{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
header[data-testid="stHeader"] { display: none; }
div[data-testid="stToolbar"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.card{ border:1px solid var(--border2); border-radius:16px; padding:16px 18px; background:var(--panel); }
.card-soft{ border:1px solid var(--border2); border-radius:16px; padding:16px 18px; background:var(--panel2); }
.section-title{ font-weight: 900; margin: 2px 0 10px 0; }
.muted{ color: var(--muted); font-size: 0.95rem; }
.small{ color: var(--muted2); font-size: 0.88rem; }
.topbar{
  display:flex; justify-content:space-between; align-items:center; gap:14px;
  padding:12px 14px; border:1px solid var(--border2); border-radius:16px;
  background: linear-gradient(180deg, rgba(34,197,94,0.12), rgba(255,255,255,0.02));
  margin-bottom: 14px;
}
.topbar-title{ font-weight: 1000; font-size: 18px; color: var(--text); }
.topbar-sub{ color: var(--muted); font-size: 13px; margin-top: 2px; }
.topbar-right{ display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
.pill{ border:1px solid var(--border2); border-radius:999px; padding:7px 11px; font-size:12px; color:var(--text); background: rgba(0,0,0,0.18); }
.pill-accent{ border:1px solid rgba(34,197,94,0.55); background: rgba(34,197,94,0.12); }
.kpi-row{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
.kpi{ width:102px; height:102px; border-radius:999px; border:2px solid rgba(255,255,255,0.14); display:flex; flex-direction:column; justify-content:center; align-items:center; background: rgba(0,0,0,0.10); }
.kpi .v{ font-size:18px; font-weight:1000; color:var(--text); }
.kpi .l{ font-size:12px; color:var(--muted); text-align:center; margin-top:2px; }
.kpi.good{ border-color: rgba(34,197,94,0.80); }
.kpi.mid{  border-color: rgba(242,201,76,0.75); }
.kpi.bad{  border-color: rgba(255,107,107,0.75); }
.stTabs [data-baseweb="tab"]{ font-weight: 800; }
.stTabs [aria-selected="true"]{ color: var(--accent) !important; }

/* AI forecast specific */
.ai-stat{ border:1px solid var(--border2); border-radius:14px; padding:14px 16px;
           background:var(--panel); text-align:center; }
.ai-stat .val{ font-size:22px; font-weight:1000; color:var(--text); }
.ai-stat .lbl{ font-size:11px; color:var(--muted2); margin-top:3px; }
.stress-ok  { border:1px solid rgba(34,197,94,0.5);  background:rgba(34,197,94,0.07);
               border-radius:14px; padding:14px 16px; }
.stress-bad { border:1px solid rgba(255,107,107,0.5); background:rgba(255,107,107,0.07);
               border-radius:14px; padding:14px 16px; }
.config-card{ border-radius:14px; padding:14px 16px; text-align:center; }
.irr-bar-wrap{ background:rgba(255,255,255,0.08); border-radius:999px; height:8px; margin-top:8px; }
.dli-gauge-wrap{ position:relative; display:inline-block; }
</style>
""",
    unsafe_allow_html=True,
)

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
SENSORS_DIR = DATA_DIR / "sensors"
PRED_DIR    = DATA_DIR / "predictions"

# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path, timeout=10):
    r = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(path, payload):
    r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

def api_put(path, payload):
    r = requests.put(f"{BACKEND_URL}{path}", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

def api_delete(path):
    r = requests.delete(f"{BACKEND_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()

# ── shared UI helpers ─────────────────────────────────────────────────────────

def top_bar(title_text, subtitle_text, pills, accent_index=0):
    pills_html = ""
    for i, t in enumerate(pills):
        cls = "pill pill-accent" if accent_index is not None and i == accent_index else "pill"
        pills_html += f'<div class="{cls}">{t}</div>'
    st.markdown(
        f'<div class="topbar"><div><div class="topbar-title">{title_text}</div>'
        f'<div class="topbar-sub">{subtitle_text}</div></div>'
        f'<div class="topbar-right">{pills_html}</div></div>',
        unsafe_allow_html=True,
    )

def kpi_class(v):
    if v >= 70: return "good"
    if v >= 40: return "mid"
    return "bad"

def kpi_circles(pv, comfort, water):
    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi {kpi_class(pv)}"><div class="v">{int(round(pv))}%</div><div class="l">PV performance</div></div>'
        f'<div class="kpi {kpi_class(comfort)}"><div class="v">{int(round(comfort))}%</div><div class="l">Crop comfort</div></div>'
        f'<div class="kpi {kpi_class(water)}"><div class="v">{int(round(water))}%</div><div class="l">Water savings</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def farm_conditions_card(location, solar, wind, humidity):
    st.markdown(
        f'<div class="card"><div class="section-title">Farm location condition</div>'
        f'<div class="muted">{location}</div>'
        f'<div style="display:flex;gap:22px;margin-top:10px;flex-wrap:wrap;">'
        f'<div><b>Solar</b><div class="small">{solar}</div></div>'
        f'<div><b>Wind</b><div class="small">{wind}</div></div>'
        f'<div><b>Humidity</b><div class="small">{humidity}</div></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

def clamp(x, lo, hi): return max(lo, min(hi, x))

def soil_factor(soil):
    if soil == "Dry":    return 0.0
    if soil == "Medium": return 0.5
    return 1.0

def simulate_scenario(params):
    panel_height  = float(params["panel_height_m"])
    panel_spacing = float(params["panel_spacing_m"])
    tilt          = float(params["tilt_deg"])
    canopy_height = float(params["canopy_height_m"])
    lai           = float(params["lai"])
    soil          = str(params["soil_wetness"])
    tracking      = bool(params["single_axis_tracking"])
    sf       = soil_factor(soil)
    pv_gain  = 12.0 if tracking else 0.0
    pv_gain += 6.0 * math.exp(-((tilt - 25.0) / 18.0) ** 2)
    pv_gain -= 0.6 * max(0.0, panel_height - 3.0)
    pv_gain += 0.35 * (panel_spacing - 2.0)
    pv_gain  = clamp(pv_gain, 0.0, 20.0)
    shade        = clamp((panel_height - 1.0) / 3.0, 0.0, 1.0)
    ventilation  = clamp((panel_spacing - 1.5) / 3.0, 0.0, 1.0)
    lai_effect   = clamp(lai / 4.0, 0.0, 1.0)
    leaf_cooling = 25.0*(0.35*shade+0.35*sf+0.30*lai_effect)*(0.75+0.25*ventilation)
    leaf_cooling = clamp(leaf_cooling, 0.0, 25.0)
    water_savings = clamp(10.0+35.0*(0.40*shade+0.40*sf+0.20*lai_effect), 0.0, 50.0)
    heat_index_reduction = clamp(0.5*leaf_cooling+4.0*sf, 0.0, 20.0)
    comfort_score  = clamp(45.0+2.0*leaf_cooling+0.35*water_savings-0.6*abs(canopy_height-1.4)*10.0, 0.0, 100.0)
    pv_performance = clamp(65.0 + pv_gain, 0.0, 100.0)
    water_kpi      = clamp(water_savings * 2.0, 0.0, 100.0)
    return {"pv_performance": pv_performance, "crop_comfort": comfort_score,
            "water_savings_kpi": water_kpi, "leaf_cooling_c": leaf_cooling,
            "pv_gain_percent": pv_gain, "water_savings_percent": water_savings,
            "heat_index_reduction_c": heat_index_reduction, "comfort_score": comfort_score}

def story_curves(result):
    hours = np.arange(0, 24)
    base  = 28 + 10 * np.sin((hours - 6) / 24 * 2 * np.pi)
    cooling = result["leaf_cooling_c"]; hi_red = result["heat_index_reduction_c"]; pv_gain = result["pv_gain_percent"]
    leaf_curve       = base - (cooling/25.0)*(6 + 3*np.sin((hours-9)/24*2*np.pi))
    heat_index_curve = (42 - 18*np.exp(-((hours-13)/3.2)**2)) - (hi_red/20.0)*6
    pv_curve         = (2  + 8*np.exp(-((hours-13)/4.2)**2)) + (pv_gain/20.0)*4
    return {"leaf": pd.DataFrame({"hour": hours, "leaf_temp_c": leaf_curve}),
            "hi":   pd.DataFrame({"hour": hours, "heat_index_proxy": heat_index_curve}),
            "pv":   pd.DataFrame({"hour": hours, "pv_benefit_proxy": pv_curve})}

def detect_unit(col: str) -> str:
    c = col.lower()
    if any(k in c for k in ["temp","temperature"]): return "°C"
    if any(k in c for k in ["moisture","humidity","rh"]): return "%"
    if any(k in c for k in ["pressure"]): return "hPa"
    if any(k in c for k in ["wind","speed"]): return "m/s"
    if any(k in c for k in ["rain","precip"]): return "mm"
    if any(k in c for k in ["radiation","solar","irrad"]): return "W/m²"
    return ""


# ── pages ─────────────────────────────────────────────────────────────────────

def page_home():
    top_bar("Farm Dashboard", "Location, KPIs, alerts, updates", [])
    try:
        locs_data    = api_get("/locations")
        kpis         = api_get("/kpis")
        alerts_data  = api_get("/alerts")
        updates_data = api_get("/updates")
    except Exception as e:
        st.error("Backend is not reachable. Start backend first.")
        st.caption(str(e)); return
    locations        = locs_data.get("locations", {})
    default_location = locs_data.get("default_location") or next(iter(locations.keys()), "")
    if not locations: st.error("No locations found. Add one in Settings."); return
    names = list(locations.keys())
    if "location" not in st.session_state: st.session_state["location"] = default_location
    if st.session_state["location"] not in names: st.session_state["location"] = names[0]
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Location</div>', unsafe_allow_html=True)
        st.session_state["location"] = st.selectbox("Choose location", names, index=names.index(st.session_state["location"]))
        st.markdown("</div>", unsafe_allow_html=True)
        loc = st.session_state["location"]; cond = locations.get(loc, {})
        farm_conditions_card(loc, cond.get("solar","-"), cond.get("wind","-"), cond.get("humidity","-"))
        st.markdown("")
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent alerts</div>', unsafe_allow_html=True)
        alerts = alerts_data.get("items", [])
        if not alerts: st.info("No alerts")
        else:
            for i, a in enumerate(alerts[:5]):
                t=a.get("type","Info"); m=a.get("message",""); ts=a.get("created_at","")
                c1,c2=st.columns([0.82,0.18])
                with c1:
                    if t=="Info": st.info(f"{m}\n\nAdded {ts}")
                    elif t=="Warning": st.warning(f"{m}\n\nAdded {ts}")
                    else: st.error(f"{m}\n\nAdded {ts}")
                with c2:
                    if st.button("Remove", key=f"home_rm_alert_{i}"): api_delete(f"/alerts/{a.get('id',i)}"); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key outcomes</div>', unsafe_allow_html=True)
        kpi_circles(kpis.get("pv",0), kpis.get("comfort",0), kpis.get("water",0))
        st.markdown('<div class="small" style="margin-top:10px;">These KPIs come from backend settings for now.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Add alert and update</div>', unsafe_allow_html=True)
        a_col, u_col = st.columns(2, gap="medium")
        with a_col:
            st.markdown("**Add alert**")
            with st.form("home_add_alert", clear_on_submit=True):
                a_type=st.selectbox("Type",["Info","Warning","Critical"],key="home_a_type")
                a_msg=st.text_area("Message",placeholder="Heat stress risk high from 12:00 to 15:00",key="home_a_msg")
                ok=st.form_submit_button("Add alert")
            if ok:
                if a_msg.strip(): api_post("/alerts",{"type":a_type,"message":a_msg.strip()}); st.success("Alert added"); st.rerun()
                else: st.warning("Alert message cannot be empty")
        with u_col:
            st.markdown("**Add update**")
            with st.form("home_add_update", clear_on_submit=True):
                u_title=st.text_input("Title",placeholder="System update",key="home_u_title")
                u_body=st.text_area("Details",placeholder="Describe what changed",key="home_u_body")
                ok2=st.form_submit_button("Add update")
            if ok2:
                if u_title.strip() and u_body.strip(): api_post("/updates",{"title":u_title.strip(),"body":u_body.strip()}); st.success("Update added"); st.rerun()
                else: st.warning("Update title and details cannot be empty")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent updates</div>', unsafe_allow_html=True)
        updates = updates_data.get("items", [])
        if not updates: st.info("No updates")
        else:
            for i, u in enumerate(updates[:5]):
                title=u.get("title",""); body=u.get("body",""); ts=u.get("created_at","")
                c1,c2=st.columns([0.82,0.18])
                with c1: st.markdown(f"**{title}**"); st.caption(f"{body}\n\nAdded {ts}")
                with c2:
                    if st.button("Remove", key=f"home_rm_update_{i}"): api_delete(f"/updates/{u.get('id',i)}"); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def page_design():
    top_bar("Design", "Choose parameters and run simulation", [])
    if "design_result" not in st.session_state: st.session_state["design_result"] = None
    if "saved_scenarios" not in st.session_state: st.session_state["saved_scenarios"] = []
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Design mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("Choose mode", ["Agrivoltaic design", "Open cropland"])
    st.caption("Open cropland locks PV geometry because there are no panels.")
    st.markdown("</div>", unsafe_allow_html=True)
    open_cropland = mode == "Open cropland"
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Design parameters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if open_cropland:
            panel_height  = st.slider("Panel height (m)",  0.5, 5.0, 0.8, 0.1, disabled=True)
            panel_spacing = st.slider("Panel spacing (m)", 0.5, 6.0, 1.2, 0.1, disabled=True)
            tilt          = st.slider("Tilt angle (deg)",  0.0,60.0, 0.0, 1.0, disabled=True)
        else:
            panel_height  = st.slider("Panel height (m)",  0.5, 5.0, 2.0, 0.1)
            panel_spacing = st.slider("Panel spacing (m)", 0.5, 6.0, 3.0, 0.1)
            tilt          = st.slider("Tilt angle (deg)",  0.0,60.0,25.0, 1.0)
    with c2:
        canopy_height = st.slider("Canopy height (m)", 0.1, 3.0, 1.4, 0.1)
        lai           = st.slider("Leaf area index (LAI)", 0.0, 6.0, 3.0, 0.1)
        soil          = st.radio("Soil wetness", ["Dry","Medium","Wet"], horizontal=True)
        tracking      = st.toggle("Single axis tracking", value=False, disabled=open_cropland)
    st.markdown("</div>", unsafe_allow_html=True)
    params = {"panel_height_m": panel_height, "panel_spacing_m": panel_spacing, "tilt_deg": tilt,
              "canopy_height_m": canopy_height, "lai": lai, "soil_wetness": soil, "single_axis_tracking": tracking}
    st.markdown("")
    if st.button("Run simulation"):
        st.session_state["design_result"] = {"params": params, "result": simulate_scenario(params), "mode": mode}
    if st.session_state["design_result"]:
        out      = st.session_state["design_result"]["result"]
        used_mode = st.session_state["design_result"].get("mode","")
        st.markdown("")
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Outputs</div>', unsafe_allow_html=True)
        st.caption(f"Mode used: {used_mode}")
        kpi_circles(out["pv_performance"], out["crop_comfort"], out["water_savings_kpi"])
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("PV gain (%)",            f'{out["pv_gain_percent"]:.1f}')
        m2.metric("Leaf cooling (C)",        f'{out["leaf_cooling_c"]:.1f}')
        m3.metric("Water savings (%)",       f'{out["water_savings_percent"]:.1f}')
        m4.metric("Heat index reduction (C)",f'{out["heat_index_reduction_c"]:.1f}')
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        saved = st.session_state["saved_scenarios"]
        st.markdown(f'<div class="section-title">Save scenario for Compare ({len(saved)}/3)</div>', unsafe_allow_html=True)
        if len(saved) >= 3:
            st.warning("3 scenarios already saved. Remove one from the Compare tab to free a slot.")
        else:
            default_name = "Open cropland" if open_cropland else "Agrivoltaic design"
            scenario_name = st.text_input("Scenario name", value=default_name)
            if st.button("Save scenario"):
                name_clean = scenario_name.strip() or "Scenario"
                existing_idx = next((i for i, s in enumerate(saved) if s["name"] == name_clean), None)
                if existing_idx is not None:
                    st.session_state["saved_scenarios"][existing_idx] = {"name": name_clean, "params": params, "result": out}
                    st.success(f"Updated '{name_clean}'")
                else:
                    st.session_state["saved_scenarios"].append({"name": name_clean, "params": params, "result": out})
                    st.success(f"Saved '{name_clean}' ({len(saved)+1}/3)")
        st.markdown("</div>", unsafe_allow_html=True)


def page_compare():
    top_bar("Compare", "Compare up to 3 of your saved scenarios", [])
    if "saved_scenarios" not in st.session_state: st.session_state["saved_scenarios"] = []
    saved = st.session_state["saved_scenarios"]
    COLORS  = ["#22c55e", "#f2c94c", "#ff6b6b"]
    METRICS = [
        ("pv_gain_percent",       "PV Gain (%)",          20.0),
        ("leaf_cooling_c",        "Leaf Cooling (°C)",     25.0),
        ("water_savings_percent", "Water Savings (%)",     50.0),
        ("comfort_score",         "Comfort Score",        100.0),
        ("heat_index_reduction_c","Heat Index Red. (°C)",  20.0),
    ]
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Saved scenarios ({len(saved)}/3)</div>', unsafe_allow_html=True)
    if not saved:
        st.info("No scenarios saved yet. Go to the **Design** tab, run a simulation, and click **Save scenario**.")
    else:
        for i, s in enumerate(saved[:3]):
            c1,c2,c3 = st.columns([0.48,0.42,0.10])
            with c1: st.markdown(f'<span style="font-weight:800;color:{COLORS[i]}">{s["name"]}</span>', unsafe_allow_html=True)
            with c2: st.caption(f'PV {s["result"]["pv_performance"]:.1f}% · Comfort {s["result"]["crop_comfort"]:.1f}% · Water {s["result"]["water_savings_kpi"]:.1f}%')
            with c3:
                if st.button("✕", key=f"rm_sc_{i}"): st.session_state["saved_scenarios"].pop(i); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    if len(saved) < 2:
        st.markdown("")
        st.warning("Save at least **2 scenarios** from the **Design** tab to enable comparison.")
        return
    st.markdown("")
    scenarios_list = saved[:3]; n = len(scenarios_list)
    colors  = COLORS[:n]; names = [s["name"] for s in scenarios_list]; results = [s["result"] for s in scenarios_list]

    def winner_html(rank):
        labels  = ["🥇 Best","🥈 2nd","🥉 3rd"]
        bgs     = ["rgba(34,197,94,0.18)","rgba(242,201,76,0.18)","rgba(255,107,107,0.18)"]
        borders = ["rgba(34,197,94,0.6)","rgba(242,201,76,0.6)","rgba(255,107,107,0.6)"]
        return f'<span style="border:1px solid {borders[rank]};border-radius:999px;padding:2px 9px;font-size:11px;background:{bgs[rank]}">{labels[rank]}</span>'

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scenario comparison</div>', unsafe_allow_html=True)
    cols = st.columns(n, gap="medium")
    for ci, (name, res) in enumerate(zip(names, results)):
        with cols[ci]:
            bc = colors[ci]
            st.markdown(f'<div style="border:1.5px solid {bc};border-radius:14px;padding:14px 16px;background:rgba(255,255,255,0.02);">', unsafe_allow_html=True)
            st.markdown(f'<div style="font-weight:900;font-size:15px;margin-bottom:8px;color:{bc}">{name}</div>', unsafe_allow_html=True)
            for key, label, max_val in METRICS:
                vals = [r[key] for r in results]; rank = sorted(vals, reverse=True).index(res[key])
                pct  = res[key] / max_val * 100
                st.markdown(f'''<div style="margin-bottom:9px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
                    <span style="font-size:12px;color:rgba(255,255,255,0.75)">{label}</span>
                    <span style="display:flex;gap:6px;align-items:center;"><span style="font-size:13px;font-weight:700">{res[key]:.1f}</span>{winner_html(rank)}</span>
                  </div>
                  <div style="background:rgba(255,255,255,0.08);border-radius:999px;height:5px;">
                    <div style="width:{min(pct,100):.1f}%;background:{bc};height:5px;border-radius:999px;"></div>
                  </div></div>''', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual comparison</div>', unsafe_allow_html=True)
    radar_col, bar_col = st.columns(2, gap="large")
    with radar_col:
        st.markdown('<div class="muted" style="margin-bottom:6px">Radar chart (normalised 0–1)</div>', unsafe_allow_html=True)
        metric_labels = [m[1] for m in METRICS]
        fig_radar = go.Figure()
        for ci, (name, res) in enumerate(zip(names, results)):
            nv = [res[k]/mv for k,_,mv in METRICS]; nv_c = nv+nv[:1]; th_c = metric_labels+metric_labels[:1]
            fig_radar.add_trace(go.Scatterpolar(r=nv_c,theta=th_c,fill='toself',name=name,line=dict(color=colors[ci],width=2.5),opacity=0.3))
            fig_radar.add_trace(go.Scatterpolar(r=nv_c,theta=th_c,fill=None,showlegend=False,line=dict(color=colors[ci],width=2.5)))
        fig_radar.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,1],gridcolor="rgba(255,255,255,0.12)",tickfont=dict(color="rgba(255,255,255,0.45)",size=8)),angularaxis=dict(gridcolor="rgba(255,255,255,0.12)",tickfont=dict(color="rgba(255,255,255,0.8)",size=10))),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.85)"),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)),margin=dict(l=30,r=30,t=20,b=20),height=330)
        st.plotly_chart(fig_radar, use_container_width=True)
    with bar_col:
        st.markdown('<div class="muted" style="margin-bottom:6px">Side-by-side metrics</div>', unsafe_allow_html=True)
        mk = [m[0] for m in METRICS]; mls = ["PV Gain","Leaf Cool.","Water Sav.","Comfort","Heat Red."]
        fig_bar = go.Figure()
        for ci, (name, res) in enumerate(zip(names, results)):
            fig_bar.add_trace(go.Bar(name=name,x=mls,y=[res[k] for k in mk],marker_color=colors[ci],opacity=0.85))
        fig_bar.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.85)",size=10),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)),xaxis=dict(gridcolor="rgba(255,255,255,0.06)",tickfont=dict(size=9)),yaxis=dict(gridcolor="rgba(255,255,255,0.10)"),margin=dict(l=10,r=10,t=20,b=10),height=330)
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Category winners</div>', unsafe_allow_html=True)
    win_cols = st.columns(len(METRICS), gap="small")
    for i, (key, label, _) in enumerate(METRICS):
        vals = [r[key] for r in results]; best_i = vals.index(max(vals))
        with win_cols[i]:
            st.markdown(f'<div style="text-align:center;border:1px solid rgba(255,255,255,0.10);border-radius:12px;padding:10px 6px;background:rgba(255,255,255,0.02);"><div style="font-size:11px;color:rgba(255,255,255,0.55);margin-bottom:4px">{label}</div><div style="font-size:13px;font-weight:800;color:{colors[best_i]}">{names[best_i]}</div><div style="font-size:12px;margin-top:2px">{results[best_i][key]:.1f}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    if st.button("📈 View story charts"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Story charts</div>', unsafe_allow_html=True)
        st.caption(f"Synthetic curves based on: {names[0]}")
        curves = story_curves(results[0])
        c1,c2,c3 = st.columns(3, gap="medium")
        cs = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.8)"),margin=dict(l=10,r=10,t=30,b=10),height=220,xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),yaxis=dict(gridcolor="rgba(255,255,255,0.07)"))
        with c1:
            f=px.line(curves["leaf"],x="hour",y="leaf_temp_c",labels={"hour":"Hour","leaf_temp_c":"Leaf Temp (°C)"},title="Leaf Temperature")
            f.update_traces(line=dict(color="#22c55e",width=2)); f.update_layout(**cs); st.plotly_chart(f,use_container_width=True)
        with c2:
            f=px.line(curves["hi"],x="hour",y="heat_index_proxy",labels={"hour":"Hour","heat_index_proxy":"Heat Index"},title="Heat Index Proxy")
            f.update_traces(line=dict(color="#f2c94c",width=2)); f.update_layout(**cs); st.plotly_chart(f,use_container_width=True)
        with c3:
            f=px.line(curves["pv"],x="hour",y="pv_benefit_proxy",labels={"hour":"Hour","pv_benefit_proxy":"PV Benefit"},title="PV Benefit Proxy")
            f.update_traces(line=dict(color="#ff6b6b",width=2)); f.update_layout(**cs); st.plotly_chart(f,use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def page_ai_forecast():
    top_bar("AI Forecast", "BiLSTM live predictions — PV power & crop light", [])

    # ── fetch forecast data ───────────────────────────────────────────────────
    try:
        d = api_get("/forecast", timeout=15)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 503:
            st.warning("⏳ AI model is warming up on the server — this takes ~30 seconds on first load. Please refresh in a moment.")
            st.caption("If this persists, the model files may not be deployed yet. See setup instructions below.")
            with st.expander("Setup instructions"):
                st.markdown("""
**To deploy the AI model on Render:**
1. Copy these files into your backend repo root:
   - `model.py`, `inference.py`, `dli_engine.py`
   - `model_weights.pt`, `scaler_X.pkl`, `scaler_y.pkl`
2. Add to `requirements.txt`: `torch`, `scikit-learn`, `joblib`
3. Push to GitHub — Render will redeploy automatically.
""")
        else:
            st.error(f"Backend error: {e}")
        return
    except Exception as e:
        st.error("Could not reach backend.")
        st.caption(str(e))
        return

    ts = d.get("timestamp","")
    crop = d.get("crop","lettuce").capitalize()

    # top meta bar
    next_update = "Next in 30 min"
    target = d.get("dli_threshold", 14.0)
    st.markdown(
        f'<div style="font-size:12px;color:rgba(255,255,255,0.5);margin-bottom:12px;">'
        f'Last update: {ts[11:16] if len(ts)>15 else ts} · {next_update} · '
        f'Crop: {crop} · Target: {target} mol/m²/day</div>',
        unsafe_allow_html=True,
    )

    # ── Row 1: dual forecast chart + DLI gauge ────────────────────────────────
    chart_col, gauge_col = st.columns([1.6, 1.0], gap="large")

    with chart_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">PV power + crop light forecast — next 60 minutes</div>', unsafe_allow_html=True)

        pv_vals  = d.get("pv_forecast_kw", [])
        par_vals = d.get("par_forecast", [])
        minutes  = [f"+{(i+1)*5}m" for i in range(len(pv_vals))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes, y=pv_vals, name="PV power (kW)",
            line=dict(color="#22c55e", width=2.5),
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=minutes, y=par_vals, name="Crop light PAR (μmol/s/m²)",
            line=dict(color="#6b7280", width=2, dash="dot"),
            yaxis="y2",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.85)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18, font=dict(size=10)),
            yaxis=dict(title="kW", gridcolor="rgba(255,255,255,0.07)", side="left"),
            yaxis2=dict(title="μmol/s/m²", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=10, r=10, t=10, b=40), height=240,
        )
        st.plotly_chart(fig, use_container_width=True)

        # summary stats below chart
        s1, s2, s3 = st.columns(3)
        s1.metric("PV peak",          f'{d.get("pv_peak_kw",0):.1f} kW')
        s2.metric("Energy next hour",  f'{d.get("pv_total_kwh",0):.1f} kWh')
        s3.metric("Crop light mean",   f'{d.get("par_mean",0):.0f}')
        st.markdown("</div>", unsafe_allow_html=True)

    with gauge_col:
        st.markdown('<div class="card" style="height:100%">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Daily light integral — crop budget</div>', unsafe_allow_html=True)

        dli_pct      = d.get("dli_pct", 0)
        dli_acc      = d.get("dli_accumulated", 0)
        dli_proj     = d.get("dli_projected_eod", 0)
        dli_thresh   = d.get("dli_threshold", 14.0)
        dli_deficit  = d.get("dli_deficit", 0)

        # donut gauge
        gauge_color = "#22c55e" if dli_pct >= 70 else ("#f2c94c" if dli_pct >= 40 else "#ff6b6b")
        fig_gauge = go.Figure(go.Pie(
            values=[dli_pct, max(0, 100 - dli_pct)],
            hole=0.72,
            marker_colors=[gauge_color, "rgba(255,255,255,0.06)"],
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
            direction="clockwise",
            rotation=90,
        ))
        fig_gauge.add_annotation(
            text=f"<b>{dli_pct:.0f}%</b><br><span style='font-size:11px'>of target</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white"),
            align="center",
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10), height=180,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f'<div style="font-size:12px;color:rgba(255,255,255,0.6);line-height:1.8;">'
            f'Collected: <b>{dli_acc:.1f} mol/m²</b><br>'
            f'Projected sunset: <b>{dli_proj:.1f} mol/m²</b><br>'
            f'Target: <b>{dli_thresh:.1f} mol/m²/day</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 2: stress alert + panel config + irrigation ───────────────────────
    alert_col, config_col, irr_col = st.columns([1.1, 1.2, 0.9], gap="large")

    with alert_col:
        stressed = d.get("stress_alert", False)
        cls      = "stress-bad" if stressed else "stress-ok"
        icon     = "⚠️" if stressed else "✅"
        status   = "Crop stress alert" if stressed else "Crop light OK"
        msg      = d.get("alert_message", "")
        deficit  = d.get("dli_deficit", 0)
        st.markdown(
            f'<div class="{cls}">'
            f'<div class="section-title">{icon} {status}</div>'
            f'<div style="font-size:12px;color:rgba(255,255,255,0.7);line-height:1.6;">{msg}</div>'
            f'<div style="margin-top:10px;display:flex;gap:24px;">'
            f'<div><div style="font-size:18px;font-weight:900">{deficit:.1f}</div><div style="font-size:11px;color:rgba(255,255,255,0.5)">Deficit mol/m²</div></div>'
            f'<div><div style="font-size:18px;font-weight:900">{dli_pct:.0f}%</div><div style="font-size:11px;color:rgba(255,255,255,0.5)">DLI progress</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with config_col:
        rec_cfg    = d.get("recommended_config", "—")
        fixed_par  = d.get("fixed_par_mean", 0)
        vert_par   = d.get("vertical_par_mean", 0)
        fixed_pv   = d.get("fixed_pv_kwh", 0)
        vert_pv    = d.get("vertical_pv_kwh", 0)
        reason     = d.get("recommendation_reason","")
        alpha_val  = d.get("alpha_crop_priority", 0.7)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Panel configuration recommendation</div>', unsafe_allow_html=True)

        fc1, fc2 = st.columns(2, gap="small")
        for col_el, cfg_name, par_v, pv_v in [
            (fc1, "Fixed-tilt", fixed_par, fixed_pv),
            (fc2, "Vertical",   vert_par,  vert_pv),
        ]:
            is_rec = cfg_name == rec_cfg
            border = "rgba(34,197,94,0.6)" if is_rec else "rgba(255,255,255,0.12)"
            bg     = "rgba(34,197,94,0.08)" if is_rec else "rgba(255,255,255,0.02)"
            badge  = '<span style="font-size:10px;background:rgba(34,197,94,0.2);border:1px solid rgba(34,197,94,0.5);border-radius:999px;padding:1px 7px;margin-left:6px;">Recommended</span>' if is_rec else ""
            with col_el:
                st.markdown(
                    f'<div style="border:1.5px solid {border};border-radius:12px;padding:10px 12px;background:{bg};text-align:center;">'
                    f'<div style="font-size:12px;font-weight:700">{cfg_name}{badge}</div>'
                    f'<div style="font-size:20px;font-weight:1000;margin-top:6px">{par_v:.0f}</div>'
                    f'<div style="font-size:10px;color:rgba(255,255,255,0.5)">μmol/s/m² PAR</div>'
                    f'<div style="font-size:13px;font-weight:700;margin-top:4px">{pv_v:.1f} kWh</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown(f'<div style="font-size:11px;color:rgba(255,255,255,0.45);margin-top:8px;">{reason}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with irr_col:
        irr_pct = d.get("irrigation_pct", 100)
        irr_col_css = "#22c55e" if irr_pct >= 90 else ("#f2c94c" if irr_pct >= 70 else "#ff6b6b")
        st.markdown(
            f'<div class="card" style="text-align:center;">'
            f'<div class="section-title">Irrigation automation</div>'
            f'<div style="font-size:38px;font-weight:1000;color:{irr_col_css};margin:8px 0 2px 0">{irr_pct}%</div>'
            f'<div style="font-size:12px;color:rgba(255,255,255,0.5);margin-bottom:10px">of normal schedule</div>'
            f'<div class="irr-bar-wrap"><div style="width:{irr_pct}%;background:{irr_col_css};height:8px;border-radius:999px;"></div></div>'
            f'<div style="font-size:11px;color:rgba(255,255,255,0.45);margin-top:10px;">'
            f'{"No adjustment needed — crop is receiving adequate light." if irr_pct==100 else f"Reduced by {100-irr_pct}% due to DLI deficit."}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Controls row ─────────────────────────────────────────────────────────
    with st.expander("⚙️ AI model controls"):
        ctrl1, ctrl2 = st.columns(2, gap="large")
        with ctrl1:
            st.markdown("**Crop type**")
            crop_opts = ["lettuce","spinach","wheat","tomato","cucumber","pepper"]
            cur_crop = d.get("crop","lettuce")
            new_crop = st.selectbox("Select crop", crop_opts,
                                    index=crop_opts.index(cur_crop) if cur_crop in crop_opts else 0,
                                    key="ai_crop_sel")
            if st.button("Apply crop"):
                try:
                    api_post(f"/crop/{new_crop}", {})
                    st.success(f"Crop set to {new_crop}"); st.rerun()
                except Exception as e:
                    st.error(str(e))
        with ctrl2:
            st.markdown("**Crop vs energy priority (α)**")
            st.caption("0 = optimise for energy only · 1 = optimise for crop light only")
            new_alpha = st.slider("α", 0.0, 1.0, float(d.get("alpha_crop_priority", 0.7)), 0.05, key="ai_alpha_sl")
            if st.button("Apply α"):
                try:
                    api_post("/treatment/alpha", {"alpha": new_alpha})
                    st.success(f"α set to {new_alpha}"); st.rerun()
                except Exception as e:
                    st.error(str(e))

    if st.button("🔄 Refresh forecast"):
        st.rerun()


def page_report():
    top_bar("Report", "Sensor data & ML predictions analysis", [])
    sensor_files = sorted([p for p in SENSORS_DIR.rglob("*.csv") if p.is_file()]) if SENSORS_DIR.exists() else []
    pred_files   = sorted([p for p in PRED_DIR.rglob("*.csv")   if p.is_file()]) if PRED_DIR.exists()   else []
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📡 Sensor Data</div>', unsafe_allow_html=True)
    if not sensor_files:
        st.info("No sensor CSV files found in data/sensors")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        sname = st.selectbox("Choose sensors CSV", [p.name for p in sensor_files])
        df = pd.read_csv(next(p for p in sensor_files if p.name == sname))
        time_col = None
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                try: df[c] = pd.to_datetime(df[c]); time_col = c
                except: pass
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.markdown('<div class="section-title" style="margin-top:12px">Summary statistics</div>', unsafe_allow_html=True)
        stats = df[num_cols].agg(["mean","median","std","min","max"]).T.round(4)
        stats.columns = ["Mean","Median","Std","Min","Max"]
        stats.insert(0, "Unit", [detect_unit(c) for c in stats.index])
        st.dataframe(stats, use_container_width=True)
        if num_cols:
            pill_cols = st.columns(len(num_cols))
            for i, col in enumerate(num_cols):
                with pill_cols[i]:
                    unit = detect_unit(col); label = f"{col} ({unit})" if unit else col
                    st.metric(label, f"{df[col].mean():.3f}", f"σ {df[col].std():.3f}")
        if time_col and num_cols:
            st.markdown('<div class="section-title" style="margin-top:14px">Sensor readings over time</div>', unsafe_allow_html=True)
            selected_cols = st.multiselect("Select columns to plot", num_cols, default=num_cols[:2] if len(num_cols)>=2 else num_cols)
            if selected_cols:
                fig = go.Figure()
                colors_list = ["#22c55e","#f2c94c","#ff6b6b","#60a5fa","#c084fc"]
                for i, col in enumerate(selected_cols):
                    fig.add_trace(go.Scatter(x=df[time_col], y=df[col], name=col, line=dict(color=colors_list[i%len(colors_list)],width=2)))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.85)"),legend=dict(bgcolor="rgba(0,0,0,0)"),xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),margin=dict(l=10,r=10,t=20,b=10),height=280)
                st.plotly_chart(fig, use_container_width=True)
        if len(num_cols) >= 2:
            st.markdown('<div class="section-title" style="margin-top:14px">Correlation heatmap</div>', unsafe_allow_html=True)
            corr = df[num_cols].corr().round(2)
            fig_corr = go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,colorscale="RdYlGn",zmin=-1,zmax=1,text=corr.values.round(2),texttemplate="%{text}",showscale=True))
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.85)"),margin=dict(l=10,r=10,t=10,b=10),height=260)
            st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('<div class="section-title" style="margin-top:14px">Raw data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 ML Predictions Analysis</div>', unsafe_allow_html=True)
    if not pred_files:
        st.info("No prediction CSV files found in data/predictions")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    pname = st.selectbox("Choose predictions CSV", [p.name for p in pred_files])
    df2 = pd.read_csv(next(p for p in pred_files if p.name == pname))
    true_col = next((c for c in df2.columns if "true" in c.lower()), None)
    pred_col = next((c for c in df2.columns if "pred" in c.lower()), None)
    if true_col and pred_col:
        y_true = df2[true_col].values.astype(float); y_pred = df2[pred_col].values.astype(float)
        mae  = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        ss_res = np.sum((y_true - y_pred)**2); ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2   = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
        if r2 >= 0.9:   verdict, v_color = "Excellent", "#22c55e"
        elif r2 >= 0.7: verdict, v_color = "Good",      "#22c55e"
        elif r2 >= 0.5: verdict, v_color = "Acceptable","#f2c94c"
        else:           verdict, v_color = "Needs work", "#ff6b6b"
        st.markdown('<div class="section-title" style="margin-top:4px">Model performance metrics</div>', unsafe_allow_html=True)
        r2_pct = max(0,min(100,r2*100)); mae_pct = max(0,100-min(100,mae*20)); rmse_pct = max(0,100-min(100,rmse*20))
        st.markdown(f'''<div class="kpi-row" style="margin-bottom:14px">
          <div class="kpi {kpi_class(r2_pct)}"><div class="v">{r2:.3f}</div><div class="l">R² Score</div></div>
          <div class="kpi {kpi_class(mae_pct)}"><div class="v">{mae:.3f}</div><div class="l">MAE</div></div>
          <div class="kpi {kpi_class(rmse_pct)}"><div class="v">{rmse:.3f}</div><div class="l">RMSE</div></div>
          <div style="margin-left:10px;border:1px solid {v_color};border-radius:12px;padding:10px 18px;background:rgba(255,255,255,0.02);">
            <div style="font-size:12px;color:rgba(255,255,255,0.55)">Model verdict</div>
            <div style="font-size:20px;font-weight:900;color:{v_color};margin-top:4px">{verdict}</div>
          </div></div>''', unsafe_allow_html=True)
        cs = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.8)"),margin=dict(l=10,r=10,t=30,b=10),height=260,xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),yaxis=dict(gridcolor="rgba(255,255,255,0.07)"))
        ch1,ch2 = st.columns(2, gap="medium")
        with ch1:
            st.markdown('<div class="muted" style="margin-bottom:4px">Actual vs Predicted</div>', unsafe_allow_html=True)
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(y=y_true,name="Actual",   line=dict(color="#22c55e",width=2)))
            fig_line.add_trace(go.Scatter(y=y_pred,name="Predicted",line=dict(color="#f2c94c",width=2,dash="dash")))
            fig_line.update_layout(**cs); st.plotly_chart(fig_line,use_container_width=True)
        with ch2:
            st.markdown('<div class="muted" style="margin-bottom:4px">Scatter: Actual vs Predicted</div>', unsafe_allow_html=True)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(x=y_true,y=y_pred,mode="markers",marker=dict(color="#60a5fa",size=6,opacity=0.7),name="Points"))
            mn,mx = float(min(y_true.min(),y_pred.min())),float(max(y_true.max(),y_pred.max()))
            fig_scatter.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",line=dict(color="#ff6b6b",dash="dash",width=1.5),name="Perfect"))
            fig_scatter.update_layout(xaxis_title="Actual",yaxis_title="Predicted",**cs); st.plotly_chart(fig_scatter,use_container_width=True)
        ch3,ch4 = st.columns(2, gap="medium")
        with ch3:
            st.markdown('<div class="muted" style="margin-bottom:4px">Residuals (Actual − Predicted)</div>', unsafe_allow_html=True)
            residuals = y_true - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(y=residuals,mode="markers",marker=dict(color="#c084fc",size=5,opacity=0.7),name="Residual"))
            fig_res.add_hline(y=0,line=dict(color="#ff6b6b",dash="dash",width=1.5))
            fig_res.update_layout(yaxis_title="Residual",**cs); st.plotly_chart(fig_res,use_container_width=True)
        with ch4:
            st.markdown('<div class="muted" style="margin-bottom:4px">Error distribution</div>', unsafe_allow_html=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=residuals,nbinsx=20,marker_color="#f2c94c",opacity=0.8,name="Error"))
            fig_hist.update_layout(xaxis_title="Error",yaxis_title="Count",**cs); st.plotly_chart(fig_hist,use_container_width=True)
    else:
        st.info("Could not detect y_true / y_pred columns automatically.")
    st.markdown('<div class="section-title" style="margin-top:14px">Raw predictions data</div>', unsafe_allow_html=True)
    st.dataframe(df2.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_settings():
    top_bar("Settings", "Manage locations and KPI values", [])
    try:
        locs_data = api_get("/locations"); kpis = api_get("/kpis")
    except Exception as e:
        st.error("Backend is not reachable. Start backend first."); st.caption(str(e)); return
    locations = locs_data.get("locations", {}); default_location = locs_data.get("default_location",""); names = list(locations.keys())
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Default location</div>', unsafe_allow_html=True)
    if names:
        idx = names.index(default_location) if default_location in names else 0
        new_default = st.selectbox("Choose default", names, index=idx)
        if st.button("Save default"): api_put(f"/default_location/{new_default}",{}); st.success("Saved"); st.rerun()
    else: st.info("No locations yet. Add one below.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Add or update location</div>', unsafe_allow_html=True)
    with st.form("loc_form", clear_on_submit=True):
        name     = st.text_input("Location name")
        solar    = st.selectbox("Solar",    ["Low","Medium","High"], index=2)
        wind     = st.selectbox("Wind",     ["Low","Medium","High"], index=1)
        humidity = st.selectbox("Humidity", ["Low","Medium","High"], index=1)
        save_loc = st.form_submit_button("Save location")
    if save_loc:
        if name.strip(): api_put(f"/locations/{name.strip()}",{"solar":solar,"wind":wind,"humidity":humidity}); st.success("Saved"); st.rerun()
        else: st.warning("Name cannot be empty")
    if names:
        del_name = st.selectbox("Delete location", names)
        if st.button("Delete selected"): api_delete(f"/locations/{del_name}"); st.success("Deleted"); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">KPI values</div>', unsafe_allow_html=True)
    pv      = st.slider("PV performance (%)", 0, 100, int(kpis.get("pv",82)))
    comfort = st.slider("Crop comfort (%)",   0, 100, int(kpis.get("comfort",74)))
    water   = st.slider("Water savings (%)",  0, 100, int(kpis.get("water",31)))
    if st.button("Save KPI values"): api_put("/kpis",{"pv":pv,"comfort":comfort,"water":water}); st.success("Saved")
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Farm Dashboard", layout="wide")
    tabs = st.tabs(["Home", "Design", "Compare", "AI Forecast", "Report", "Settings"])
    with tabs[0]: page_home()
    with tabs[1]: page_design()
    with tabs[2]: page_compare()
    with tabs[3]: page_ai_forecast()
    with tabs[4]: page_report()
    with tabs[5]: page_settings()

if __name__ == "__main__":
    main()
