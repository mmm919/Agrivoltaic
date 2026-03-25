from pathlib import Path
from typing import Any, Dict, List, Optional
import math

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
</style>
""",
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SENSORS_DIR = DATA_DIR / "sensors"
PRED_DIR = DATA_DIR / "predictions"

def api_get(path):
    r = requests.get(f"{BACKEND_URL}{path}", timeout=10)
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

def top_bar(title_text, subtitle_text, pills, accent_index=0):
    pills_html = ""
    for i, t in enumerate(pills):
        cls = "pill pill-accent" if accent_index is not None and i == accent_index else "pill"
        pills_html += f'<div class="{cls}">{t}</div>'
    st.markdown(f'<div class="topbar"><div><div class="topbar-title">{title_text}</div><div class="topbar-sub">{subtitle_text}</div></div><div class="topbar-right">{pills_html}</div></div>', unsafe_allow_html=True)

def kpi_class(v):
    if v >= 70: return "good"
    if v >= 40: return "mid"
    return "bad"

def kpi_circles(pv, comfort, water):
    st.markdown(f'<div class="kpi-row"><div class="kpi {kpi_class(pv)}"><div class="v">{int(round(pv))}%</div><div class="l">PV performance</div></div><div class="kpi {kpi_class(comfort)}"><div class="v">{int(round(comfort))}%</div><div class="l">Crop comfort</div></div><div class="kpi {kpi_class(water)}"><div class="v">{int(round(water))}%</div><div class="l">Water savings</div></div></div>', unsafe_allow_html=True)

def farm_conditions_card(location, solar, wind, humidity):
    st.markdown(f'<div class="card"><div class="section-title">Farm location condition</div><div class="muted">{location}</div><div style="display:flex;gap:22px;margin-top:10px;flex-wrap:wrap;"><div><b>Solar</b><div class="small">{solar}</div></div><div><b>Wind</b><div class="small">{wind}</div></div><div><b>Humidity</b><div class="small">{humidity}</div></div></div></div>', unsafe_allow_html=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def soil_factor(soil):
    if soil == "Dry": return 0.0
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
    sf = soil_factor(soil)
    pv_gain  = 12.0 if tracking else 0.0
    pv_gain += 6.0 * math.exp(-((tilt - 25.0) / 18.0) ** 2)
    pv_gain -= 0.6 * max(0.0, panel_height - 3.0)
    pv_gain += 0.35 * (panel_spacing - 2.0)
    pv_gain  = clamp(pv_gain, 0.0, 20.0)
    shade       = clamp((panel_height - 1.0) / 3.0, 0.0, 1.0)
    ventilation = clamp((panel_spacing - 1.5) / 3.0, 0.0, 1.0)
    lai_effect  = clamp(lai / 4.0, 0.0, 1.0)
    leaf_cooling  = 25.0 * (0.35*shade + 0.35*sf + 0.30*lai_effect) * (0.75 + 0.25*ventilation)
    leaf_cooling  = clamp(leaf_cooling, 0.0, 25.0)
    water_savings = clamp(10.0 + 35.0*(0.40*shade + 0.40*sf + 0.20*lai_effect), 0.0, 50.0)
    heat_index_reduction = clamp(0.5*leaf_cooling + 4.0*sf, 0.0, 20.0)
    comfort_score = clamp(45.0 + 2.0*leaf_cooling + 0.35*water_savings - 0.6*abs(canopy_height-1.4)*10.0, 0.0, 100.0)
    pv_performance = clamp(65.0 + pv_gain, 0.0, 100.0)
    water_kpi = clamp(water_savings * 2.0, 0.0, 100.0)
    return {"pv_performance": pv_performance, "crop_comfort": comfort_score, "water_savings_kpi": water_kpi,
            "leaf_cooling_c": leaf_cooling, "pv_gain_percent": pv_gain, "water_savings_percent": water_savings,
            "heat_index_reduction_c": heat_index_reduction, "comfort_score": comfort_score}

def story_curves(result):
    hours = np.arange(0, 24)
    base  = 28 + 10 * np.sin((hours - 6) / 24 * 2 * np.pi)
    cooling = result["leaf_cooling_c"]; hi_red = result["heat_index_reduction_c"]; pv_gain = result["pv_gain_percent"]
    leaf_curve        = base - (cooling/25.0)*(6 + 3*np.sin((hours-9)/24*2*np.pi))
    heat_index_curve  = (42 - 18*np.exp(-((hours-13)/3.2)**2)) - (hi_red/20.0)*6
    pv_curve          = (2 + 8*np.exp(-((hours-13)/4.2)**2)) + (pv_gain/20.0)*4
    return {"leaf": pd.DataFrame({"hour": hours, "leaf_temp_c": leaf_curve}),
            "hi":   pd.DataFrame({"hour": hours, "heat_index_proxy": heat_index_curve}),
            "pv":   pd.DataFrame({"hour": hours, "pv_benefit_proxy": pv_curve})}

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
    if not locations:
        st.error("No locations found. Add one in Settings."); return
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
                    if st.button("Remove", key=f"home_rm_alert_{i}"): api_delete(f"/alerts/{i}"); st.rerun()
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
                    if st.button("Remove", key=f"home_rm_update_{i}"): api_delete(f"/updates/{i}"); st.rerun()
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
        out = st.session_state["design_result"]["result"]
        used_mode = st.session_state["design_result"].get("mode","")
        st.markdown("")
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Outputs</div>', unsafe_allow_html=True)
        st.caption(f"Mode used: {used_mode}")
        kpi_circles(out["pv_performance"], out["crop_comfort"], out["water_savings_kpi"])
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("PV gain (%)", f'{out["pv_gain_percent"]:.1f}')
        m2.metric("Leaf cooling (C)", f'{out["leaf_cooling_c"]:.1f}')
        m3.metric("Water savings (%)", f'{out["water_savings_percent"]:.1f}')
        m4.metric("Heat index reduction (C)", f'{out["heat_index_reduction_c"]:.1f}')
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Save scenario for Compare</div>', unsafe_allow_html=True)
        default_name = "Open cropland" if open_cropland else "Agrivoltaic design"
        scenario_name = st.text_input("Scenario name", value=default_name)
        if st.button("Save scenario"):
            st.session_state["saved_scenarios"].append({"name": scenario_name.strip() or "Scenario", "params": params, "result": out})
            st.success("Saved to Compare")
        st.markdown("</div>", unsafe_allow_html=True)

def page_compare():
    top_bar("Compare", "Compare scenarios and view the story", [])
    if "saved_scenarios" not in st.session_state: st.session_state["saved_scenarios"] = []

    base_open = simulate_scenario({"panel_height_m":0.8,"panel_spacing_m":1.2,"tilt_deg":0.0,"canopy_height_m":1.4,"lai":3.0,"soil_wetness":"Medium","single_axis_tracking":False})
    pv_only_dry = simulate_scenario({"panel_height_m":3.0,"panel_spacing_m":4.0,"tilt_deg":25.0,"canopy_height_m":1.4,"lai":0.8,"soil_wetness":"Dry","single_axis_tracking":True})
    if st.session_state["saved_scenarios"]:
        s=st.session_state["saved_scenarios"][-1]; agr_title=s["name"]; agr_res=s["result"]
    else:
        agr_title="Agrivoltaic"; agr_res=simulate_scenario({"panel_height_m":2.0,"panel_spacing_m":3.0,"tilt_deg":25.0,"canopy_height_m":1.4,"lai":3.0,"soil_wetness":"Medium","single_axis_tracking":False})

    scenarios = {agr_title: agr_res, "Open cropland": base_open, "PV only dry": pv_only_dry}
    names   = list(scenarios.keys())
    results = list(scenarios.values())
    METRICS = [
        ("pv_gain_percent","PV Gain (%)",20.0),
        ("leaf_cooling_c","Leaf Cooling (°C)",25.0),
        ("water_savings_percent","Water Savings (%)",50.0),
        ("comfort_score","Comfort Score",100.0),
        ("heat_index_reduction_c","Heat Index Red. (°C)",20.0),
    ]
    COLORS = ["#22c55e","#f2c94c","#ff6b6b"]

    def winner_html(idx):
        labels=["🥇 Best","🥈 2nd","🥉 3rd"]
        bgs=["rgba(34,197,94,0.18)","rgba(242,201,76,0.18)","rgba(255,107,107,0.18)"]
        borders=["rgba(34,197,94,0.6)","rgba(242,201,76,0.6)","rgba(255,107,107,0.6)"]
        return f'<span style="border:1px solid {borders[idx]};border-radius:999px;padding:2px 9px;font-size:11px;background:{bgs[idx]}">{labels[idx]}</span>'

    # Section 1: scenario cards
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scenario comparison</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for ci, (name, res) in enumerate(scenarios.items()):
        with cols[ci]:
            bc = COLORS[ci]
            st.markdown(f'<div style="border:1.5px solid {bc};border-radius:14px;padding:14px 16px;background:rgba(255,255,255,0.02);">', unsafe_allow_html=True)
            st.markdown(f'<div style="font-weight:900;font-size:15px;margin-bottom:8px;color:{bc}">{name}</div>', unsafe_allow_html=True)
            for key, label, max_val in METRICS:
                vals = [r[key] for r in results]
                rank = sorted(vals, reverse=True).index(res[key])
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

    # Section 2: radar + bar
    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual comparison</div>', unsafe_allow_html=True)
    radar_col, bar_col = st.columns(2, gap="large")
    with radar_col:
        st.markdown('<div class="muted" style="margin-bottom:6px">Radar chart (normalised 0–1)</div>', unsafe_allow_html=True)
        metric_labels = [m[1] for m in METRICS]
        fig_radar = go.Figure()
        for ci, (name, res) in enumerate(scenarios.items()):
            nv = [res[k]/mv for k,_,mv in METRICS] ; nv_c = nv+nv[:1] ; th_c = metric_labels+metric_labels[:1]
            fig_radar.add_trace(go.Scatterpolar(r=nv_c,theta=th_c,fill='toself',name=name,line=dict(color=COLORS[ci],width=2.5),opacity=0.3))
            fig_radar.add_trace(go.Scatterpolar(r=nv_c,theta=th_c,fill=None,showlegend=False,line=dict(color=COLORS[ci],width=2.5)))
        fig_radar.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,1],gridcolor="rgba(255,255,255,0.12)",tickfont=dict(color="rgba(255,255,255,0.45)",size=8)),angularaxis=dict(gridcolor="rgba(255,255,255,0.12)",tickfont=dict(color="rgba(255,255,255,0.8)",size=10))),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.85)"),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)),margin=dict(l=30,r=30,t=20,b=20),height=330)
        st.plotly_chart(fig_radar, use_container_width=True)
    with bar_col:
        st.markdown('<div class="muted" style="margin-bottom:6px">Side-by-side metrics</div>', unsafe_allow_html=True)
        mk = [m[0] for m in METRICS]; mls = ["PV Gain","Leaf Cool.","Water Sav.","Comfort","Heat Red."]
        fig_bar = go.Figure()
        for ci,(name,res) in enumerate(scenarios.items()):
            fig_bar.add_trace(go.Bar(name=name,x=mls,y=[res[k] for k in mk],marker_color=COLORS[ci],opacity=0.85))
        fig_bar.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.85)",size=10),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)),xaxis=dict(gridcolor="rgba(255,255,255,0.06)",tickfont=dict(size=9)),yaxis=dict(gridcolor="rgba(255,255,255,0.10)"),margin=dict(l=10,r=10,t=20,b=10),height=330)
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 3: category winners
    st.markdown("")
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Category winners</div>', unsafe_allow_html=True)
    win_cols = st.columns(len(METRICS), gap="small")
    for i,(key,label,_) in enumerate(METRICS):
        vals=[ r[key] for r in results]; best_i=vals.index(max(vals))
        with win_cols[i]:
            st.markdown(f'<div style="text-align:center;border:1px solid rgba(255,255,255,0.10);border-radius:12px;padding:10px 6px;background:rgba(255,255,255,0.02);"><div style="font-size:11px;color:rgba(255,255,255,0.55);margin-bottom:4px">{label}</div><div style="font-size:13px;font-weight:800;color:{COLORS[best_i]}">{names[best_i]}</div><div style="font-size:12px;margin-top:2px">{results[best_i][key]:.1f}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 4: story charts
    st.markdown("")
    if st.button("📈 View story charts"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Story charts</div>', unsafe_allow_html=True)
        st.caption("Synthetic curves that reflect your selected scenario outputs")
        curves = story_curves(agr_res)
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

def page_report():
    top_bar("Report", "Inspect CSV files", [])
    sensor_files = sorted([p for p in SENSORS_DIR.rglob("*.csv") if p.is_file()]) if SENSORS_DIR.exists() else []
    pred_files   = sorted([p for p in PRED_DIR.rglob("*.csv")   if p.is_file()]) if PRED_DIR.exists()   else []
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sensors</div>', unsafe_allow_html=True)
    if not sensor_files: st.info("No sensor CSV files found in data/sensors")
    else:
        sname = st.selectbox("Choose sensors CSV", [p.name for p in sensor_files])
        df = pd.read_csv(next(p for p in sensor_files if p.name == sname))
        st.dataframe(df.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predictions</div>', unsafe_allow_html=True)
    if not pred_files: st.info("No prediction CSV files found in data/predictions")
    else:
        pname = st.selectbox("Choose predictions CSV", [p.name for p in pred_files])
        df2 = pd.read_csv(next(p for p in pred_files if p.name == pname))
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
    tabs = st.tabs(["Home","Design","Compare","Report","Settings"])
    with tabs[0]: page_home()
    with tabs[1]: page_design()
    with tabs[2]: page_compare()
    with tabs[3]: page_report()
    with tabs[4]: page_settings()

if __name__ == "__main__":
    main()
