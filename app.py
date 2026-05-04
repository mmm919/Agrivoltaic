from pathlib import Path
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = "https://agrivoltaic.onrender.com"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
:root {
  --bg:#0f1117; --surface:#1a1d27; --surface2:#21242f;
  --border:rgba(255,255,255,0.08); --text:rgba(255,255,255,0.92);
  --muted:rgba(255,255,255,0.55); --muted2:rgba(255,255,255,0.35);
  --accent:#22c55e; --radius:14px;
  --shadow:0 1px 3px rgba(0,0,0,0.4),0 4px 16px rgba(0,0,0,0.3);
}
* { font-family:'DM Sans',sans-serif; }
.block-container{padding-top:1.5rem;padding-bottom:2rem;max-width:1160px;background:var(--bg);}
header[data-testid="stHeader"],div[data-testid="stToolbar"],#MainMenu,footer{display:none!important;visibility:hidden!important;}
.stApp{background:var(--bg);}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px 22px;box-shadow:var(--shadow);}
.card-green{background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.25);border-radius:var(--radius);padding:20px 22px;}
.card-red{background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.25);border-radius:var(--radius);padding:20px 22px;}
.card-amber{background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25);border-radius:var(--radius);padding:20px 22px;}
.card-blue{background:rgba(96,165,250,0.07);border:1px solid rgba(96,165,250,0.25);border-radius:var(--radius);padding:20px 22px;}
.ph{padding:18px 0 16px 0;border-bottom:1px solid var(--border);margin-bottom:20px;}
.pt{font-size:22px;font-weight:700;color:var(--text);letter-spacing:-0.3px;}
.ps{font-size:13px;color:var(--muted);margin-top:3px;}
.lbl{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:4px;}
.vlg{font-size:28px;font-weight:700;color:var(--text);letter-spacing:-0.5px;line-height:1.1;}
.vmd{font-size:20px;font-weight:600;color:var(--text);}
.vsm{font-size:14px;font-weight:500;color:var(--text);}
.cap{font-size:12px;color:var(--muted2);margin-top:3px;}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px;}
.bg{background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid rgba(34,197,94,0.3);}
.br{background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3);}
.ba{background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);}
.bb{background:rgba(96,165,250,0.15);color:#93c5fd;border:1px solid rgba(96,165,250,0.3);}
.brec{background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid rgba(34,197,94,0.4);}
.pt-track{background:rgba(255,255,255,0.08);border-radius:999px;height:8px;margin-top:8px;overflow:hidden;}
.pt-fill{height:8px;border-radius:999px;}
.sbox{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;}
.div{height:1px;background:var(--border);margin:16px 0;}
.action-item{background:var(--surface2);border-left:3px solid var(--accent);border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px;}
.action-warn{background:rgba(245,158,11,0.07);border-left:3px solid #f59e0b;border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px;}
.action-red{background:rgba(239,68,68,0.07);border-left:3px solid #ef4444;border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-bottom:2px solid var(--border)!important;gap:0!important;padding:0!important;}
.stTabs [data-baseweb="tab"]{font-weight:600!important;font-size:13px!important;color:var(--muted)!important;padding:11px 20px!important;border-bottom:2px solid transparent!important;margin-bottom:-2px!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;background:transparent!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:20px!important;}
div[data-testid="stMetric"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:12px 16px!important;}
div[data-testid="stMetricValue"]{color:var(--text)!important;}
div[data-testid="stMetricLabel"]{color:var(--muted)!important;}
</style>
""", unsafe_allow_html=True)

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.6)", family="DM Sans", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=8,r=8,t=8,b=8),
)

def api_get(path, timeout=12):
    r = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
    r.raise_for_status(); return r.json()

def api_post(path, payload=None, timeout=10):
    r = requests.post(f"{BACKEND_URL}{path}", json=payload or {}, timeout=timeout)
    r.raise_for_status(); return r.json()

def clamp(x,lo,hi): return max(lo,min(hi,x))
def soil_factor(s): return {"Dry":0.0,"Medium":0.5,"Wet":1.0}.get(s,0.5)

def simulate_scenario(p):
    ph=float(p["panel_height_m"]); ps=float(p["panel_spacing_m"])
    tilt=float(p["tilt_deg"]); ch=float(p["canopy_height_m"])
    lai=float(p["lai"]); sf=soil_factor(p["soil_wetness"]); trk=bool(p["single_axis_tracking"])
    pv=12.0 if trk else 0.0
    pv+=6.0*math.exp(-((tilt-25.0)/18.0)**2); pv-=0.6*max(0.0,ph-3.0); pv+=0.35*(ps-2.0); pv=clamp(pv,0.0,20.0)
    sh=clamp((ph-1.0)/3.0,0.0,1.0); vt=clamp((ps-1.5)/3.0,0.0,1.0); le=clamp(lai/4.0,0.0,1.0)
    lc=25.0*(0.35*sh+0.35*sf+0.30*le)*(0.75+0.25*vt); lc=clamp(lc,0.0,25.0)
    ws=clamp(10.0+35.0*(0.40*sh+0.40*sf+0.20*le),0.0,50.0)
    hi=clamp(0.5*lc+4.0*sf,0.0,20.0)
    cs=clamp(45.0+2.0*lc+0.35*ws-0.6*abs(ch-1.4)*10.0,0.0,100.0)
    return {"pv_performance":clamp(65.0+pv,0.0,100.0),"crop_comfort":cs,"water_savings_kpi":clamp(ws*2.0,0.0,100.0),
            "leaf_cooling_c":lc,"pv_gain_percent":pv,"water_savings_percent":ws,"heat_index_reduction_c":hi,"comfort_score":cs}

def fetch_live(alpha=0.7):
    try:
        comp = api_get(f"/treatment/compare?alpha={alpha}", timeout=15)
        comp["recommended_config"] = comp.get("recommended_config","—").replace("Fixedtilt","Fixed-tilt")
        d, _ = fetch_forecast()
        return comp, d or {}, None
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 503: return None, {}, "warming_up"
        return None, {}, str(e)
    except Exception as e:
        return None, {}, str(e)

def fetch_forecast():
    try:
        d = api_get("/forecast", timeout=15)
        return d, None
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 503: return None, "warming_up"
        return None, str(e)
    except Exception as e:
        return None, str(e)

def show_err(err):
    if err == "warming_up": st.warning("⏳ AI model warming up — refresh in ~30 seconds.")
    else: st.error("Backend not reachable."); st.caption(str(err))

def pbar(pct, color): return f'<div class="pt-track"><div class="pt-fill" style="width:{min(pct,100):.0f}%;background:{color}"></div></div>'

def run_status_bar(d):
    """Show last run timestamp and next run countdown."""
    ts = d.get("timestamp","")
    if not ts:
        return
    try:
        from datetime import datetime, timedelta
        # Convert to Beirut time (UTC+3)
        last_utc = datetime.fromisoformat(ts)
        beirut_offset = timedelta(hours=3)
        last_beirut = last_utc + beirut_offset
        now_beirut  = datetime.utcnow() + beirut_offset
        mins_ago = int((now_beirut - last_beirut).total_seconds() / 60)
        next_in  = max(0, 30 - mins_ago)
        if mins_ago <= 0:
            last_str = "just now"
        elif mins_ago == 1:
            last_str = "1 min ago"
        else:
            last_str = f"{mins_ago} min ago"
        next_str = "due now" if next_in == 0 else f"in {next_in} min"
        # Always read crop from ai/status for accuracy
        try:
            _ai_s = api_get("/ai/status", timeout=3)
            crop = _ai_s.get("crop", d.get("crop","lettuce")).capitalize()
        except:
            crop = d.get("crop","lettuce").capitalize()
        time_display = last_beirut.strftime("%H:%M")
        st.markdown(
            f'''<div style="display:flex;gap:24px;align-items:center;padding:8px 14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;margin-bottom:16px;flex-wrap:wrap">
            <div style="font-size:11px;color:rgba(255,255,255,0.4)">🕐 Last run: <span style="color:rgba(255,255,255,0.7);font-weight:600">{last_str}</span> &nbsp;({time_display} Beirut)</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.4)">⏭ Next run: <span style="color:rgba(255,255,255,0.7);font-weight:600">{next_str}</span></div>
            <div style="font-size:11px;color:rgba(255,255,255,0.4)">🌿 Active crop: <span style="color:#22c55e;font-weight:600">{crop}</span></div>
            </div>''',
            unsafe_allow_html=True
        )
    except:
        pass

def header(title, sub): st.markdown(f'<div class="ph"><div class="pt">{title}</div><div class="ps">{sub}</div></div>', unsafe_allow_html=True)

# ── AUTHENTICATION ────────────────────────────────────────────────────────────
def auth_page():
    """Sign in / Sign up page."""
    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;min-height:70vh">
    <div style="width:380px">
    <div style="text-align:center;margin-bottom:32px">
        <div style="font-size:36px;margin-bottom:8px">🌱</div>
        <div style="font-size:24px;font-weight:700;color:rgba(255,255,255,0.92);letter-spacing:-0.5px">AgroVision AI</div>
        <div style="font-size:13px;color:rgba(255,255,255,0.45);margin-top:4px">Agrivoltaic Farm Intelligence System</div>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email", placeholder="your@email.com")
                password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                if submitted:
                    if not email or not password:
                        st.error("Please fill in all fields.")
                    else:
                        try:
                            r = requests.post(f"{BACKEND_URL}/auth/login",
                                              json={"email": email, "password": password}, timeout=10)
                            if r.status_code == 200:
                                data = r.json()
                                st.session_state["authenticated"] = True
                                st.session_state["user_email"] = data.get("email", email)
                                st.rerun()
                            else:
                                detail = r.json().get("detail", "Login failed.")
                                st.error(detail)
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach backend. Please try again later.")
                        except Exception as e:
                            st.error(f"Error: {e}")

        with tab_signup:
            with st.form("signup_form"):
                new_email = st.text_input("Email", key="signup_email", placeholder="your@email.com")
                new_pass = st.text_input("Password", type="password", key="signup_pass", placeholder="Min 6 characters")
                confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm", placeholder="Re-enter password")
                signed = st.form_submit_button("Create Account", use_container_width=True)
                if signed:
                    if not new_email or not new_pass or not confirm_pass:
                        st.error("Please fill in all fields.")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match.")
                    elif len(new_pass) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        try:
                            r = requests.post(f"{BACKEND_URL}/auth/signup",
                                              json={"email": new_email, "password": new_pass}, timeout=10)
                            if r.status_code == 200:
                                st.success("Account created! Please sign in.")
                            else:
                                detail = r.json().get("detail", "Signup failed.")
                                st.error(detail)
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach backend. Please try again later.")
                        except Exception as e:
                            st.error(f"Error: {e}")

def check_auth():
    """Returns True if user is authenticated."""
    return st.session_state.get("authenticated", False)


# ── PAGE 1: OVERVIEW ─────────────────────────────────────────────────────────
def page_overview():
    header("🌱 Overview", "Farm status at a glance")

    # Fetch all data
    try:
        ai = api_get("/ai/status", timeout=5)
        alpha_val = ai.get("alpha", 0.7)
        current_crop = ai.get("crop", "lettuce")
    except:
        alpha_val = 0.7; current_crop = "lettuce"

    d, err = fetch_forecast()
    if err: show_err(err); return
    run_status_bar(d)

    try:
        comp = api_get(f"/treatment/compare?alpha={alpha_val}", timeout=15)
        comp["recommended_config"] = comp.get("recommended_config","—").replace("Fixedtilt","Fixed-tilt")
        rec_cfg = comp.get("recommended_config","—")
        fc = comp.get("vertical_forecast",{}) if rec_cfg=="Vertical" else comp.get("fixed_forecast",{})
    except:
        rec_cfg = "—"; fc = {}

    stressed  = d.get("stress_alert", False)
    dli_pct   = d.get("dli_pct", 0)
    dli_acc   = d.get("dli_accumulated", 0)
    dli_def   = d.get("dli_deficit", 0)
    irr_pct   = d.get("irrigation_pct", 100)
    # Read threshold from ai/status so it updates immediately when crop changes
    try:
        _ai_thresh = api_get("/ai/status", timeout=3)
        dli_thresh = float(_ai_thresh.get("dli_threshold", d.get("dli_threshold", 14.0)))
    except:
        dli_thresh = d.get("dli_threshold", 14.0)
    # Rebuild alert message using current crop name
    if stressed:
        msg = (f"Crop light stress detected. Projected DLI {d.get('dli_projected_eod',0):.1f} mol/m² "
               f"is below {current_crop.capitalize()} target {dli_thresh:.0f} mol/m²/day "
               f"(deficit {dli_def:.1f} mol/m²). Irrigation reduced to {irr_pct}% of normal.")
    else:
        msg = (f"DLI on track — projected {d.get('dli_projected_eod',0):.1f} mol/m² by sunset. "
               f"{current_crop.capitalize()} target {dli_thresh:.0f} mol/m²/day will be met. "
               f"Irrigation at {irr_pct}%.")

    # ── 4 KPI cards ───────────────────────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4, gap="medium")
    with k1: st.markdown(f'<div class="card"><div class="lbl">PV Peak Power</div><div class="vlg">{fc.get("pv_peak_kw",0):.1f} <span style="font-size:14px;color:var(--muted)">kW</span></div><div class="cap">Next 60 min</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="card"><div class="lbl">Crop Light (PAR)</div><div class="vlg">{fc.get("par_mean",0):.0f} <span style="font-size:14px;color:var(--muted)">umol/s/m2</span></div><div class="cap">Mean next hour</div></div>', unsafe_allow_html=True)
    with k3:
        gc = "#22c55e" if not stressed else "#ef4444"
        badge = '<span class="badge bg">On track</span>' if not stressed else '<span class="badge br">Stress</span>'
        st.markdown(f'<div class="card"><div class="lbl">DLI — {current_crop.capitalize()}</div><div class="vlg">{dli_pct:.0f}<span style="font-size:14px;color:var(--muted)">%</span></div>{pbar(dli_pct,gc)}<div style="margin-top:6px">{badge}</div></div>', unsafe_allow_html=True)
    with k4:
        ic = "#22c55e" if irr_pct>=90 else ("#f59e0b" if irr_pct>=70 else "#ef4444")
        st.markdown(f'<div class="card"><div class="lbl">Irrigation</div><div class="vlg" style="color:{ic}">{irr_pct}<span style="font-size:14px;color:var(--muted)">%</span></div>{pbar(irr_pct,ic)}<div class="cap">of schedule</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3 status panels ───────────────────────────────────────────────────────
    s1,s2,s3 = st.columns(3, gap="large")

    with s1:
        cls = "card-red" if stressed else "card-green"
        icon = "⚠️" if stressed else "✅"
        st.markdown(f'''<div class="{cls}">
            <div style="font-weight:700;margin-bottom:6px">{icon} Crop Light Status</div>
            <div style="font-size:13px;color:var(--text);line-height:1.6">{msg}</div>
            <div style="display:flex;gap:20px;margin-top:10px">
                <div><div class="lbl">Collected</div><div class="vsm">{dli_acc:.1f} mol/m²</div></div>
                <div><div class="lbl">Target</div><div class="vsm">{dli_thresh:.0f} mol/m²/day</div></div>
            </div>
            <div style="margin-top:8px"><span style="font-size:11px;color:var(--muted)">→ See DLI tab for details</span></div>
        </div>''', unsafe_allow_html=True)

    with s2:
        irr_cls = "card-green" if irr_pct>=90 else ("card-amber" if irr_pct>=70 else "card-red")
        irr_msg = "Running at full schedule" if irr_pct==100 else f"Reduced by {100-irr_pct}% due to light deficit"
        st.markdown(f'''<div class="{irr_cls}">
            <div style="font-weight:700;margin-bottom:6px">💧 Irrigation Status</div>
            <div style="font-size:13px;color:var(--text);line-height:1.6">{irr_msg}</div>
            <div style="margin-top:10px"><div class="lbl">Current schedule</div>
            <div style="font-size:22px;font-weight:700;color:{ic}">{irr_pct}%</div></div>
            <div style="margin-top:8px"><span style="font-size:11px;color:var(--muted)">→ See Irrigation tab for details</span></div>
        </div>''', unsafe_allow_html=True)

    with s3:
        fixed_par=comp.get("fixed_par_mean",0) if comp else 0
        vert_par=comp.get("vertical_par_mean",0) if comp else 0
        fixed_pv=comp.get("fixed_pv_kwh",0) if comp else 0
        vert_pv=comp.get("vertical_pv_kwh",0) if comp else 0
        reason=comp.get("reason","") if comp else ""
        st.markdown(f'''<div class="card-green">
            <div style="font-weight:700;margin-bottom:6px">☀️ Panel Recommendation</div>
            <div style="font-size:20px;font-weight:700;color:#22c55e;margin-bottom:4px">{rec_cfg}</div>
            <div style="font-size:13px;color:var(--text);line-height:1.6">{reason}</div>
            <div style="margin-top:8px"><span style="font-size:11px;color:var(--muted)">→ See AI Forecast tab for details</span></div>
        </div>''', unsafe_allow_html=True)

    if st.button("🔄 Refresh", key="ov_ref"): st.rerun()


# ── PAGE 2: AI FORECAST ───────────────────────────────────────────────────────
def page_forecast():
    header("🤖 AI Forecast", "BiLSTM model — PV R²=0.9085 · PAR R²=0.8926")
    c1,c2,c3 = st.columns([1,1,1], gap="large")
    with c1: crop_sel = st.selectbox("🌿 Crop", ["lettuce","tomato","wheat"], key="fc_crop")
    # Read current alpha from backend to keep in sync with Overview
    try:
        _ai = api_get("/ai/status", timeout=5)
        _backend_alpha = float(_ai.get("alpha", 0.7))
    except:
        _backend_alpha = 0.7
    # Only set default from backend if slider hasn't been touched this session
    if "fc_alpha" not in st.session_state:
        st.session_state["fc_alpha"] = _backend_alpha
    with c2: alpha_sel = st.slider("⚖️ Crop vs energy (α)", 0.0, 1.0, _backend_alpha, 0.05, key="fc_alpha", help="0=max energy · 1=max crop light")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Apply & refresh", key="fc_apply"):
            try: api_post(f"/crop/{crop_sel}"); api_post("/treatment/alpha",{"alpha":alpha_sel}); st.success("Applied")
            except Exception as e: st.error(str(e))

    st.markdown("<br>", unsafe_allow_html=True)
    try:
        comp = api_get(f"/treatment/compare?alpha={alpha_sel}", timeout=15)
        comp["recommended_config"] = comp.get("recommended_config","—").replace("Fixedtilt","Fixed-tilt")
    except requests.exceptions.HTTPError as e:
        show_err("warming_up" if e.response and e.response.status_code==503 else str(e)); return
    except Exception as e: show_err(str(e)); return

    rec_cfg=comp.get("recommended_config","—")
    fc=comp.get("vertical_forecast",{}) if rec_cfg=="Vertical" else comp.get("fixed_forecast",{})
    pv_v=fc.get("pv_forecast_kw",[]); par_v=fc.get("par_forecast",[])
    mins=[f"+{(i+1)*5}m" for i in range(len(pv_v))]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="lbl" style="margin-bottom:10px">BILSTM FORECAST — {rec_cfg.upper()} · α={alpha_sel}</div>', unsafe_allow_html=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=mins,y=pv_v,name="PV (kW)",line=dict(color="#22c55e",width=2.5),fill="tozeroy",fillcolor="rgba(34,197,94,0.08)",yaxis="y1",mode="lines+markers",marker=dict(size=5,color="#22c55e")))
    fig.add_trace(go.Scatter(x=mins,y=par_v,name="PAR (umol/s/m2)",line=dict(color="#f59e0b",width=2.5),yaxis="y2",mode="lines+markers",marker=dict(size=5,color="#f59e0b")))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.6)",family="DM Sans",size=11),xaxis=dict(gridcolor="rgba(255,255,255,0.08)",linecolor="rgba(255,255,255,0.08)",tickfont=dict(size=10)),yaxis=dict(title="PV (kW)",gridcolor="rgba(255,255,255,0.08)"),yaxis2=dict(title="PAR",overlaying="y",side="right",gridcolor="rgba(0,0,0,0)"),legend=dict(orientation="h",y=-0.2,bgcolor="rgba(0,0,0,0)"),height=280,margin=dict(l=8,r=8,t=8,b=50))
    st.plotly_chart(fig,use_container_width=True)
    s1,s2,s3=st.columns(3)
    s1.metric("PV peak",f'{fc.get("pv_peak_kw",0):.1f} kW')
    s2.metric("Energy next hour",f'{fc.get("pv_total_kwh",0):.1f} kWh')
    s3.metric("Crop light mean",f'{fc.get("par_mean",0):.0f}')
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    fixed_par=comp.get("fixed_par_mean",0); vert_par=comp.get("vertical_par_mean",0)
    fixed_pv=comp.get("fixed_pv_kwh",0); vert_pv=comp.get("vertical_pv_kwh",0)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="lbl" style="margin-bottom:12px">FIXED-TILT VS VERTICAL</div>', unsafe_allow_html=True)
    fig2=go.Figure()
    fig2.add_trace(go.Bar(name="Fixed-tilt",x=["PAR","PV Energy (x10)"],y=[fixed_par,fixed_pv*10],marker_color="#6b7280",opacity=0.85))
    fig2.add_trace(go.Bar(name="Vertical",x=["PAR","PV Energy (x10)"],y=[vert_par,vert_pv*10],marker_color="#22c55e",opacity=0.85))
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.6)",family="DM Sans",size=11),xaxis=dict(gridcolor="rgba(255,255,255,0.08)",linecolor="rgba(255,255,255,0.08)",tickfont=dict(size=10)),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)),yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),barmode="group",height=220,margin=dict(l=8,r=8,t=8,b=8))
    st.plotly_chart(fig2,use_container_width=True)
    cc1,cc2=st.columns(2,gap="large")
    for col,cfg,pv_val,par_val in [(cc1,"Fixed-tilt",fixed_pv,fixed_par),(cc2,"Vertical",vert_pv,vert_par)]:
        is_rec=cfg==rec_cfg
        b="2px solid #22c55e" if is_rec else "1px solid rgba(255,255,255,0.08)"
        bg="rgba(34,197,94,0.07)" if is_rec else "rgba(255,255,255,0.03)"
        rb='<span class="badge brec">Recommended</span>' if is_rec else ""
        with col: st.markdown(f'<div style="border:{b};background:{bg};border-radius:10px;padding:14px 16px"><div style="font-weight:600;margin-bottom:8px">{cfg} {rb}</div><div style="display:flex;gap:20px"><div><div class="lbl">PAR</div><div class="vsm">{par_val:.0f}</div></div><div><div class="lbl">Energy</div><div class="vsm">{pv_val:.1f} kWh</div></div></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cap" style="margin-top:8px">{comp.get("reason","")}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE 3: DLI ───────────────────────────────────────────────────────────────
def page_dli():
    header("🌞 Daily Light Integral", "Crop light budget tracking and recommendations")
    d, err = fetch_forecast()
    if err: show_err(err); return
    # Sync crop from backend so it matches AI Forecast page
    try:
        ai = api_get("/ai/status", timeout=5)
        d["crop"] = ai.get("crop", d.get("crop","lettuce"))
    except: pass
    run_status_bar(d)

    crop      = d.get("crop","lettuce").capitalize()
    dli_acc   = d.get("dli_accumulated", 0)
    dli_proj  = d.get("dli_projected_eod", 0)
    dli_thresh= d.get("dli_threshold", 14.0)
    dli_pct   = d.get("dli_pct", 0)
    dli_def   = d.get("dli_deficit", 0)
    stressed  = d.get("stress_alert", False)
    msg       = d.get("alert_message", "")

    # Status card
    cls = "card-red" if stressed else "card-green"
    icon = "⚠️" if stressed else "✅"
    status = "Crop light stress detected" if stressed else "Crop light adequate"
    st.markdown(f'<div class="{cls}"><div style="font-size:15px;font-weight:700;margin-bottom:8px">{icon} {status}</div><div style="font-size:13px;color:var(--text);line-height:1.7">{msg}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # DLI gauge + stats
    g1,g2,g3,g4 = st.columns(4, gap="medium")
    with g1: st.markdown(f'<div class="card"><div class="lbl">Collected Today</div><div class="vlg">{dli_acc:.1f}<span style="font-size:14px;color:var(--muted)"> mol/m²</span></div></div>', unsafe_allow_html=True)
    with g2: st.markdown(f'<div class="card"><div class="lbl">Projected Sunset</div><div class="vlg">{dli_proj:.1f}<span style="font-size:14px;color:var(--muted)"> mol/m²</span></div></div>', unsafe_allow_html=True)
    with g3: st.markdown(f'<div class="card"><div class="lbl">Daily Target ({crop})</div><div class="vlg">{dli_thresh:.0f}<span style="font-size:14px;color:var(--muted)"> mol/m²</span></div></div>', unsafe_allow_html=True)
    with g4:
        def_color = "#ef4444" if dli_def > 0 else "#22c55e"
        st.markdown(f'<div class="card"><div class="lbl">Deficit</div><div class="vlg" style="color:{def_color}">{dli_def:.1f}<span style="font-size:14px;color:var(--muted)"> mol/m²</span></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Progress bar
    gc = "#22c55e" if not stressed else "#ef4444"
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="lbl" style="margin-bottom:10px">DLI PROGRESS — {dli_pct:.0f}% OF DAILY TARGET</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:rgba(255,255,255,0.08);border-radius:999px;height:16px;overflow:hidden"><div style="width:{min(dli_pct,100):.0f}%;background:{gc};height:16px;border-radius:999px;transition:width 0.4s"></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="display:flex;justify-content:space-between;margin-top:6px"><div class="cap">0 mol/m²</div><div class="cap">{dli_thresh:.0f} mol/m²/day target</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Actions
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="lbl" style="margin-bottom:12px">RECOMMENDED ACTIONS</div>', unsafe_allow_html=True)

    if stressed:
        deficit_pct = (dli_def / dli_thresh * 100) if dli_thresh > 0 else 0
        if deficit_pct > 50:
            st.markdown('<div class="action-red"><div style="font-weight:700;color:#f87171;margin-bottom:4px">🚨 Critical light deficit</div><div style="font-size:13px;color:var(--text)">Projected DLI is more than 50% below target. Consider supplemental lighting if available, or adjust panel tilt to reduce shading during peak hours.</div></div>', unsafe_allow_html=True)
        elif deficit_pct > 20:
            st.markdown('<div class="action-warn"><div style="font-weight:700;color:#fbbf24;margin-bottom:4px">⚠️ Moderate light deficit</div><div style="font-size:13px;color:var(--text)">Switch to Vertical panel configuration to allow more diffuse light to reach the crop. Reduce irrigation by {:.0f}% as photosynthesis is reduced.'.format(min(40, deficit_pct)) + '</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="action-warn"><div style="font-weight:700;color:#fbbf24;margin-bottom:4px">⚠️ Minor light deficit</div><div style="font-size:13px;color:var(--text)">Monitor closely. If deficit persists, consider adjusting panel spacing to allow more sunlight through.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-item"><div style="font-weight:700;color:#4ade80;margin-bottom:4px">✅ Reduce irrigation</div><div style="font-size:13px;color:var(--text)">Less light means less photosynthesis and less water demand. Irrigation has been automatically reduced.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-item"><div style="font-weight:700;color:#4ade80;margin-bottom:4px">✅ Check panel orientation</div><div style="font-size:13px;color:var(--text)">Verify that panels are not causing excessive shading during the 10:00–14:00 peak sunlight window.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="action-item"><div style="font-weight:700;color:#4ade80;margin-bottom:4px">✅ Light levels are optimal</div><div style="font-size:13px;color:var(--text)">Crop is on track to meet its daily light target. No immediate action required.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-item"><div style="font-weight:700;color:#4ade80;margin-bottom:4px">✅ Maintain current configuration</div><div style="font-size:13px;color:var(--text)">Current panel setup is providing adequate PAR. Continue monitoring throughout the day.</div></div>', unsafe_allow_html=True)
        if dli_pct > 90:
            st.markdown('<div class="action-item"><div style="font-weight:700;color:#93c5fd;margin-bottom:4px">ℹ️ Approaching daily target</div><div style="font-size:13px;color:var(--text)">DLI target will likely be met before sunset. You may increase panel tilt slightly to boost energy output for the remainder of the day.</div></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Crop targets reference
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="lbl" style="margin-bottom:12px">CROP DLI TARGETS REFERENCE</div>', unsafe_allow_html=True)
    crops_ref = {"Lettuce":14,"Tomato":25,"Wheat":22}
    cols = st.columns(len(crops_ref), gap="small")
    for i,(c,t) in enumerate(crops_ref.items()):
        is_cur = c.lower() == d.get("crop","lettuce").lower()
        b = "2px solid #22c55e" if is_cur else "1px solid rgba(255,255,255,0.08)"
        bg = "rgba(34,197,94,0.07)" if is_cur else "rgba(255,255,255,0.02)"
        with cols[i]: st.markdown(f'<div style="border:{b};background:{bg};border-radius:10px;padding:10px;text-align:center"><div class="lbl">{c}</div><div style="font-weight:700;font-size:16px">{t}</div><div class="cap">mol/m²/day</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Refresh", key="dli_ref"): st.rerun()


# ── PAGE 4: IRRIGATION ────────────────────────────────────────────────────────
def page_irrigation():
    header("💧 Irrigation", "Automated irrigation management and recommendations")
    d, err = fetch_forecast()
    if err: show_err(err); return
    # Sync crop from backend so it matches AI Forecast page
    try:
        ai = api_get("/ai/status", timeout=5)
        d["crop"] = ai.get("crop", d.get("crop","lettuce"))
    except: pass
    run_status_bar(d)

    irr_pct   = d.get("irrigation_pct", 100)
    irr_factor= d.get("irrigation_factor", 1.0)
    dli_pct   = d.get("dli_pct", 0)
    dli_def   = d.get("dli_deficit", 0)
    dli_thresh= d.get("dli_threshold", 14.0)
    stressed  = d.get("stress_alert", False)
    crop      = d.get("crop","lettuce").capitalize()

    reduction = 100 - irr_pct
    ic = "#22c55e" if irr_pct>=90 else ("#f59e0b" if irr_pct>=70 else "#ef4444")
    cls = "card-green" if irr_pct>=90 else ("card-amber" if irr_pct>=70 else "card-red")

    # Status
    if irr_pct == 100:
        status_msg = "Irrigation running at full schedule. Crop is receiving adequate light and water demand is normal."
        status_title = "✅ Full irrigation active"
    elif irr_pct >= 80:
        status_msg = f"Irrigation reduced by {reduction}% due to lower crop light levels. Less photosynthesis means less water demand through the leaves."
        status_title = "⚠️ Irrigation slightly reduced"
    else:
        status_msg = f"Irrigation significantly reduced by {reduction}%. Crop light deficit is causing stomata to partially close, reducing water demand. This is the biologically correct response."
        status_title = "🚨 Irrigation significantly reduced"

    st.markdown(f'<div class="{cls}"><div style="font-size:15px;font-weight:700;margin-bottom:8px">{status_title}</div><div style="font-size:13px;color:var(--text);line-height:1.7">{status_msg}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Stats
    i1,i2,i3,i4 = st.columns(4, gap="medium")
    with i1: st.markdown(f'<div class="card"><div class="lbl">Current Schedule</div><div class="vlg" style="color:{ic}">{irr_pct}<span style="font-size:14px;color:var(--muted)">%</span></div>{pbar(irr_pct,ic)}</div>', unsafe_allow_html=True)
    with i2: st.markdown(f'<div class="card"><div class="lbl">Reduction</div><div class="vlg" style="color:{"#22c55e" if reduction==0 else "#f59e0b"}">{reduction}<span style="font-size:14px;color:var(--muted)">%</span></div><div class="cap">from normal schedule</div></div>', unsafe_allow_html=True)
    with i3: st.markdown(f'<div class="card"><div class="lbl">Irrigation Factor</div><div class="vlg">{irr_factor:.2f}</div><div class="cap">Jarvis 1976 model</div></div>', unsafe_allow_html=True)
    with i4: st.markdown(f'<div class="card"><div class="lbl">DLI Progress</div><div class="vlg">{dli_pct:.0f}<span style="font-size:14px;color:var(--muted)">%</span></div><div class="cap">drives irrigation level</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="lbl" style="margin-bottom:12px">HOW IRRIGATION IS CALCULATED</div>', unsafe_allow_html=True)
    st.markdown(f'''<div style="font-size:13px;color:var(--text);line-height:1.8">
    The irrigation system uses the <b>Jarvis (1976) stomatal conductance model</b>:<br><br>
    <code style="background:rgba(255,255,255,0.08);padding:8px 12px;border-radius:6px;display:block;margin:8px 0">
    factor = 1 − (deficit / target × 0.5) &nbsp;&nbsp; minimum = 0.60
    </code><br>
    When the crop has a DLI deficit of <b>{dli_def:.1f} mol/m²</b> against a target of <b>{dli_thresh:.0f} mol/m²</b>,
    photosynthesis slows down, stomata partially close, and less water is lost through the leaves.
    The system automatically reduces irrigation accordingly — never cutting more than 40%.
    </div>''', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Actions
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="lbl" style="margin-bottom:12px">RECOMMENDED ACTIONS</div>', unsafe_allow_html=True)
    if irr_pct == 100:
        st.markdown('<div class="action-item"><div style="font-weight:700;color:#4ade80;margin-bottom:4px">✅ No adjustment needed</div><div style="font-size:13px;color:var(--text)">Crop is receiving adequate light. Maintain full irrigation schedule.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-item"><div style="font-weight:700;color:#4ade80;margin-bottom:4px">✅ Monitor soil moisture</div><div style="font-size:13px;color:var(--text)">Check soil sensors if available to confirm moisture levels are in the optimal range for {}.'.format(crop) + '</div></div>', unsafe_allow_html=True)
    elif irr_pct >= 80:
        st.markdown('<div class="action-warn"><div style="font-weight:700;color:#fbbf24;margin-bottom:4px">⚠️ Minor reduction active</div><div style="font-size:13px;color:var(--text)">Irrigation reduced by {}%. Monitor crop health and revert to full schedule if wilting is observed.'.format(reduction) + '</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-item"><div style="font-weight:700;color:#4ade80;margin-bottom:4px">✅ Check for heat stress signs</div><div style="font-size:13px;color:var(--text)">Inspect leaf edges and color for early signs of water stress, especially during midday hours.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="action-red"><div style="font-weight:700;color:#f87171;margin-bottom:4px">🚨 Significant reduction active — monitor closely</div><div style="font-size:13px;color:var(--text)">Irrigation reduced by {}%. Visually inspect crops for wilting. If stress signs appear, override to manual full schedule.'.format(reduction) + '</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-warn"><div style="font-weight:700;color:#fbbf24;margin-bottom:4px">⚠️ Investigate light deficit cause</div><div style="font-size:13px;color:var(--text)">Check for panel misalignment, cloud cover, or shading issues that may be reducing crop light below target.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-item"><div style="font-weight:700;color:#4ade80;margin-bottom:4px">✅ Consider panel reconfiguration</div><div style="font-size:13px;color:var(--text)">Switching to Vertical panel orientation may improve crop light and help restore normal irrigation levels.</div></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Schedule visualizer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="lbl" style="margin-bottom:12px">IRRIGATION SCHEDULE — CURRENT VS NORMAL</div>', unsafe_allow_html=True)
    hours = [f"{h:02d}:00" for h in range(6,21)]
    normal = [100]*len(hours)
    actual = [irr_pct if 7<=h<=18 else 0 for h in range(6,21)]
    fig=go.Figure()
    fig.add_trace(go.Bar(name="Normal schedule",x=hours,y=normal,marker_color="rgba(255,255,255,0.15)",opacity=0.6))
    fig.add_trace(go.Bar(name="Current schedule",x=hours,y=actual,marker_color=ic,opacity=0.85))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.6)",family="DM Sans",size=11),xaxis=dict(gridcolor="rgba(255,255,255,0.08)",linecolor="rgba(255,255,255,0.08)",tickfont=dict(size=10)),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)),barmode="overlay",height=200,yaxis=dict(title="%",gridcolor="rgba(255,255,255,0.08)",range=[0,120]),margin=dict(l=8,r=8,t=8,b=8))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Refresh", key="irr_ref"): st.rerun()


# ── PAGE 5: DESIGN & COMPARE ──────────────────────────────────────────────────
def page_history():
    header("📈 History", "Last 24 hours of AI forecast records")
    try:
        hist=api_get("/history",timeout=10)
    except Exception as e:
        st.error("Could not load history."); st.caption(str(e)); return

    records=hist.get("records",[])
    if len(records)<2:
        st.info("Not enough history yet — data saves every 30 min. Keep the site open to accumulate records.")
        return

    df=pd.DataFrame(records)
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    df=df.sort_values("timestamp")
    df["time"]=df["timestamp"].dt.strftime("%H:%M")

    def hchart(y,name,color,fill_color,height=200):
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df["time"],y=df[y],fill="tozeroy",fillcolor=fill_color,line=dict(color=color,width=2),mode="lines+markers",marker=dict(size=4)))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.6)",family="DM Sans",size=11),xaxis=dict(gridcolor="rgba(255,255,255,0.08)",linecolor="rgba(255,255,255,0.08)",tickfont=dict(size=10)),yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),margin=dict(l=8,r=8,t=8,b=8),height=height)
        return fig

    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="card"><div class="lbl" style="margin-bottom:8px">PV PEAK POWER (kW)</div>', unsafe_allow_html=True)
        st.plotly_chart(hchart("pv_peak_kw","PV","#22c55e","rgba(34,197,94,0.1)"),use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="lbl" style="margin-bottom:8px">CROP LIGHT PAR</div>', unsafe_allow_html=True)
        st.plotly_chart(hchart("par_mean","PAR","#f59e0b","rgba(245,158,11,0.1)"),use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3,c4=st.columns(2,gap="large")
    with c3:
        st.markdown('<div class="card"><div class="lbl" style="margin-bottom:8px">DLI ACCUMULATED (mol/m²)</div>', unsafe_allow_html=True)
        st.plotly_chart(hchart("dli_accumulated","DLI","#60a5fa","rgba(96,165,250,0.1)"),use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card"><div class="lbl" style="margin-bottom:8px">IRRIGATION SCHEDULE (%)</div>', unsafe_allow_html=True)
        fig4=go.Figure()
        fig4.add_trace(go.Scatter(x=df["time"],y=df["irrigation_pct"],fill="tozeroy",fillcolor="rgba(167,139,250,0.1)",line=dict(color="#a78bfa",width=2),mode="lines+markers",marker=dict(size=4)))
        fig4.add_hline(y=100,line=dict(color="rgba(255,255,255,0.2)",dash="dot",width=1))
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="rgba(255,255,255,0.6)",family="DM Sans",size=11),xaxis=dict(gridcolor="rgba(255,255,255,0.08)",linecolor="rgba(255,255,255,0.08)",tickfont=dict(size=10)),yaxis=dict(gridcolor="rgba(255,255,255,0.08)",range=[50,105]),margin=dict(l=8,r=8,t=8,b=8),height=200)
        st.plotly_chart(fig4,use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="lbl" style="margin-bottom:12px">PERIOD SUMMARY</div>', unsafe_allow_html=True)
    s1,s2,s3,s4=st.columns(4,gap="medium")
    s1.metric("Avg PV peak",f'{df["pv_peak_kw"].mean():.1f} kW')
    s2.metric("Avg PAR",f'{df["par_mean"].mean():.0f}')
    s3.metric("Max DLI",f'{df["dli_accumulated"].max():.1f} mol/m²')
    s4.metric("Stress events",str(int(df["stress_alert"].sum())))
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Refresh history",key="hist_ref"): st.rerun()


def main():
    st.set_page_config(page_title="Agrivoltaic Dashboard", layout="wide")

    if not check_auth():
        auth_page()
        return

    # Logged in — show logout button in sidebar-like top area
    top_left, top_right = st.columns([8, 2])
    with top_right:
        user_email = st.session_state.get("user_email", "")
        st.markdown(f'<div style="text-align:right;font-size:12px;color:rgba(255,255,255,0.45);padding:4px 0">👤 {user_email}</div>', unsafe_allow_html=True)
        if st.button("Logout", key="logout_btn"):
            st.session_state["authenticated"] = False
            st.session_state["user_email"] = ""
            st.rerun()

    tabs=st.tabs(["Overview","AI Forecast","DLI","Irrigation","History"])
    with tabs[0]: page_overview()
    with tabs[1]: page_forecast()
    with tabs[2]: page_dli()
    with tabs[3]: page_irrigation()
    with tabs[4]: page_history()

if __name__=="__main__":
    main()
