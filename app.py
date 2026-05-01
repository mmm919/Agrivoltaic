def page_overview():
    header("🌱 Overview", "At a glance farm summary")

    try:
        ai = api_get("/ai/status", timeout=5)
        alpha_val = ai.get("alpha", st.session_state.get("fc_alpha", 0.7))
        current_crop = ai.get("crop", "lettuce")
    except:
        alpha_val = st.session_state.get("fc_alpha", 0.7)
        current_crop = "lettuce"

    comp, d, err = fetch_live(alpha_val)
    if err:
        show_err(err)
        return

    rec_cfg = comp.get("recommended_config", "—").replace("Fixedtilt", "Fixed-tilt")
    fc = comp.get("vertical_forecast", {}) if rec_cfg == "Vertical" else comp.get("fixed_forecast", {})

    stressed = d.get("stress_alert", False)
    dli_pct = d.get("dli_pct", 0)
    dli_def = d.get("dli_deficit", 0)
    irr_pct = d.get("irrigation_pct", 100)

    # ── 4 KPI cards ───────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:
        st.markdown(
            f'''
            <div class="card">
                <div class="lbl">PV Peak Power</div>
                <div class="vlg">{fc.get("pv_peak_kw", 0):.1f}
                    <span style="font-size:14px;color:var(--muted)">kW</span>
                </div>
                <div class="cap">Next 60 min forecast</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f'''
            <div class="card">
                <div class="lbl">Crop Light PAR</div>
                <div class="vlg">{fc.get("par_mean", 0):.0f}
                    <span style="font-size:14px;color:var(--muted)">umol/s/m²</span>
                </div>
                <div class="cap">Mean over next hour</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with k3:
        dli_color = "#22c55e" if not stressed else "#ef4444"
        badge = '<span class="badge bg">On track</span>' if not stressed else '<span class="badge br">Stress</span>'
        st.markdown(
            f'''
            <div class="card">
                <div class="lbl">DLI — {current_crop.capitalize()}</div>
                <div class="vlg" style="color:{dli_color}">{dli_pct:.0f}
                    <span style="font-size:14px;color:var(--muted)">%</span>
                </div>
                {pbar(dli_pct, dli_color)}
                <div style="margin-top:6px">{badge}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with k4:
        irr_color = "#22c55e" if irr_pct >= 90 else ("#f59e0b" if irr_pct >= 70 else "#ef4444")
        st.markdown(
            f'''
            <div class="card">
                <div class="lbl">Irrigation</div>
                <div class="vlg" style="color:{irr_color}">{irr_pct}
                    <span style="font-size:14px;color:var(--muted)">%</span>
                </div>
                {pbar(irr_pct, irr_color)}
                <div class="cap">of normal schedule</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── timestamp bar ─────────────────────────────────────────────────────────
    if d:
        run_status_bar(d)

    # ── quick area statuses ───────────────────────────────────────────────────
    if stressed:
        dli_status = f"DLI is below target with a deficit of {dli_def:.1f} mol/m²."
        dli_class = "card-red"
        dli_icon = "⚠️"
    else:
        dli_status = "DLI is on track for the selected crop."
        dli_class = "card-green"
        dli_icon = "✅"

    reduction = 100 - irr_pct
    if irr_pct >= 90:
        irr_status = "Irrigation is running close to the normal schedule."
        irr_class = "card-green"
        irr_icon = "✅"
    elif irr_pct >= 70:
        irr_status = f"Irrigation is reduced by {reduction}% due to crop light conditions."
        irr_class = "card-amber"
        irr_icon = "⚠️"
    else:
        irr_status = f"Irrigation is strongly reduced by {reduction}%, monitor crop condition."
        irr_class = "card-red"
        irr_icon = "🚨"

    reason = comp.get("reason", "")
    if not reason:
        reason = "Current configuration is selected from the crop light and energy balance."

    panel_status = f"{rec_cfg} is currently recommended."
    panel_class = "card-green"
    panel_icon = "🧭"

    s1, s2, s3 = st.columns(3, gap="medium")

    with s1:
        st.markdown(
            f'''
            <div class="{dli_class}">
                <div style="font-size:15px;font-weight:700;margin-bottom:6px">
                    {dli_icon} DLI status
                </div>
                <div style="font-size:13px;color:var(--text);line-height:1.6">
                    {dli_status}
                </div>
                <div class="cap">Open the DLI tab for details.</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f'''
            <div class="{irr_class}">
                <div style="font-size:15px;font-weight:700;margin-bottom:6px">
                    {irr_icon} Irrigation status
                </div>
                <div style="font-size:13px;color:var(--text);line-height:1.6">
                    {irr_status}
                </div>
                <div class="cap">Open the Irrigation tab for details.</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with s3:
        st.markdown(
            f'''
            <div class="{panel_class}">
                <div style="font-size:15px;font-weight:700;margin-bottom:6px">
                    {panel_icon} Panel configuration
                </div>
                <div style="font-size:13px;color:var(--text);line-height:1.6">
                    {panel_status}
                </div>
                <div class="cap">Open AI Forecast for charts and comparison.</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 Refresh", key="ov_ref"):
        st.rerun()
