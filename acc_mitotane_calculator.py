"""
ACC Adjuvant Mitotane Benefit Calculator
Doubly-Robust IPTW Cox model (development n=755, external validation n=97).
Scope: ENSAT stage I-III, resection R0/RX/R1. NOT validated for ENSAT IV or R2.

Individualized estimates: survival probability, absolute treatment benefit (CATE),
counterfactual survival, and Number Needed to Treat (NNT).
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "cox_calculator_params.json")))
# Pre-computed 95% CI for the treatment benefit (CATE), bootstrap B=500 on the
# weighted Cox model. The benefit is far more precisely estimated than each arm's
# absolute survival because both curves share the same baseline hazard and
# covariate coefficients, differing only in the treatment term.
CATE_CI = json.load(open(os.path.join(HERE, "cox_cate_ci.json")))


def cate_ci(age, sympt, ensat, rstatus, ki67, hz_key):
    key = f"{age}-{sympt}-{ensat}-{rstatus}-{ki67}"
    b = CATE_CI[key][hz_key]
    return b["cate"], b["lo"], b["hi"]


# ---------- Cox engine (pure Python, validated identical to R to 1e-8) ----------
def _lp(ep, age, sympt, ensat, rstatus, ki67, mitotane):
    c = P[ep]["coef"]; mu = P[ep]["means"]
    x = {"Mitotane_adjuvant": mitotane, "Age_f1": int(age == 1), "Sympt_f1": int(sympt == 1),
         "ENSAT_f1": int(ensat == 1), "R_f1": int(rstatus == 1), "R_f2": int(rstatus == 2),
         "Ki67_f1": int(ki67 == 1), "Ki67_f2": int(ki67 == 2)}
    return sum(c[k] * (x[k] - mu[k]) for k in c)

def surv_curve(ep, mitotane, **kw):
    tg = np.array(P[ep]["time"]); H = np.array(P[ep]["cumhaz"])
    return tg, np.exp(-H * np.exp(_lp(ep, mitotane=mitotane, **kw)))

def surv_at(ep, t, mitotane, **kw):
    tg, S = surv_curve(ep, mitotane, **kw)
    return float(S[np.searchsorted(tg, t, side="right") - 1]) if t >= tg[0] else 1.0

def sgras(age, sympt, ensat, rstatus, ki67):
    s = age + sympt + ensat + rstatus + ki67
    grp = ("Low" if s <= 1 else "Intermediate" if s <= 3 else "High" if s <= 5 else "Very high")
    return s, grp

def render_sgras_badge(score, grp):
    # Set colors based on group matching the visual specs in user screenshot
    if grp == "Low":
        circle_emoji = "🟢"
        text_color = "#166534" # dark green
        border_color = "#22c55e" # green-500
        bg_color = "#e8f5e9" # soft green background
        risk_label = "LOW RISK"
    elif grp == "Intermediate":
        circle_emoji = "🟡"
        text_color = "#854d0e" # dark yellow
        border_color = "#eab308" # yellow-500
        bg_color = "#fefde8" # soft yellow background
        risk_label = "INTERMEDIATE RISK"
    elif grp == "High":
        circle_emoji = "🟠"
        text_color = "#9a3412" # dark orange
        border_color = "#f97316" # orange-500
        bg_color = "#fff7ed" # soft orange background
        risk_label = "HIGH RISK"
    else:
        circle_emoji = "🔴"
        text_color = "#991b1b" # dark red
        border_color = "#ef4444" # red-500
        bg_color = "#fef2f2" # soft red background
        risk_label = "VERY HIGH RISK"
        
    # Build the horizontal 0-9 scale HTML
    scale_html = ""
    for i in range(10):
        # Determine background color for scale block
        if i <= 1:
            block_bg = "#a7f3d0" # green
        elif i <= 3:
            block_bg = "#fef08a" # yellow
        elif i <= 5:
            block_bg = "#fed7aa" # orange
        else:
            block_bg = "#fecaca" # red
            
        # Highlight the current score block with a thick black border
        if i == score:
            block_style = f"background-color: {block_bg}; border: 2px solid #000000; font-weight: 700; scale: 1.05;"
        else:
            block_style = f"background-color: {block_bg}; border: 1px solid rgba(0,0,0,0.05); font-weight: 400; opacity: 0.85;"
            
        scale_html += f"<div style='width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-size: 14px; color: #1f2937; {block_style}'>{i}</div>"
        
    # Build the main card HTML
    html = f"""
    <div style='display: flex; flex-direction: column; align-items: center; margin: 10px 0 24px 0;'>
        <div style='border: 2px solid {border_color}; background-color: {bg_color}; border-radius: 14px; width: 320px; padding: 20px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);'>
            <div style='display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 8px;'>
                <span style='font-size: 24px;'>{circle_emoji}</span>
                <span style='font-size: 22px; font-weight: 600; color: {text_color};'>S-GRAS Score:</span>
            </div>
            <div style='font-size: 42px; font-weight: 800; color: {text_color}; margin: 8px 0; line-height: 1;'>{score}</div>
            <div style='font-size: 18px; font-weight: 700; color: {text_color}; letter-spacing: 0.05em; margin-top: 12px;'>{risk_label}</div>
        </div>
        <div style='display: flex; justify-content: center; gap: 5px; margin-top: 14px; width: 100%; max-width: 380px;'>
            {scale_html}
        </div>
    </div>
    """
    return html


# ---------- UI & Styling ----------
st.set_page_config(page_title="ACC Mitotane Benefit Calculator", layout="wide")

# Custom CSS for Premium, Ultra-Compact & Responsive Design
st.markdown(
    """
    <style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default Streamlit top padding and expand container safely below the top menu bar */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 95% !important;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-bottom: 2px;
        line-height: 1.1;
    }
    
    .main-caption {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 14px;
    }
    
    /* Custom metric card - Fully Responsive Flexbox */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 12px;
        width: 100%;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        flex: 1 1 180px;
        min-width: 150px;
    }
    
    /* NNT Card Styling */
    .nnt-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 12px 0 16px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        width: 100%;
    }
    
    /* Make standard Streamlit widgets look cleaner and tighter */
    div[data-testid="stRadio"] {
        margin-bottom: 0px !important;
    }
    div[data-testid="stRadio"] > label, div[data-testid="stSelectbox"] label {
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: #475569 !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 8px !important;
    }
    
    /* Custom spacing for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Custom Tooltip Styling */
    .custom-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #94a3b8;
        font-size: 14px;
    }
    
    .custom-tooltip .tooltip-text {
        visibility: hidden;
        width: 280px;
        background-color: #1e293b;
        color: #ffffff;
        text-align: left;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 11px;
        line-height: 1.4;
        font-weight: 400;
        
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        right: 0px;
        
        opacity: 0;
        transition: opacity 0.05s ease-in-out;
    }
    
    .custom-tooltip .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        right: 8px;
        border-width: 5px;
        border-style: solid;
        border-color: #1e293b transparent transparent transparent;
    }
    
    .custom-tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">Adjuvant Mitotane Benefit Calculator — Adrenocortical Carcinoma</div>
    <div class="main-caption">Doubly-Robust IPTW Cox model · development n=755, external validation n=97 · scope: ENSAT I–III, R0/RX/R1, age ≥18 y</div>
    """,
    unsafe_allow_html=True
)

# S-GRAS components inside a collapsible expander at the top in a single row
with st.expander("Patient Characteristics & Tumor Parameters", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        age = st.radio(
            "Age at initial diagnosis",
            [0, 1],
            format_func=lambda x: "<50 years" if x == 0 else "≥50 years",
            help="Patient chronological age at the time of adrenocortical carcinoma diagnosis"
        )
    with c2:
        sympt = st.radio(
            "Clinical presentation",
            [0, 1],
            format_func=lambda x: "Asymptomatic" if x == 0 else "Symptomatic",
            help="Presence of hormone-related, tumor-related, or systemic manifestations at initial presentation"
        )
    with c3:
        ensat = st.selectbox(
            "ENSAT Tumor Stage",
            [0, 1],
            format_func=lambda x: "Stage I-II" if x == 0 else "Stage III",
            help="European Network for the Study of Adrenal Tumors staging classification system"
        )
    with c4:
        rstatus = st.selectbox(
            "Surgical resection status",
            [0, 1, 2],
            format_func=lambda x: {0: "R0", 1: "RX", 2: "R1"}[x],
            help="R0: No residual tumor; RX: Presence of residual tumor cannot be assessed; R1: Microscopic residual disease; R2: Macroscopic residual disease"
        )
    with c5:
        ki67 = st.selectbox(
            "Ki-67 proliferation index",
            [0, 1, 2],
            format_func=lambda x: {0: "0%-9%", 1: "10%-19%", 2: "≥20%"}[x],
            help="Immunohistochemical proliferation marker representing the percentage of tumor cells in active phases of cell cycle"
        )

kw = dict(age=age, sympt=sympt, ensat=ensat, rstatus=rstatus, ki67=ki67)
score, grp = sgras(age, sympt, ensat, rstatus, ki67)


def render_endpoint_results(ep, hz, kw):
    yr_lbl = hz // 12
    if ep == "OS":
        lbl = f"{yr_lbl}-year Overall survival"
    else:
        lbl = f"{yr_lbl}-year Progression-free survival"
        
    col_left, col_right = st.columns([1, 1.3], gap="large")
    
    with col_left:
        st.markdown(f"<div style='font-size: 1.15rem; font-weight: 600; color: #0f172a; margin: 12px 0 16px 0;'>{lbl}</div>", unsafe_allow_html=True)
        
        s0 = surv_at(ep, hz, 0, **kw); s1 = surv_at(ep, hz, 1, **kw)
        rec = CATE_CI[f"{kw['age']}-{kw['sympt']}-{kw['ensat']}-{kw['rstatus']}-{kw['ki67']}"][f"{ep}{hz}"]
        c_pt, c_lo, c_hi = rec["cate"], rec["lo"], rec["hi"]
        nnt, nnt_lo, nnt_hi = rec["nnt"], rec.get("nnt_lo"), rec.get("nnt_hi")
        yr = hz // 12
        
        # Responsive metrics
        st.markdown(
            f"<div class='metric-container'>"
            f"  <div class='metric-card' style='border-left: 4px solid #ef4444;'>"
            f"    <div style='font-size: 10px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>{ep} without mitotane</div>"
            f"    <div style='font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 4px;'>{s0*100:.1f}%</div>"
            f"  </div>"
            f"  <div class='metric-card' style='border-left: 4px solid #3b82f6;'>"
            f"    <div style='font-size: 10px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>{ep} with mitotane</div>"
            f"    <div style='font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 4px;'>{s1*100:.1f}%</div>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        nnt_ci = (f"{nnt_lo}–{nnt_hi}" if nnt_lo and nnt_hi else "—")
        outcome = "survivor" if ep == "OS" else "progression-free patient"
        tip = (f"One additional {yr}-year {outcome} is expected for every {nnt} patients "
               f"treated with adjuvant mitotane. NNT = 1 / absolute benefit; the 95% CI is "
               f"obtained by inverting the bounds of the absolute-benefit CI (Altman 1998). "
               f"Confidence intervals are from a 1000-replicate bootstrap of the "
               f"doubly-robust IPTW Cox model.")
        
        st.markdown(
            f"<div class='nnt-card'>"
            f"  <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; flex-wrap: wrap; gap: 6px;'>"
            f"    <span style='font-size: 12px; font-weight: 500; color: #475569;'>Number needed to treat ({yr}-year {ep})</span>"
            f"    <div class='custom-tooltip'>&#9432;<span class='tooltip-text'>{tip}</span></div>"
            f"  </div>"
            f"  <div style='display: flex; align-items: baseline; gap: 6px; margin-bottom: 2px; flex-wrap: wrap;'>"
            f"    <span style='font-size: 34px; font-weight: 700; color: #1e3a8a; line-height: 1;'>{nnt}</span>"
            f"    <span style='font-size: 14px; font-weight: 500; color: #1e40af;'>(95% CI {nnt_ci})</span>"
            f"  </div>"
            f"  <div style='font-size: 12px; color: #475569; margin-top: 4px;'>"
            f"    Absolute {ep} benefit <b>{c_pt:+.1f}%</b> (95% CI {c_lo:+.1f} to {c_hi:+.1f} pp)"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    with col_right:
        # Interactive Plotly Step Chart
        tg, S0 = surv_curve(ep, 0, **kw); _, S1 = surv_curve(ep, 1, **kw)
        mask = tg <= hz
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tg[mask],
            y=S0[mask] * 100,
            line=dict(shape="hv", color="#ef4444", width=2.5),
            name="No mitotane",
            mode="lines",
            hovertemplate="No mitotane: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=tg[mask],
            y=S1[mask] * 100,
            line=dict(shape="hv", color="#3b82f6", width=2.5),
            name="Adj. mitotane",
            mode="lines",
            hovertemplate="Adj. mitotane: %{y:.1f}%<extra></extra>"
        ))
        
        fig.update_layout(
            xaxis_title="Months",
            yaxis_title=f"{ep} probability (%)",
            yaxis=dict(range=[0, 100], gridcolor="#f1f5f9", zeroline=False),
            xaxis=dict(range=[0, hz], gridcolor="#f1f5f9", zeroline=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=10, t=10, b=40),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            ),
            hovermode="x unified",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# Header above the S-GRAS Risk Card
st.markdown("<div style='font-size: 1.35rem; font-weight: 700; color: #0f172a; margin: 18px 0 10px 0;'>📊 S-GRAS Risk Stratification Assessment</div>", unsafe_allow_html=True)
st.markdown(render_sgras_badge(score, grp), unsafe_allow_html=True)

# 3 tabs for OS, 1-year PFS, 3-year PFS
tab_os, tab_pfs1, tab_pfs3 = st.tabs([
    "5-year Overall Survival (OS)", 
    "1-year Progression-Free Survival (PFS)", 
    "3-year Progression-Free Survival (PFS)"
])

with tab_os:
    render_endpoint_results("OS", 60, kw)
    
with tab_pfs1:
    render_endpoint_results("PFS", 12, kw)
    
with tab_pfs3:
    render_endpoint_results("PFS", 36, kw)

hr = P["meta"]["HR"]
st.divider()
st.markdown(
    f"**Model summary.** Adjusted hazard ratio for adjuvant mitotane: "
    f"OS {hr['OS'][0]} (95% CI {hr['OS'][1]}–{hr['OS'][2]}); "
    f"PFS {hr['PFS'][0]} (95% CI {hr['PFS'][1]}–{hr['PFS'][2]}). "
    f"Doubly-robust IPTW Cox on the five S-GRAS components."
)

# ---------- Model performance (development vs external validation) ----------
with st.expander("Model performance — development vs. external validation"):
    st.markdown(
        "Development cohort: **n = 755** (multicentre S-GRAS cohort, ENSAT I–III, R0/RX/R1). "
        "External validation cohort: **n = 97** (independent, later-recruited patients). "
        "Discrimination is reported as Uno's censoring-adjusted C-index (primary; robust to "
        "differing follow-up between cohorts) with Harrell's C shown alongside, and as "
        "time-dependent AUC at clinically relevant horizons.")
    perf = pd.DataFrame({
        "Endpoint": ["OS", "OS", "PFS", "PFS"],
        "Cohort": ["Development (n=755)", "External (n=97)",
                   "Development (n=755)", "External (n=97)"],
        "Uno C-index": ["0.74", "0.62 (95% CI 0.52–0.71)",
                        "0.72", "0.65 (95% CI 0.58–0.73)"],
        "Harrell C-index": ["0.74", "0.63", "0.71", "0.64"],
    })
    st.table(perf)
    auc = pd.DataFrame({
        "Endpoint / horizon": ["OS, 5-year (60 mo)", "PFS, 1-year (12 mo)",
                               "PFS, 3-year (36 mo)"],
        "Development AUC": ["0.78", "0.75", "0.77"],
        "External AUC (95% CI)": ["0.65 (0.52–0.76)", "0.70 (0.59–0.80)",
                                  "0.71 (0.59–0.82)"],
    })
    st.table(auc)
    st.caption(
        "Time-dependent AUC via cumulative/dynamic ROC (Uno estimator); external 95% CIs "
        "by 1000-fold bootstrap. The development-to-external decline reflects case-mix "
        "transportability (the external cohort is more advanced and event-dense), not "
        "internal overfitting — bootstrap optimism was ≤0.007 (development-corrected C-index "
        "OS 0.730, PFS 0.710).")

st.divider()
st.error(
    "**FOR RESEARCH AND EDUCATIONAL USE ONLY — NOT A MEDICAL DEVICE.** "
    "This calculator is an investigational, research-grade decision-support prototype. "
    "It has **not** been reviewed, cleared, or approved by the U.S. Food and Drug "
    "Administration (FDA), the European Medicines Agency (EMA), or any other regulatory "
    "authority, and it is **not** intended for the diagnosis, treatment, cure, mitigation, "
    "or prevention of disease in individual patients.")
st.warning(
    "**Important limitations.** Estimates are population-level, model-based projections "
    "derived from a retrospective observational cohort and carry irreducible statistical "
    "uncertainty (see confidence intervals). They do **not** constitute medical advice and "
    "must **not** be used as the sole basis for any clinical decision. Individual treatment "
    "decisions require a qualified physician integrating the full clinical context, "
    "comorbidities, patient preferences, and current guidelines. The model is valid **only** "
    "for adrenocortical carcinoma of ENSAT stage I–III with R0/RX/R1 resection; it has not "
    "been validated for ENSAT IV or R2 disease. Causal benefit estimates rest on the "
    "no-unmeasured-confounding assumption inherent to observational data.")
st.caption(
    "Model: doubly-robust IPTW Cox on the five S-GRAS components · development n=755, "
    "external validation n=97 · random seed 20260717 · © 2026 research use. "
    "Estimates are provided for research purposes only; their accuracy and generalisability "
    "beyond the validated population have not been established, and any interpretation "
    "remains the responsibility of a qualified investigator.")
