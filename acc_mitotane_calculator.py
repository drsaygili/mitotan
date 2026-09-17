"""
ACC Adjuvant Mitotane Benefit Calculator
Doubly-Robust IPTW Cox model, analytic cohort n=852.
Internally validated (bootstrap, B=1000). External validation pending.
Prognostic coefficients carry a uniform shrinkage factor; the treatment
coefficient is not shrunk.
Scope: ENSAT stage I-III, resection R0/RX/R1. NOT valid for ENSAT IV or R2.

Reports estimated quantities only — survival probability, absolute treatment
benefit (CATE), counterfactual survival, and Number Needed to Treat (NNT),
each with its uncertainty interval. It issues no treatment recommendation.
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
_load_error = None
try:
    P = json.load(open(os.path.join(HERE, "cox_calculator_params.json")))
# Pre-computed 95% CI for the treatment benefit (CATE) and for each arm's
# absolute survival. Both come from the same 1000-replicate bootstrap of the
# pipeline: every replicate re-estimates the propensity model, the stabilized
# weights, the Cox models and the baseline hazard, and applies the shrinkage
# factor estimated in the full cohort (that factor is not itself re-estimated
# per replicate, so its uncertainty is not propagated). Percentile intervals.
    CATE_CI = json.load(open(os.path.join(HERE, "cox_cate_ci.json")))
    ARM_CI = json.load(open(os.path.join(HERE, "cox_arm_ci.json")))
except (OSError, ValueError) as _e:
    P = CATE_CI = ARM_CI = None
    _load_error = str(_e)


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
            
        scale_html += f"<div style='flex: 1 1 0; min-width: 0; max-width: 30px; height: 28px; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-size: 13px; color: #1f2937; {block_style}'>{i}</div>"
        
    # Build the main card HTML (Short, Rectangular and Compact!)
    html = f"""
    <div style='display: flex; flex-direction: column; align-items: center; margin: 4px 0 12px 0;'>
        <div style='border: 1.5px solid {border_color}; background-color: {bg_color}; border-radius: 10px; width: 100%; max-width: 340px; box-sizing: border-box; padding: 10px 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; row-gap: 4px; column-gap: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);'>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='font-size: 20px; line-height: 1;'>{circle_emoji}</span>
                <span style='font-size: 15px; font-weight: 600; color: {text_color};'>S-GRAS Score:</span>
                <span style='font-size: 22px; font-weight: 800; color: {text_color}; line-height: 1;'>{score}</span>
            </div>
            <div style='font-size: 13px; font-weight: 700; color: {text_color}; letter-spacing: 0.03em;'>{risk_label}</div>
        </div>
        <div style='display: flex; justify-content: center; gap: 4px; margin-top: 10px; width: 100%; max-width: 340px;'>
            {scale_html}
        </div>
    </div>
    """
    return html



# ---------- UI & Styling ----------
_ICON = os.path.join(HERE, "assets", "icon.png")
st.set_page_config(page_title="ACC Mitotane Benefit Calculator", layout="wide",
                   page_icon=_ICON if os.path.exists(_ICON) else None)

if _load_error is not None:
    st.error(
        "The model parameter files could not be loaded, so no estimates can be shown. "
        "Please report this to the investigators named at the foot of this page.")
    st.stop()

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

    /* Hide Streamlit footer and host badge */
    footer {
        visibility: hidden !important;
        height: 0px !important;
        padding: 0px !important;
    }
    div[data-testid="stFooter"] {
        display: none !important;
    }
    
    /* Hide 'Hosted with Streamlit' red banner */
    div[class*="viewerBadge"] {
        display: none !important;
    }
    span[class*="viewerBadge"] {
        display: none !important;
    }
    
    /* Hide viewer profile icon / app toolbar at the bottom */
    div[class*="StyledAppToolbar"] {
        display: none !important;
    }
    div[data-testid="stViewerToolbar"] {
        display: none !important;
    }
    div[class*="viewerToolbar"] {
        display: none !important;
    }

    /* Notice bar directly under the title */
    .top-notice {
        background-color: #fff7ed;
        border: 1px solid #fdba74;
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.80rem;
        line-height: 1.35;
        color: #7c2d12;
        margin: 0 0 14px 0;
    }

    /* ---------------- Phones ---------------- */
    @media (max-width: 640px) {
        .block-container {
            padding-top: 2.75rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: 100% !important;
        }
        .main-title  { font-size: 1.40rem; line-height: 1.15; }
        .title-mark  { width: 34px; height: 34px; flex: 0 0 auto; }
        .main-caption { font-size: 0.72rem; margin-bottom: 10px; }
        .top-notice  { font-size: 0.74rem; padding: 7px 10px; }
        .metric-container { gap: 8px; }
        .metric-card { flex: 1 1 100%; min-width: 0; padding: 10px 12px; }
        .nnt-card { padding: 14px 14px; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            height: 38px; padding: 6px 10px; font-size: 0.82rem;
        }
        /* Hover tooltips never fire on touch screens; the explanations live in the
           "How to read these numbers" panel instead, so hide the dead icon. */
        .custom-tooltip { display: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo mark: the two diverging counterfactual survival curves the app estimates.
# Inlined rather than loaded from a file so it costs no extra request and no vertical
# space on a phone; it sits on the same row as the title.
LOGO_SVG = (
    '<svg class="title-mark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" '
    'width="46" height="46" role="img" aria-label="Two diverging survival curves">'
    '<rect x="1.2" y="1.2" width="45.6" height="45.6" rx="10" fill="#ffffff" '
    'stroke="#cbd5e1" stroke-width="1.6"/>'
    '<polyline points="9,14 14,14 14,21 21,21 21,28 29,28 29,34 39,34" fill="none" '
    'stroke="#ef4444" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '<polyline points="9,14 17,14 17,18 25,18 25,22 33,22 33,25 39,25" fill="none" '
    'stroke="#3b82f6" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 2px;">
      {LOGO_SVG}
      <div class="main-title">Adjuvant Mitotane Benefit Calculator — Adrenocortical Carcinoma</div>
    </div>
    <div class="main-caption">Doubly-Robust IPTW Cox model · analytic cohort n=852 · internally validated (bootstrap) · scope: ENSAT I–III, R0/RX/R1, age ≥18 y</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div class='top-notice'><b>Research use only — not a medical device.</b> "
    "For clinicians and researchers; not intended for patients. Every estimate below "
    "carries substantial uncertainty — read the interval next to each number, not the "
    "number alone. Full limitations at the foot of this page.</div>",
    unsafe_allow_html=True
)

# S-GRAS components inside a collapsible expander at the top in a single row
with st.expander("Patient Characteristics & Tumor Parameters", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        age = st.selectbox(
            "Age at initial diagnosis",
            [0, 1],
            format_func=lambda x: "<50 years" if x == 0 else "≥50 years",
            help="Patient chronological age at the time of adrenocortical carcinoma diagnosis"
        )
    with c2:
        sympt = st.selectbox(
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
    st.caption(
        "All five components must be known. The model has no category for an unknown "
        "Ki-67, and Ki-67 shifts the estimate more than any other variable, so a guessed "
        "value produces a misleading result. Resection status does have an explicit RX "
        "category for 'cannot be assessed'.")

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
        
        # Pointwise 95% CI for each arm's absolute survival
        arm = ARM_CI[f"{kw['age']}-{kw['sympt']}-{kw['ensat']}-{kw['rstatus']}-{kw['ki67']}"][f"{ep}{hz}"]
        arm_tip = ("95% bootstrap percentile interval for the absolute survival "
                   "probability, from 1000 replicates in which the propensity model, "
                   "the weights, the Cox fit and the baseline hazard were all "
                   "re-estimated. The shrinkage factor is held fixed at its "
                   "full-cohort value, so its own uncertainty is not included.")
        
        # Responsive metrics
        st.markdown(
            f"<div class='metric-container'>"
            f"  <div class='metric-card' style='border-left: 4px solid #ef4444;' title='{arm_tip}'>"
            f"    <div style='font-size: 10px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>{ep} without mitotane</div>"
            f"    <div style='font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 4px;'>{s0*100:.1f}%</div>"
            f"    <div style='font-size: 11px; color: #64748b; margin-top: 2px;'>95% CI {arm['s0_lo']:.1f}%–{arm['s0_hi']:.1f}%</div>"
            f"  </div>"
            f"  <div class='metric-card' style='border-left: 4px solid #3b82f6;' title='{arm_tip}'>"
            f"    <div style='font-size: 10px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>{ep} with mitotane</div>"
            f"    <div style='font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 4px;'>{s1*100:.1f}%</div>"
            f"    <div style='font-size: 11px; color: #64748b; margin-top: 2px;'>95% CI {arm['s1_lo']:.1f}%–{arm['s1_hi']:.1f}%</div>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        nnt_ci = (f"{nnt_lo}–{nnt_hi}" if nnt_lo and nnt_hi else "—")
        outcome = "survivor" if ep == "OS" else "progression-free patient"
        tip = (f"Model-estimated: one additional {yr}-year {outcome} per {nnt} patients "
               f"treated with adjuvant mitotane. NNT = 1 / absolute benefit; the 95% CI is "
               f"obtained by inverting the bounds of the absolute-benefit CI (Altman 1998). "
               f"Intervals are 1000-replicate bootstrap percentiles of the complete "
               f"doubly-robust IPTW Cox pipeline. This is an estimate, not a recommendation.")
        
        st.markdown(
            f"<div class='nnt-card'>"
            f"  <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; flex-wrap: wrap; gap: 6px;'>"
            f"    <span style='font-size: 12px; font-weight: 500; color: #475569;'>Number needed to treat ({yr}-year {ep})</span>"
            f"    <div class='custom-tooltip'>&#9432;<span class='tooltip-text'>{tip}</span></div>"
            f"  </div>"
            f"  <div style='display: flex; align-items: baseline; gap: 6px; margin-bottom: 2px; flex-wrap: wrap;'>"
            f"    <span style='font-size: 34px; font-weight: 700; color: #1e3a8a; line-height: 1;'>{nnt}</span>"
            f"    <span style='font-size: 20px; font-weight: 600; color: #1e40af;'>(95% CI {nnt_ci})</span>"
            f"  </div>"
            f"  <div style='font-size: 13.5px; color: #334155; margin-top: 6px;'>"
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
        
        # Horizon marker plus the 95% interval at that horizon, so the two clean
        # lines are not read as more precise than they are.
        fig.add_vline(x=hz, line_dash="dash", line_color="#94a3b8", line_width=1)
        for val, lo, hi, col, nm in [
            (s0 * 100, arm["s0_lo"], arm["s0_hi"], "#ef4444", "No mitotane"),
            (s1 * 100, arm["s1_lo"], arm["s1_hi"], "#3b82f6", "Adj. mitotane"),
        ]:
            fig.add_trace(go.Scatter(
                x=[hz], y=[val],
                error_y=dict(type="data", symmetric=False,
                             array=[hi - val], arrayminus=[val - lo],
                             color=col, thickness=1.6, width=6),
                mode="markers", marker=dict(color=col, size=7),
                showlegend=False,
                hovertemplate=(f"{nm} at {hz} mo: {val:.1f}% "
                               f"(95% CI {lo:.1f}\u2013{hi:.1f}%)<extra></extra>")
            ))

        fig.update_layout(
            xaxis_title="Months",
            yaxis_title=f"{ep} probability (%)",
            yaxis=dict(range=[0, 100], gridcolor="#f1f5f9", zeroline=False),
            xaxis=dict(range=[0, hz * 1.04], gridcolor="#f1f5f9", zeroline=False),
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
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# Header above the S-GRAS Risk Card
st.markdown("<div style='font-size: 1.35rem; font-weight: 700; color: #0f172a; margin: 18px 0 10px 0;'>📊 S-GRAS Risk Stratification Assessment</div>", unsafe_allow_html=True)
st.markdown(render_sgras_badge(score, grp), unsafe_allow_html=True)

# 3 tabs for OS, 1-year PFS, 3-year PFS
tab_os, tab_pfs1, tab_pfs3 = st.tabs([
    "OS \u00b7 5 y",
    "PFS \u00b7 1 y",
    "PFS \u00b7 3 y"
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

# ---------- Model performance (internal validation) ----------
with st.expander("Model performance — internal validation (bootstrap)"):
    st.markdown(
        "Analytic cohort: **n = 852** (multicentre S-GRAS cohort, ENSAT I–III, R0/RX/R1; "
        "323 deaths, 498 progression events). The model was developed in the full cohort "
        "and validated internally by bootstrap (B = 1000). Every replicate "
        "re-estimates the propensity model and the stabilized weights before refitting the "
        "Cox models, so the optimism estimate covers the whole pipeline rather than the "
        "Cox step alone.")
    perf = pd.DataFrame({
        "Endpoint": ["OS", "PFS"],
        "Harrell C — apparent": ["0.724", "0.707"],
        "Optimism": ["0.006", "0.004"],
        "Harrell C — optimism-corrected": ["0.717", "0.703"],
        "Uniform shrinkage factor (95% CI)": ["0.953 (0.83–1.11)", "0.966 (0.84–1.11)"],
    })
    st.table(perf)
    cal = pd.DataFrame({
        "Endpoint / horizon": ["OS, 5-year (60 mo)", "PFS, 1-year (12 mo)",
                               "PFS, 3-year (36 mo)"],
        "Mean predicted risk": ["0.359", "0.298", "0.533"],
        "Observed risk (1 − KM)": ["0.351", "0.291", "0.520"],
        "Calibration-in-the-large": ["+0.008", "+0.007", "+0.013"],
    })
    st.table(cal)
    st.caption(
        "Optimism was ≤0.006 at both endpoints, so the model's apparent discrimination is "
        "close to what is expected in a new patient from the same population. The shrinkage "
        "factor was applied to the seven prognostic coefficients only — the treatment "
        "coefficient is left unshrunk — and the baseline hazard was re-estimated afterwards, "
        "which is why calibration-in-the-large stays near zero.")

with st.expander("How to read these numbers"):
    st.markdown(
        """
- **Survival with and without mitotane.** Two model-estimated probabilities for a
  patient with these five characteristics: one if adjuvant mitotane is started, one
  if it is not. Each carries a 95% interval.
- **Absolute benefit.** The difference between those two probabilities, in
  percentage points. The model estimates this difference more precisely than either
  probability on its own, because both curves share the same baseline hazard and the
  same prognostic coefficients.
- **Number needed to treat (NNT).** 1 divided by the absolute benefit: how many
  patients would have to start adjuvant mitotane for one additional patient to be
  alive (OS) or free of progression (PFS) at that time point. Its 95% interval comes
  from inverting the bounds of the benefit interval (Altman 1998).
- **Where the intervals come from.** A 1000-replicate bootstrap of the complete
  pipeline: every replicate re-estimates the propensity model, the stabilized
  weights, the Cox models and the baseline hazard. The uniform shrinkage factor is
  held fixed at its full-cohort value, so its own uncertainty is not included.
- **The interval carries the message, not the point estimate.** Across scenarios the
  NNT interval commonly spans a two- to six-fold range. A single number quoted
  without its interval is not an interpretable result.
- **Rounding.** The survival percentages and the absolute benefit are each rounded to
  one decimal independently, so subtracting the two displayed percentages can differ
  from the displayed benefit by 0.1 points. The displayed benefit is the more
  accurate of the two.
        """
    )

st.divider()
st.error(
    "**FOR RESEARCH AND EDUCATIONAL USE ONLY — NOT A MEDICAL DEVICE.** "
    "This is an investigational, research-grade calculator. It reports model-estimated "
    "quantities with their uncertainty and makes no treatment recommendation. "
    "It is intended for clinicians and researchers and is **not** intended for patients "
    "or members of the public. "
    "It has **not** been reviewed, cleared, or approved by the U.S. Food and Drug "
    "Administration (FDA), the European Medicines Agency (EMA), or any other regulatory "
    "authority, and it is **not** intended for the diagnosis, treatment, cure, mitigation, "
    "or prevention of disease in individual patients. "
    "It is provided \"as is\", without warranty of any kind, express or implied; the "
    "authors accept no liability for any decision, action, or outcome based on it.")
st.warning(
    """**Important limitations — please read before using these numbers.**

- **What these numbers are.** Model-based projections for a group of patients
  with this profile, calculated from a retrospective observational cohort. They
  are not a prediction of what will happen to one individual. The confidence
  intervals show how far each estimate could reasonably shift.
- **Not medical advice.** These estimates must **not** be the sole basis for any
  treatment decision. A qualified physician has to weigh the whole clinical
  picture — comorbidities, the patient's own priorities, and current guidelines.
- **Where the model applies.** Adrenocortical carcinoma of ENSAT stage I–III with
  R0, RX or R1 resection, in adults (≥18 years) **only**. It has not been validated
  for ENSAT IV or R2 disease and should not be used for those patients.
- **All five inputs must be known.** The model has no category for an unknown Ki-67,
  and Ki-67 moves the estimate more than any other variable (hazard ratio about 3.5
  for ≥20% versus 0–9%). If the proliferation index was not reported, the estimate is
  not valid for that patient.
- **How it was validated.** Internally, by bootstrap (B = 1000). No independent
  external cohort was available, so confirmation by other groups remains open.
- **The causal assumption.** Benefit estimates assume that no important
  prognostic factor outside the model influenced who received mitotane. This
  assumption is inherent to observational data and cannot be verified.
- **Treatment was recorded as yes/no.** Mitotane exposure is binary (started vs
  not started), with no dose and no plasma concentration. The estimates therefore
  describe the effect of *starting* adjuvant mitotane, and are probably smaller
  than the effect of adequate exposure.
- **One hazard ratio for every patient.** The model assumes mitotane lowers the
  hazard by the same proportion in everyone, so absolute benefit differs between
  patients only because their baseline risk differs. A pre-specified test
  indicated that benefit is in fact larger at higher baseline risk, so for
  low-risk patients these numbers may overstate the benefit."""
)
st.caption(
    "Model: doubly-robust IPTW Cox on the five S-GRAS components · analytic cohort n=852 · "
    "bootstrap internal validation (B=1000). "
    "Selections are processed in the current session only; they are not stored, logged, or "
    "shared, and no patient identifiers are requested. "
    "Responsible investigators: Saygili E (Çanakkale Onsekiz Mart University, Türkiye) and "
    "Ronchi CL (University of Birmingham, United Kingdom). "
    "© 2026 research use. "
    "Estimates are provided for research purposes only; their accuracy and generalisability "
    "beyond the validated population have not been established, and any interpretation "
    "remains the responsibility of a qualified investigator.")
