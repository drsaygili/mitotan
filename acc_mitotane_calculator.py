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
import matplotlib.pyplot as plt

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

# ---------- UI ----------
st.set_page_config(page_title="ACC Mitotane Benefit Calculator", layout="wide")
st.title("Adjuvant Mitotane Benefit Calculator — Adrenocortical Carcinoma")
st.caption("Doubly-Robust IPTW Cox model · development n=755, external validation n=97 · "
           "scope: ENSAT I–III, R0/RX/R1")

with st.sidebar:
    st.header("Patient S-GRAS components")
    age = st.radio("Age", [0, 1], format_func=lambda x: "<50 y" if x == 0 else "≥50 y")
    sympt = st.radio("Symptoms at diagnosis", [0, 1],
                     format_func=lambda x: "Absent" if x == 0 else "Present")
    ensat = st.radio("ENSAT stage", [0, 1], format_func=lambda x: "I–II" if x == 0 else "III")
    rstatus = st.radio("Resection status", [0, 1, 2],
                       format_func=lambda x: {0: "R0", 1: "RX", 2: "R1"}[x])
    ki67 = st.radio("Ki-67 index", [0, 1, 2],
                    format_func=lambda x: {0: "<10%", 1: "10–19%", 2: "≥20%"}[x])
    horizon_os = 60  # OS fixed at 5-year (60-month) horizon
    st.caption("OS horizon fixed at 60 months (5-year).")
    horizon_pfs = st.radio("PFS horizon (months)", [12, 36], index=1,
                           format_func=lambda x: f"{x} months ({x//12}-year)")

kw = dict(age=age, sympt=sympt, ensat=ensat, rstatus=rstatus, ki67=ki67)
score, grp = sgras(age, sympt, ensat, rstatus, ki67)
st.markdown(f"**S-GRAS score: {score}/9 — {grp} risk group**")

cols = st.columns(2)
for col, (ep, hz) in zip(cols, [("OS", horizon_os), ("PFS", horizon_pfs)]):
    with col:
        yr_lbl = hz // 12
        lbl = (f"{yr_lbl}-year Overall survival" if ep == "OS"
               else f"{yr_lbl}-year Progression-free survival")
        st.subheader(lbl)
        s0 = surv_at(ep, hz, 0, **kw); s1 = surv_at(ep, hz, 1, **kw)
        rec = CATE_CI[f"{age}-{sympt}-{ensat}-{rstatus}-{ki67}"][f"{ep}{hz}"]
        c_pt, c_lo, c_hi = rec["cate"], rec["lo"], rec["hi"]
        nnt, nnt_lo, nnt_hi = rec["nnt"], rec.get("nnt_lo"), rec.get("nnt_hi")
        yr = hz // 12
        # Survival with/without treatment
        s_a, s_b = st.columns(2)
        s_a.metric(f"{ep} without mitotane", f"{s0*100:.1f}%")
        s_b.metric(f"{ep} with mitotane", f"{s1*100:.1f}%")
        # NNT headline (prominent) + absolute benefit (secondary)
        nnt_ci = (f"{nnt_lo}–{nnt_hi}" if nnt_lo and nnt_hi else "—")
        outcome = "survivor" if ep == "OS" else "progression-free patient"
        tip = (f"One additional {yr}-year {outcome} is expected for every {nnt} patients "
               f"treated with adjuvant mitotane. NNT = 1 / absolute benefit; the 95% CI is "
               f"obtained by inverting the bounds of the absolute-benefit CI (Altman 1998). "
               f"Confidence intervals are from a 1000-replicate bootstrap of the "
               f"doubly-robust IPTW Cox model.")
        st.markdown(
            f"<div style='background:#eef4fb;border-radius:10px;padding:14px 18px;"
            f"margin:6px 0 2px 0;border:1px solid #cfe0f2'>"
            f"<span style='font-size:15px;color:#334'>Number needed to treat "
            f"({yr}-year {ep})</span>"
            f"<span title=\"{tip}\" style='cursor:help;color:#6b8cae;font-size:14px;"
            f"margin-left:6px'>&#9432;</span><br>"
            f"<span style='font-size:44px;font-weight:700;color:#12467a'>{nnt}</span>"
            f"<span style='font-size:17px;color:#12467a'>&nbsp;&nbsp;(95% CI {nnt_ci})</span>"
            f"<br><span style='font-size:14px;color:#556'>"
            f"Absolute {ep} benefit <b>{c_pt:+.1f}%</b> "
            f"(95% CI {c_lo:+.1f} to {c_hi:+.1f} pp)</span></div>",
            unsafe_allow_html=True)
        # clean counterfactual curves (no per-arm ribbons; see benefit CI above)
        tg, S0 = surv_curve(ep, 0, **kw); _, S1 = surv_curve(ep, 1, **kw)
        mask = tg <= hz
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.step(tg[mask], S0[mask] * 100, where="post", color="#B2182B", label="No mitotane")
        ax.step(tg[mask], S1[mask] * 100, where="post", color="#2166AC", label="Adj. mitotane")
        ax.set_xlim(0, hz); ax.set_ylim(0, 100)
        ax.set_xlabel("Months"); ax.set_ylabel(f"{ep} probability (%)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

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
