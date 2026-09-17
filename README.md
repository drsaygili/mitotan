# Adjuvant Mitotane Benefit Calculator — Adrenocortical Carcinoma

Research-grade online calculator that reports individualized, model-estimated
quantities for adjuvant mitotane in adrenocortical carcinoma: survival with and
without treatment, absolute treatment benefit, and number needed to treat — each
with a 95% interval. **It issues no treatment recommendation.**

Live app: https://mitotane.streamlit.app

## Scope

Valid for adrenocortical carcinoma of **ENSAT stage I–III** with **R0, RX or R1**
resection, in **adults (≥18 years)** only. Not validated for ENSAT IV or R2 disease.
All five S-GRAS components must be known; the model has no category for an unknown
Ki-67.

## Model

Doubly-robust IPTW Cox model on the five S-GRAS components (age, symptoms at
diagnosis, ENSAT stage, resection status, Ki-67 index).

| | |
|---|---|
| Analytic cohort | n = 852 (323 deaths, 498 progression events) |
| Adjusted hazard ratio, OS | 0.717 (95% CI 0.570–0.903) |
| Adjusted hazard ratio, PFS | 0.609 (95% CI 0.506–0.733) |
| Validation | Internal, bootstrap B = 1000, full pipeline |
| Harrell C, optimism-corrected | 0.717 (OS), 0.703 (PFS) |
| Random seed | 20260717 |

No independent external cohort was available; external validation by independent
groups remains open.

The app performs no model fitting. All parameters are read from
`cox_calculator_params.json` (coefficients and baseline cumulative hazard),
`cox_arm_ci.json` (per-arm survival intervals) and `cox_cate_ci.json` (benefit and
NNT intervals), all produced by the analysis pipeline reported in the manuscript.

## Running locally

```bash
pip install -r requirements.txt
streamlit run acc_mitotane_calculator.py
```

Dependencies are pinned on purpose: the app is reached by a QR code on a conference
poster and must not change behaviour when a new upstream release appears.

## Responsible investigators

- Saygili E — Division of Endocrinology and Metabolism, Department of Internal
  Medicine, Faculty of Medicine, Çanakkale Onsekiz Mart University, Çanakkale, Türkiye
- Ronchi CL — Department of Metabolism and Systems Science, College of Medicine and
  Health, University of Birmingham, Birmingham, United Kingdom (corresponding)

## Citation

Please cite the source publication (manuscript under review, 2026) when using these
estimates.

## Disclaimer

For research and educational use only — **not a medical device**. Intended for
clinicians and researchers; not intended for patients or members of the public. Not
reviewed, cleared, or approved by the FDA, the EMA, or any other regulatory
authority, and not intended for the diagnosis, treatment, cure, mitigation, or
prevention of disease in individual patients. Provided "as is", without warranty of
any kind, express or implied; the authors accept no liability for any decision,
action, or outcome based on it. Selections entered in the app are processed in the
current session only; they are not stored, logged, or shared, and no patient
identifiers are requested.
