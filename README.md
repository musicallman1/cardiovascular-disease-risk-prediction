# Cardiovascular Disease Prediction — 10-Year CHD Risk

A logistic regression model predicting a patient's 10-year risk of coronary
heart disease (CHD), built on the Framingham Heart Study dataset.

## Problem

Roughly 15% of patients in this dataset go on to develop CHD within 10 years.
The goal is to build a model that flags at-risk patients accurately enough to
be clinically useful — while being explicit about the fact that, in this
setting, a **missed CHD case carries far more cost than a false alarm**. That
asymmetry shapes every modeling decision below, not just the final threshold.

## Data

- **3,390 patients**, 17 raw columns (demographic, behavioral, and clinical
  measurements), sourced from the Framingham Heart Study.
- **Missingness:** 0.88% overall, concentrated in `glucose` (9.0%),
  `education` (2.6%), `BPMeds` (1.3%), `totChol` (1.1%), `cigsPerDay` (0.65%),
  `BMI` (0.41%), `heartRate` (0.03%). Mostly appears to be missing completely
  at random (MCAR), with one weak MAR signal between `totChol` and `glucose`
  missingness (likely shared lab-visit patterns).
- **Class balance:** 2,879 negative / 511 positive (85% / 15%) — meaningfully
  imbalanced, which rules out accuracy as a usable evaluation metric on its
  own (see Methodology).

## Methodology

**Preprocessing is fit on training data only, and applied unchanged to
validation and test data.** Medians used for imputation and bounds used for
outlier winsorizing are computed exclusively from the training split, then
applied identically to validation and test — no statistic used to transform
the data is ever computed on data the model will later be evaluated against.

- **Split:** 60% train / 20% validation / 20% test, stratified on the target.
  The test set is touched exactly once, at the very end, for final reporting.
- **Feature engineering:** sex encoded as binary; `education` (an ordinal,
  not linearly-related-to-outcome variable, with undocumented level meanings)
  recoded to a binary `less_education` flag based on EDA showing a
  non-linear relationship between education level and CHD rate.
- **Outlier handling:** IQR-based winsorizing for `totChol`/`glucose`,
  z-score winsorizing for `sysBP` — bounds fit on training data only.
- **Model:** logistic regression with `class_weight='balanced'` to
  compensate for the 85/15 imbalance.

### Metric and threshold selection

Accuracy is misleading here — a naive classifier that always predicts "no
CHD" scores **84.9%** accuracy by doing nothing useful. **ROC-AUC** is used
as the headline discrimination metric (threshold-independent), and
**precision/recall at a chosen operating threshold** are used for the
practical, decision-facing metrics.

Youden's J statistic (the standard sensitivity/specificity-balanced
criterion) was tested first, but it selects a threshold that *reduces*
recall relative to the default 0.5 cutoff — the wrong direction given the
stated cost asymmetry. Instead, the final threshold is chosen by **searching
for the threshold that maximizes precision subject to a minimum recall of
75%** on the validation set, explicitly encoding the clinical priority of
catching true CHD cases over avoiding false positives.

## Results

| Metric | Default (0.50) | Youden's J (0.581) | **Selected (0.401)** |
|--------|-----------------|---------------------|------------------------|
| Accuracy | 72% | 78% | 58% |
| Recall — CHD | 65% | 47% | **77%** |
| Precision — CHD | 30% | 33% | 23% |
| F1 — CHD | 0.41 | 0.39 | 0.36 |

- **Final test-set ROC-AUC: 0.713** — the model ranks a true CHD patient
  above a non-CHD patient 71.3% of the time, well above chance.
- **Selected threshold: 0.401**, chosen to guarantee ≥75% recall on
  validation data. On the held-out test set this achieves **77% recall**
  at **23% precision** — meaning roughly half of screened patients are
  flagged, of whom fewer than one in four actually develop CHD. This is a
  deliberate tradeoff, not an incidental one: it should be paired with
  appropriately scaled follow-up screening capacity in any real deployment,
  and is not intended as a general-purpose operating point.

### Strongest predictors (by odds ratio)

| Feature | Odds Ratio | Interpretation |
|---|---|---|
| prevalentStroke | 1.96 | Prior stroke nearly doubles CHD odds — strongest predictor |
| diabetes | 1.75 | Second strongest predictor |
| sex_male | 1.65 | Males have 65% higher CHD odds than females |
| prevalentHyp | 1.32 | Hypertension increases CHD odds by 32% |
| age | 1.06/yr | A 20-year age gap corresponds to ~3.0x higher odds |

Continuous lab measurements (BMI, glucose, totChol) contribute little once
the categorical clinical flags and age are accounted for. Full coefficient
table and discussion in the notebook, Section 3.5.

## How to run

1. Clone the repo and open `Cardiovascular_Prediction_1.ipynb` in Jupyter or
   Google Colab.
2. Place `train.csv` in the same directory as the notebook (or update the
   `pd.read_csv(...)` path in Section 1.1 to point at your copy).
3. Run all cells top to bottom. No cell depends on manual state from a
   previous session — a fresh runtime and a single top-to-bottom run will
   reproduce every number in this README.

**Requirements:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,
`scipy`.

## Notes on methodology choices

- EDA (Section 2) runs on a full-dataset copy (`cardio_cleaned`), separate
  from the modeling pipeline (Section 3), which is built from scratch on raw
  data and preprocessed using train-only statistics. This is a deliberate,
  documented tradeoff: descriptive visualization benefits from seeing the
  full dataset, but no number computed in Section 2 is reused as a modeling
  input — every statistic that actually touches the model is refit in
  Section 3.
