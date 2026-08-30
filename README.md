# Cardiovascular Risk Prediction

A Streamlit app that serves an **already-trained FT-Transformer** model
(via PyTorch Tabular) and an **already-fitted** preprocessing pipeline.
**Nothing is retrained.** This app only loads and runs what you provided.

⚠️ Research/educational tool only — not a medical diagnosis.

---

## 1. Project overview

- **Manual Prediction** — enter one patient's details, get a risk
  probability, a Low/Moderate/High category, and a live SHAP explanation.
- **Dataset Prediction** — upload a CSV of multiple patients, get
  predictions for all of them, download the results as CSV.
- **Explainability** — pick any prediction made in the current session
  (manual, or a specific row from a CSV batch) and see its SHAP-based
  feature contributions.

## 2. Folder structure

```
cardiovascular-risk-app/
├── app.py                      # Streamlit application
├── shap_explainer.py           # your existing SHAP module (used as-is)
├── preprocessing_pipeline.pkl  # your fitted preprocessing pipeline
├── ft_transformer_model/       # your saved PyTorch Tabular model bundle
│   ├── config.yml
│   ├── model.ckpt
│   ├── datamodule.sav
│   ├── callbacks.sav
│   └── custom_params.sav
├── requirements.txt
└── README.md
```

> **Why the SHAP file is `shap_explainer.py`, not `shap.py`:** you asked
> for `shap.py`, but Streamlit runs with the project folder on
> `sys.path`. A local file named `shap.py` would shadow the real `shap`
> PyPI package — and `shap_explainer.py` itself does `import shap` to
> get that package. Renaming it to `shap.py` would break SHAP with an
> import error/self-import. The filename was kept as-is on purpose.

> **Why the model ships as a folder, not `model.pkl`:** your uploaded
> `ft_transformer_model.zip` is a standard PyTorch Tabular
> `TabularModel.save_model()` **directory** (`config.yml` + `model.ckpt` +
> `datamodule.sav` + `callbacks.sav` + `custom_params.sav`), not a single
> pickle file. It's loaded with `TabularModel.load_model(dir)`. `app.py`
> will also auto-extract `ft_transformer_model.zip` at startup if you'd
> rather ship the zip — either is fine.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Important:** `requirements.txt` pins `scikit-learn==1.6.1` exactly.
This was **confirmed by actually running** `preprocessing_pipeline.pkl`
in the build environment: with scikit-learn 1.8.0 installed, the fitted
`IterativeImputer` inside it raised
`AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`
during `.transform()`. Pinning to 1.6.1 (the version it was fit with)
avoids this. Do not upgrade scikit-learn in this project without
re-testing.

`torch` / `pytorch_tabular` versions in `requirements.txt` are
reasonable defaults — they could **not** be verified in the build
sandbox (no network access there to install or run them). If
`TabularModel.load_model(...)` fails for you locally, match these two
packages to whatever versions were actually used to train the model.

## 4. Run locally

```bash
streamlit run app.py
```

## 5. Deploy on Streamlit Cloud

1. Push this whole folder to a GitHub repo (including
   `ft_transformer_model/` and `preprocessing_pipeline.pkl` — Streamlit
   Cloud needs the actual model files, it can't train or fetch them).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at the repo, branch, and `app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically.
4. No secrets or API keys are needed — everything runs locally to the
   app.

All paths in `app.py` are relative to the app file (`Path(__file__).parent`),
so this works the same locally and on Streamlit Cloud.

## 6. Required files

| File | Purpose |
|---|---|
| `ft_transformer_model/` (or `.zip`) | Trained FT-Transformer, PyTorch Tabular format |
| `preprocessing_pipeline.pkl` | Fitted imputer, IQR outlier bounds, power transformer, scaler, feature order |
| `shap_explainer.py` | SHAP explanation logic (kernel explainer wrapper) |

## 7. SHAP explainability

`shap_explainer.py` is used as provided — `create_kernel_explainer`,
`calculate_shap_values`, and `explain_patient` are called directly by
`app.py`, on live model predictions. No SHAP values are hard-coded or
randomly generated.

**Background data caveat:** SHAP's `KernelExplainer` needs a reference
("background") dataset. None of the uploaded files included the
training dataset, so:
- If you've uploaded a CSV in the **Dataset Prediction** tab this
  session, that data (summarized with `shap.kmeans`) is used as the
  background — a real, representative reference.
- Otherwise, a single synthetic "average patient" row (the processed
  feature mean, ≈0 since the fitted scaler's mean is ~0) is used
  instead. The app tells you when this fallback is in effect. For more
  representative explanations, upload a reference CSV first.

## 8. CSV batch prediction

Upload a CSV with these required columns (exact names):

```
male, age, education, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes, BMI
```

The app validates missing/extra columns, coerces non-numeric values to
missing (imputed rather than crashing), applies the saved preprocessing
pipeline (`.transform()` only — never refit), and runs the FT-Transformer
model. Results (`Prediction`, `Risk_Probability`, `Risk_Category`) are
appended to your original data and downloadable as
`prediction_results.csv`. To avoid running SHAP on every row (which
would be very slow with a kernel explainer), per-row explanations are
computed on demand for a row you select, in the Explainability tab.

---

## Known limitations / assumptions — please read

These were the judgment calls made while inspecting your files, flagged
explicitly rather than silently baked in:

1. **`diabetes` is collected but not used by the model directly.** Your
   preprocessing pipeline was fit on 9 raw columns including `diabetes`,
   but the FT-Transformer's `config.yml` only lists 8 `continuous_cols`
   (no `diabetes`). The app collects/requires `diabetes` purely so the
   fitted imputer/power-transformer/scaler receive the correct number of
   columns, then drops it right before scoring.

2. **Preprocessing step order (clip → impute → power transform → scale)
   is inferred, not recorded in the pickle.** The pickle stores four
   independent fitted objects plus IQR bounds, with no explicit pipeline
   object dictating order. The order above was chosen because the IQR
   bounds are on the *raw* scale (e.g. age 21–77, BMI 15.6–35.5), which
   only makes sense if clipping happens before any transform. If this
   doesn't match your original training notebook, this is the first
   thing to check.

3. **IQR clipping is applied exactly as saved, including a quirk on rare
   binary flags.** `BPMeds`, `prevalentStroke`, and `diabetes` all have
   IQR bounds of `[0.0, 0.0]` (because these flags are heavily
   imbalanced, so both quartiles are 0). Applied literally, any `1`
   ("Yes") entered for these three fields gets clipped to `0` before
   scoring — i.e. these three flags have very little effect on
   predictions as deployed here. This was reproduced exactly rather than
   "fixed", since the brief was to replicate the fitted training
   pipeline. To exclude specific columns from clipping, add them to
   `SKIP_CLIP_COLUMNS` near the top of `app.py`.

4. **Risk category thresholds (Low < 10%, Moderate 10–20%, High ≥ 20%)
   are a general convention, not a value found in your files.** No
   calibrated threshold/cutoff artifact was included with the uploaded
   files. The binary High/Low **label** uses the model's own
   `target_prediction` output; only the 3-tier **category** uses these
   thresholds. Adjust `RISK_LOW_MAX` / `RISK_MODERATE_MAX` in `app.py` if
   you have a project-specific cutoff.

5. **The FT-Transformer model itself could not be run in the build
   sandbox** (no network access to install `torch` / `pytorch_tabular`).
   Model loading and `model.predict()`'s output column names
   (`target_prediction`, `target_1_probability`) are taken from
   `config.yml` (target name) and from your `shap_explainer.py`'s own
   documented expectations, with a fallback search for the probability
   column if the exact name doesn't match. **Please run the app locally
   first** and check the Manual Prediction tab works before depl
