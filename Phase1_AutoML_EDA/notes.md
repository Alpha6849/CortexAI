# Phase 1 — Notes & Daily Log

## 📅 Day 1 (22/11/25)
- Set up the full project folder structure.
- Created notebooks folder and all phase 1 notebook files.
- Added the Iris dataset (sample.csv).
- Created data_loader.ipynb notebook.
- Completed 4 cells today:
  - Imports
  - Setting data path
  - CSV loading function
  - Basic dataset inspection
- Also confirmed dataset has no missing values.

## 📅 Day 2 (23/11/25) — Schema Detector Notebook Complete
- Started and completed Notebook 02 (schema_detector.ipynb).
- Added logic to detect:
  - Numeric columns
  - Categorical columns
  - Boolean columns
  - Datetime columns (with corrected safe detection)
  - ID columns
  - Low-variance columns
  - Target column candidates
- Fixed datetime misdetection issue where numeric columns were treated as timestamps.
- Built the final unified schema dictionary.
- Created master function `generate_schema(df)` that runs the entire pipeline.
- Validated everything works correctly on the Iris dataset.

## 🗓️ Day 3 (24/11/25) – Data Cleaning Notebook Completed

- Continued working on Notebook 03 (Data Cleaning).
- Loaded the cleaned dataset from previous notebook.
- Debugged schema regeneration issues.
- Added missing value imputation (median + mode).
- Implemented outlier handling using the IQR method.
- Added datatype fixing (numeric conversion, categorical cleaning).
- Built logic for dropping ID + low-variance columns.
- Discovered a major bug in ID detection (Width → "id"), fixed using regex.
- Rebuilt schema cleanly after fixing the function.
- Successfully ensured only the correct ID column ("Id") is dropped.
- Saved final cleaned dataset as:
  - CSV
  - Parquet
- Generated a professional cleaning summary report with details.

###  What I learned
- ID detection can fail if substring "id" appears inside words like "Width".
- Good debugging needs step-by-step checking of uniqueness + schema.
- Parquet format is faster and better for ML pipelines.
- Cleaning pipelines need careful schema rebuild after function changes.

### 📁 Output Generated
- `cleaned_dataset_<timestamp>.csv`
- `cleaned_dataset_<timestamp>.parquet`
- `cleaning_report_<timestamp>.txt`

###  Status :-
Notebook 03 is **fully completed** and working correctly.
Next step = Notebook 04 (EDA Engine).

## 📅 [Date: 27/11/25] — Completed Notebook 04 (EDA Engine)

###  What was done:
- Loaded the cleaned dataset from Notebook 03.
- Generated missing value summary, numeric stats, and categorical stats.
- Visualized:
  - Correlation heatmap (numeric)
  - Histograms (numeric)
  - Countplots (categorical)
  - Boxplots vs target
  - Pairplot (numeric relationships)
- Created a full automated EDA profiling report using ydata-profiling.
- Saved HTML report into `results/eda_reports/eda_report_<timestamp>.html`.
- Created a lightweight EDA Summary JSON for use in the Streamlit app.
- Verified everything runs smoothly on Python 3.10.

###  Notes:
- Kernel was switched to Python 3.10 for profiling compatibility.
- All outputs saved in `results/eda_reports/`.
- Dataset used: latest cleaned CSV from Notebook 03.
- No cleaning or mutations in Notebook 04 — only analysis + visualization.

###  Next Steps:
- Begin Notebook 05 (Model Training).
- Detect task type (classification vs regression).
- Train baseline models (Logistic Regression, Random Forest, XGBoost, Linear Regression, etc.).
- Auto-select best model.
- Save model + metrics.

