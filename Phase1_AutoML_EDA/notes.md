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

Next:
- Begin Notebook 03 — Data Cleaner.
- Implement missing value imputation, outlier handling, datatype casting.
- Save cleaned datasets into data/cleaned/.



