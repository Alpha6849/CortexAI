
# Phase 1 — AutoML + EDA Engine

This phase builds the core engine of Cortex AI.  
The goal is to make a clean, simple, usable AutoML + EDA tool.

Upload CSV → it automatically:
- Loads the data  
- Detects schema  
- Cleans missing values + outliers  
- Generates EDA report  
- Detects task type  
- Trains ML models  
- Shows metrics + charts  

This becomes the **foundation** for Phase 2 when we add LLM reasoning.

---

## ⭐ Goals for Phase 1

### ✔ Functional Requirements
- CSV data ingestion  
- Schema detection (numeric / categorical / datetime)  
- Missing value imputation  
- Outlier handling  
- Basic type conversion logic  
- Automated EDA summary  
- Profiling report  
- Automatic ML model training  
- Metric comparison (Accuracy, R², F1-score, etc.)  
- End-to-end notebook workflow  

### ✔ Technical Deliverables
- Clean Jupyter notebook pipeline  
- Modular notebooks for each step  
- Results folder for reports/models  
- Working MVP that runs locally  

---

## 📁 Folder Structure (Phase 1)

```text
Phase1_AutoML_EDA/
│
├── data/
│   ├── sample.csv
│   └── cleaned/
│
├── results/
│   ├── eda_reports/
│   ├── models/
│   └── logs/
│
└── notebooks/
    ├── 00_main_pipeline.ipynb
    ├── 01_data_loader.ipynb
    ├── 02_schema_detector.ipynb
    ├── 03_data_cleaner.ipynb
    ├── 04_eda.ipynb
    └── 05_model_training.ipynb
```

## 🔧 Phase 1 Notebook Modules

### **01_data_loader.ipynb**
- Load CSV  
- Validate file type  
- Encoding detection  

### **02_schema_detector.ipynb**
- Detect data types  
- Separate numeric / categorical / datetime  
- Suggest target column  

### **03_data_cleaner.ipynb**
- Missing values (median/mode)  
- Outlier removal (IQR)  
- Datatype casting  

### **04_eda.ipynb**
- Summary statistics  
- Correlation heatmap  
- Profiling report (HTML)  

### **05_model_training.ipynb**
- Task detection  
- Train ML models  
- Compare basic metrics  

### **00_main_pipeline.ipynb**
- Full linear AutoML + EDA pipeline  
