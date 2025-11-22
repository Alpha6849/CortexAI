
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

## 🎯 Goals for Phase 1

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
- Streamlit interface end-to-end  

### ✔ Technical Deliverables
- Modular Python code in `/src`  
- Streamlit app in `/app`  
- Results folder for reports  
- Working MVP that anyone can use locally  

---

## 📁 Folder Structure (Phase 1)
```text

Phase1_AutoML_EDA/
│
├── data/
│ ├── sample.csv
│ └── cleaned/
│
├── results/
│ ├── eda_reports/
│ ├── models/
│ └── logs/
│
├── src/
│ ├── data_loader.py
│ ├── schema_detector.py
│ ├── data_cleaner.py
│ ├── eda_engine.py
│ ├── model_trainer.py
│ ├── utils.py
│ └── init.py
│
├── app/
│ └── app.py
│
├── main.py
├── README.md
└── requirements.txt

```

---

## 🔧 Phase 1 Modules

### **data_loader.py**
- Load CSV  
- Validate file type  
- Encoding detection  

### **schema_detector.py**
- Detect data types  
- Suggest target column  

### **data_cleaner.py**
- Missing values (median/mode)  
- Outlier removal (IQR)  
- Datatype casting  

### **eda_engine.py**
- Summary stats  
- Correlation heatmap  
- Profiling report (HTML)  

### **model_trainer.py**
- Task detection  
- Train ML models  
- Return best model  

### **app/app.py**
- Full Streamlit UI flow  

---
  
