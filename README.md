# CortexAI — Where Data Learns to Think

> Upload any CSV. CortexAI automatically cleans, analyzes, models, and explains your dataset.

---

## ⚙️ Overview

**CortexAI** is an **autonomous AI data scientist** that combines:
- Automated **data cleaning & preprocessing**
- Smart **EDA (Exploratory Data Analysis)**
- AutoML model selection & training
- Interactive **visual insights**

Built with:
> 🐍 Python · ⚡ Streamlit · 🤖 Scikit-learn · 📊 Plotly ·  XGBoost

---

##  Features
 CSV Upload & Schema Detection  
 Missing Values & Outlier Handling  
 Automated EDA Summary  
 Model Training (Classification / Regression)  
 Metrics + Confusion Matrix + Visual Plots  
 Streamlit UI for seamless use  

---

##  How It Works
1. Upload any `.csv` file  
2. CortexAI automatically:
   - Detects column types (numeric, categorical, target)
   - Cleans data (missing values, outliers)
   - Performs EDA (distributions, correlations)
   - Trains the best-fit ML model
   - Shows results, metrics, and plots  

---

##  Run Locally

### 1️ Clone this repo
```bash
git clone https://github.com/Alpha6849/CortexAI.git
cd CortexAI/Phase1_AutoML_EDA
 Install dependencies
bash
Copy code
pip install -r requirements.txt
 Run the Streamlit app
bash
Copy code
streamlit run main.py
 Upload your CSV and enjoy 🚀
📈 Future Roadmap
🔜 Phase 2: LLM Reasoning (natural language pipeline planning)

🔜 Phase 3: Deep Learning Imputation + SHAP explainability

🔜 Phase 4: Streamlit Cloud + Public Demo

🧾 License
MIT License © 2025 Prathamesh

yaml
Copy code

---

# 📦 requirements.txt (copy this)

```txt
streamlit
pandas
numpy
scikit-learn
xgboost
ydata-profiling
plotly
matplotlib
