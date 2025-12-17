"""
6_AI_Analyst.py

Groq-powered AI Analyst for CortexAI.
Produces structured JSON + human-readable insights.
STRICTLY grounded using real pipeline outputs (LLM-safe).
"""

import os
import json
import streamlit as st
import requests
from pathlib import Path
from dotenv import load_dotenv

# --------------------------------------------------
# PATH + ENV
# --------------------------------------------------
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing.")
    st.stop()

# --------------------------------------------------
# USAGE / PLAN MANAGEMENT
# --------------------------------------------------
from pipeline.usage_manager import (
    init_plan_and_usage,
    enforce_limit,
    increment_usage,
    get_current_plan
)

init_plan_and_usage()

# --------------------------------------------------
# PROMPTS (STRICT + GROUNDED)
# --------------------------------------------------

BASE_CONSTRAINTS = """
CRITICAL CONSTRAINTS:
- Use ONLY the information explicitly present in the provided JSON.
- Do NOT infer, guess, or use prior knowledge.
- If a question cannot be answered, say:
  "This information is not available in the pipeline outputs."

JSON RULES:
- Output MUST be valid JSON.
- Arrays MUST be real JSON arrays (no numeric keys).
- No text outside the JSON object.
"""

SIMPLE_PROMPT = f"""
You are a data analyst.

Return EXACTLY this JSON object:
{{
  "summary": string,
  "technical_summary": string,
  "insights": [string],
  "recommendations": [string],
  "warnings": [string]
}}

{BASE_CONSTRAINTS}
"""

PRO_PROMPT = f"""
You are a senior data scientist.

Return EXACTLY this JSON object:
{{
  "summary": string,
  "technical_summary": string,
  "detailed_insights": [
    {{
      "insight": string,
      "evidence": string
    }}
  ],
  "recommendations": [string],
  "warnings": [string],
  "model_explanation": string
}}

{BASE_CONSTRAINTS}
"""

# --------------------------------------------------
# GROQ API
# --------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

FREE_MODELS = ["llama-3.1-8b-instant"]
PRO_MODELS = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

def call_groq(messages, models, temperature, max_tokens):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    last_error = None

    for model in models:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json(), model
            last_error = resp.text
        except Exception as e:
            last_error = str(e)

    raise RuntimeError(last_error)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def normalize_arrays(obj):
    """Fix numeric-key arrays returned by LLM"""
    if isinstance(obj, dict):
        # numeric keys → list
        if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
            return [normalize_arrays(v) for _, v in sorted(obj.items(), key=lambda x: int(x[0]))]
        return {k: normalize_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_arrays(v) for v in obj]
    return obj


def extract_json(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}")
        return normalize_arrays(json.loads(text[start:end + 1]))
    except Exception:
        return None


def collect_session_reports():
    keys = ["schema", "eda_report", "dataset_quality"]
    return {k: st.session_state[k] for k in keys if k in st.session_state}


def build_llm_safe_reports(reports: dict) -> dict:
    safe = {}

    if "schema" in reports:
        safe["schema"] = {
            "target": reports["schema"].get("target"),
            "num_numeric": len(reports["schema"].get("numeric", [])),
            "num_categorical": len(reports["schema"].get("categorical", []))
        }

    if "dataset_quality" in reports:
        safe["dataset_quality"] = reports["dataset_quality"]

    if "eda_report" in reports:
        eda = reports["eda_report"]
        safe["eda_summary"] = {
            "binary_outcomes": eda.get("binary_outcomes", []),
            "outcome_analysis": eda.get("outcome_analysis", {})
        }

    return safe

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(page_title="AI Analyst — CortexAI", layout="wide")
st.title("🧠 AI Analyst (Groq-powered)")
st.caption("Strictly grounded AI insights based on pipeline outputs.")

mode = st.radio("Choose Analysis Mode:", ["Simple", "Professional", "Both"])

reports = collect_session_reports()
if not reports:
    st.error("Run Load → Schema → Cleaning → EDA → Training first.")
    st.stop()

# --------------------------------------------------
# DATASET QUALITY
# --------------------------------------------------

if "dataset_quality" in reports:
    dq = reports["dataset_quality"]
    st.metric("Learnability Score", f"{dq['learnability_score']} / 100")
    st.info(f"Verdict: {dq['verdict']}")

# --------------------------------------------------
# LLM CONFIG
# --------------------------------------------------

temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
max_tokens = st.number_input("Max Output Tokens", 256, 2048, 800)
custom_question = st.text_area("Optional: Ask a custom question")

safe_reports = build_llm_safe_reports(reports)
user_json = json.dumps(safe_reports, separators=(",", ":"))

# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------

if st.button("Generate Analysis"):
    enforce_limit("llm_calls", "🚫 AI Analyst limit reached.")
    increment_usage("llm_calls")

    plan = get_current_plan()
    models = FREE_MODELS if plan == "free" else PRO_MODELS

    def run_llm(prompt):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Pipeline JSON:\n{user_json}\n\nQuestion:\n{custom_question or 'None'}"}
        ]
        raw, model_used = call_groq(messages, models, temperature, max_tokens)
        return extract_json(raw["choices"][0]["message"]["content"]), model_used

    with st.spinner("Analyzing…"):
        if mode in ["Simple", "Both"]:
            simple_out, model_used = run_llm(SIMPLE_PROMPT)
            st.subheader("📝 Simple Analysis")
            st.json(simple_out)

        if mode in ["Professional", "Both"]:
            pro_out, model_used = run_llm(PRO_PROMPT)
            st.subheader("🧪 Professional Analysis")
            st.json(pro_out)

    st.success(f"Model used: `{model_used}`")
