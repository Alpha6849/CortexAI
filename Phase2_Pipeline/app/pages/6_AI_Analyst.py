"""
6_AI_Analyst.py

CortexAI — AI Analyst (Production Final)
Balanced constraints: grounded, but useful.
"""

import os
import json
import streamlit as st
import requests
from pathlib import Path
from dotenv import load_dotenv

# --------------------------------------------------
# ENV
# --------------------------------------------------
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing.")
    st.stop()

# --------------------------------------------------
# USAGE
# --------------------------------------------------
from pipeline.usage_manager import (
    init_plan_and_usage,
    enforce_limit,
    increment_usage,
    get_current_plan
)

init_plan_and_usage()

# --------------------------------------------------
# PROMPTS 
# --------------------------------------------------

BASE_CONSTRAINTS = """
RULES:
- Use ONLY the provided JSON data.
- You MAY summarize, compare, and explain values that already exist.
- DO NOT invent new statistics or facts.
- If something truly does not exist, say:
  "This information is not available in the pipeline outputs."

OUTPUT:
- Valid JSON only
- No text outside JSON
"""

SIMPLE_PROMPT = f"""
You are a helpful data analyst.

Return JSON:
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

Return JSON:
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
# GROQ
# --------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
FREE_MODELS = ["llama-3.1-8b-instant"]
PRO_MODELS = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

def call_groq(messages, models, temperature, max_tokens):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    for model in models:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json(), model

    raise RuntimeError("Groq API failed")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def normalize(obj):
    if isinstance(obj, dict):
        if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
            return [normalize(v) for _, v in sorted(obj.items(), key=lambda x: int(x[0]))]
        return {k: normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj

def extract_json(text):
    try:
        start, end = text.find("{"), text.rfind("}")
        return normalize(json.loads(text[start:end + 1]))
    except Exception:
        return None

# --------------------------------------------------
# BUILD LLM INPUT (IMPORTANT FIX)
# --------------------------------------------------

def build_llm_safe_reports():
    eda = st.session_state.get("eda_report", {})
    schema = st.session_state.get("schema", {})
    quality = st.session_state.get("dataset_quality", {})

    return {
        "schema": {
            "target": schema.get("target"),
            "task_type": schema.get("task_type"),
            "numeric": schema.get("numeric", []),
            "categorical": schema.get("categorical", []),
            "ordinal": schema.get("ordinal", [])
        },
        "eda": {
            "basic_statistics": eda.get("basic_statistics", {}),
            "target_analysis": eda.get("target_analysis", {}),
            "numeric_analysis": eda.get("numeric_analysis", {}),
            "ordinal_analysis": eda.get("ordinal_analysis", {}),
            "binary_outcome_analysis": eda.get("binary_outcome_analysis", {}),
            "key_insights": eda.get("key_insights", [])
        },
        "dataset_quality": quality
    }

# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(page_title="AI Analyst — CortexAI", layout="wide")
st.title("🧠 AI Analyst (Groq-powered)")
st.caption("Grounded, explainable AI insights from your pipeline.")

mode = st.radio("Choose Analysis Mode:", ["Simple", "Professional", "Both"])

if "eda_report" not in st.session_state:
    st.error("Run the full pipeline first.")
    st.stop()

dq = st.session_state.get("dataset_quality", {})
if dq:
    st.metric("Learnability Score", f"{dq['learnability_score']} / 100")
    st.info(f"Verdict: {dq['verdict']}")

temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
max_tokens = st.number_input("Max Output Tokens", 256, 2048, 800)
question = st.text_area("Optional: Ask a custom question")

safe_reports = build_llm_safe_reports()
payload_json = json.dumps(safe_reports, indent=2)

# --------------------------------------------------
# RUN
# --------------------------------------------------

if st.button("Generate Analysis"):
    enforce_limit("llm_calls", "LLM usage limit reached.")
    increment_usage("llm_calls")

    plan = get_current_plan()
    models = FREE_MODELS if plan == "free" else PRO_MODELS

    def run(prompt):
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"""
Pipeline Outputs (JSON):
{payload_json}

User Question:
{question or "None"}
"""
            }
        ]
        raw, model_used = call_groq(messages, models, temperature, max_tokens)
        return extract_json(raw["choices"][0]["message"]["content"]), model_used

    with st.spinner("Analyzing…"):
        if mode in ["Simple", "Both"]:
            out, model = run(SIMPLE_PROMPT)
            st.subheader("📝 Simple Analysis")
            st.json(out)

        if mode in ["Professional", "Both"]:
            out, model = run(PRO_PROMPT)
            st.subheader("🧪 Professional Analysis")
            st.json(out)

    st.success(f"Model used: `{model}`")
