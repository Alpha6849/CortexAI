"""
6_AI_Analyst.py

Groq-powered AI Analyst for CortexAI.
Produces structured JSON + human-readable insights.
Grounded using real pipeline outputs (LLM-safe summaries only).
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit Cloud fallback
if GROQ_API_KEY is None:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        GROQ_API_KEY = None

if GROQ_API_KEY is None:
    st.warning("⚠️ No Groq API key found. Add GROQ_API_KEY in .env or Streamlit secrets.")

# --------------------------------------------------
# USAGE / PLAN MANAGEMENT
# --------------------------------------------------
from pipeline.usage_manager import (
    init_plan_and_usage,
    enforce_limit,
    increment_usage
)

init_plan_and_usage()

# --------------------------------------------------
# PROMPTS
# --------------------------------------------------

SIMPLE_PROMPT = (
    "You are a data analyst.\n"
    "You are given summarized pipeline outputs INCLUDING a dataset quality assessment.\n"
    "Base your reasoning strictly on these signals.\n\n"
    "Return ONLY a single valid JSON object.\n"
    "The JSON MUST start with '{' and end with '}'.\n\n"
    "Required keys:\n"
    "- summary (string, 2–4 sentences)\n"
    "- insights (array of strings)\n"
    "- recommendations (array of strings)\n"
    "- warnings (array of strings)\n"
)

PRO_PROMPT = (
    "You are a senior data scientist.\n"
    "You are given summarized pipeline outputs INCLUDING a dataset quality assessment.\n"
    "Explain results honestly, including limitations.\n\n"
    "Return ONLY a single valid JSON object.\n"
    "The JSON MUST start with '{' and end with '}'.\n\n"
    "Required keys:\n"
    "- summary (string, 3–6 sentences)\n"
    "- detailed_insights (array of objects with 'insight' and 'evidence')\n"
    "- recommendations (array of strings)\n"
    "- warnings (array of objects with 'message' and 'severity')\n"
    "- model_explanation (string)\n"
)

# --------------------------------------------------
# GROQ API
# --------------------------------------------------

def call_groq(messages, model="llama-3.1-8b-instant", temperature=0.0, max_tokens=1200):
    if GROQ_API_KEY is None:
        raise RuntimeError("GROQ_API_KEY not configured.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def collect_session_reports():
    keys = [
        "meta",
        "schema",
        "cleaning_report",
        "eda_report",
        "training_summary",
        "dataset_quality"
    ]

    reports = {}
    for k in keys:
        if k in st.session_state:
            reports[k] = st.session_state[k]

    return reports if reports else None


def build_llm_safe_reports(reports: dict) -> dict:
    safe = {}

    if "schema" in reports:
        safe["schema"] = {
            "target": reports["schema"].get("target"),
            "num_numeric_features": len(reports["schema"].get("numeric", [])),
            "num_categorical_features": len(reports["schema"].get("categorical", []))
        }

    if "cleaning_report" in reports:
        safe["cleaning_summary"] = reports["cleaning_report"]

    if "training_summary" in reports:
        safe["training_summary"] = {
            "best_model": reports["training_summary"].get("best_model"),
            "best_score": reports["training_summary"].get("best_score")
        }

    if "dataset_quality" in reports:
        safe["dataset_quality"] = reports["dataset_quality"]

    return safe


def extract_json_from_text(text: str):
    try:
        cleaned = text.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == -1:
            return None
        return json.loads(cleaned[start:end])
    except Exception:
        return None

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(page_title="AI Analyst — CortexAI", layout="wide")
st.title("🧠 AI Analyst (Groq-powered)")
st.caption("Grounded AI insights based on real AutoML pipeline outputs.")

mode = st.radio("Choose Analysis Mode:", ["Simple", "Professional", "Both"])

# --------------------------------------------------
# LOAD REPORTS
# --------------------------------------------------

reports = collect_session_reports()
if not reports:
    st.error("Run Load → Schema → Cleaning → EDA → Training first.")
    st.stop()

st.subheader("Reports Loaded:")
for k in reports.keys():
    st.write(f"✔️ {k}")

# --------------------------------------------------
# DATASET QUALITY (SHOWN ONCE)
# --------------------------------------------------

if "dataset_quality" in reports:
    dq = reports["dataset_quality"]

    st.subheader("🧠 Dataset Quality Summary")
    st.metric("Learnability Score", f"{dq['learnability_score']} / 100")
    st.info(f"**Verdict:** {dq['verdict']}")

    if dq.get("reasons"):
        st.markdown("**Key Reasons:**")
        for r in dq["reasons"]:
            st.write("•", r)

# --------------------------------------------------
# LLM CONFIG
# --------------------------------------------------

temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
max_tokens = st.number_input("Max Output Tokens", value=1200, min_value=256, max_value=4096)
custom_question = st.text_area("Optional: Ask a custom question", height=80)

safe_reports = build_llm_safe_reports(reports)
user_json = json.dumps(safe_reports, indent=2)

# --------------------------------------------------
# ENFORCE LLM LIMIT
# --------------------------------------------------

enforce_limit(
    key="llm_calls",
    message="🚫 AI Analyst limit reached for Free plan. Upgrade to Pro for unlimited AI insights."
)

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button("Generate Analysis"):

    increment_usage("llm_calls")

    prompt = SIMPLE_PROMPT if mode == "Simple" else PRO_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Here are the pipeline summaries (JSON):\n" + user_json}
    ]

    if custom_question.strip():
        messages.append({
            "role": "user",
            "content": "Extra question: " + custom_question.strip()
        })

    with st.spinner("Analyzing with Groq…"):
        try:
            raw = call_groq(
                messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens)
            )

            content = raw["choices"][0]["message"]["content"]

            parsed = extract_json_from_text(content)

            if not parsed:
                st.error("Could not parse valid JSON from model output.")
                st.code(content)
                st.stop()

            st.subheader("📝 Structured Analysis")
            st.json(parsed)

            st.download_button(
                "⬇️ Download JSON Report",
                data=json.dumps(parsed, indent=2),
                file_name="ai_analyst_report.json",
                mime="application/json"
            )

            if mode in ["Simple", "Both"]:
                st.subheader("📄 Human-readable Summary")

                if "summary" in parsed:
                    st.write(parsed["summary"])

                if "insights" in parsed:
                    st.markdown("### Insights")
                    for i in parsed["insights"]:
                        st.write("•", i)

                if "recommendations" in parsed:
                    st.markdown("### Recommendations")
                    for r in parsed["recommendations"]:
                        st.write("•", r)

                if "warnings" in parsed:
                    st.markdown("### Warnings")
                    for w in parsed["warnings"]:
                        st.write("⚠️", w)

            if "model_explanation" in parsed:
                st.subheader("🧩 Model Explanation")
                st.write(parsed["model_explanation"])

        except Exception as e:
            st.error(f"LLM Error: {e}")
