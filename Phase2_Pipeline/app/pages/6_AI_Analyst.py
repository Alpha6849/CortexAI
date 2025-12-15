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
# ENV LOADING
# --------------------------------------------------
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Fallback for Streamlit Cloud
if GROQ_API_KEY is None:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        GROQ_API_KEY = None

if GROQ_API_KEY is None:
    st.warning("⚠️ No Groq API key found. Add GROQ_API_KEY in .env or Streamlit secrets.")

# --------------------------------------------------
# PROMPTS (DATASET-QUALITY AWARE, STRICT JSON)
# --------------------------------------------------

SIMPLE_PROMPT = (
    "You are a data analyst.\n"
    "You are given summarized pipeline outputs INCLUDING a dataset quality assessment.\n"
    "Base your reasoning strictly on these signals.\n\n"
    "Return ONLY a single valid JSON object.\n"
    "Do NOT include explanations, markdown, code fences, or extra text.\n"
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
    "You MUST respect dataset quality when explaining results.\n"
    "If learnability is low or data is imbalanced, explain limitations clearly.\n\n"
    "Return ONLY a single valid JSON object.\n"
    "Do NOT include explanations, markdown, code fences, or extra text.\n"
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
    """
    Collect raw pipeline outputs from session_state.
    """
    reports = {}
    keys = [
        "meta",
        "schema",
        "cleaning_report",
        "eda_report",
        "training_summary",
        "dataset_quality"
    ]

    found = False
    for k in keys:
        if k in st.session_state:
            reports[k] = st.session_state[k]
            found = True

    return reports if found else None


def build_llm_safe_reports(reports: dict) -> dict:
    """
    Reduce payload size for LLM consumption.
    Keeps only high-signal, low-volume fields.
    Prevents 413 Payload Too Large errors.
    """
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
    """
    Robust JSON extraction from LLM output.
    """
    try:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]

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
st.caption("Generates grounded insights based on real AutoML pipeline outputs.")

mode = st.radio("Choose Analysis Mode:", ["Simple", "Professional", "Both"])

# --------------------------------------------------
# LOAD REPORTS
# --------------------------------------------------

reports = collect_session_reports()
if not reports:
    st.error("No pipeline results found. Run Load → Schema → Cleaning → EDA → Training first.")
    st.stop()

st.subheader("Reports Loaded:")
for k in reports.keys():
    st.write(f"✔️ {k}")

# --------------------------------------------------
# LLM CONFIG
# --------------------------------------------------

temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
max_tokens = st.number_input("Max Output Tokens", value=1200, min_value=256, max_value=4096)
custom_question = st.text_area("Optional: Ask a custom question to the analyst", height=80)

llm_reports = build_llm_safe_reports(reports)
user_json = json.dumps(llm_reports, indent=2)

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button("Generate Analysis"):

    if GROQ_API_KEY is None:
        st.error("Missing GROQ_API_KEY. Add it to .env and restart.")
        st.stop()

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

            st.subheader("🔎 Raw Model Output")
            st.code(content)

            parsed = extract_json_from_text(content)

            if parsed:
                st.subheader("📝 Parsed JSON (Structured)")
                st.json(parsed)

                st.session_state["ai_analyst_parsed"] = parsed

                st.download_button(
                    "⬇️ Download JSON Report",
                    data=json.dumps(parsed, indent=2),
                    file_name="ai_analyst_report.json",
                    mime="application/json"
                )

                # Human-readable rendering
                if mode in ["Simple", "Both"]:
                    st.subheader("📄 Human-readable Summary")

                    if "summary" in parsed:
                        st.markdown("### Summary")
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

            else:
                st.error("Could not parse valid JSON from model output.")

        except Exception as e:
            st.error(f"LLM Error: {e}")
