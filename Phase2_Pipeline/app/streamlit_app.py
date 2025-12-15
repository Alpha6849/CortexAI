import sys
import os
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from pipeline.usage_manager import (
    init_plan_and_usage,
    get_current_plan,
    get_usage_snapshot
)

# --------------------------------------------------
# INIT
# --------------------------------------------------
init_plan_and_usage()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="CortexAI AutoML",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CortexAI — AutoML Platform")
st.write("Welcome! Use the sidebar to navigate through the pipeline.")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.subheader("💳 Plan")

current_plan = get_current_plan()
st.sidebar.write(f"**Current Plan:** `{current_plan.upper()}`")

# 🔒 Developer-only toggle
if st.session_state.get("is_admin"):
    st.sidebar.markdown("### 👑 Developer Mode")
    st.sidebar.radio(
        "Force Plan View",
        ["free", "pro"],
        index=1 if current_plan == "pro" else 0,
        key="plan"
    )

# --------------------------------------------------
# USAGE METERS (FREE USERS)
# --------------------------------------------------
if current_plan == "free":
    st.sidebar.markdown("### 📊 Usage")

    usage = get_usage_snapshot()

    st.sidebar.write(f"📂 Uploads: {usage['uploads'][0]} / {usage['uploads'][1]}")
    st.sidebar.write(f"🤖 Trainings: {usage['pipeline_runs'][0]} / {usage['pipeline_runs'][1]}")
    st.sidebar.write(f"🧠 AI Calls: {usage['llm_calls'][0]} / {usage['llm_calls'][1]}")

    st.sidebar.markdown("---")
    st.sidebar.info("Upgrade to **Pro** for unlimited access 🚀")

st.sidebar.success("Choose a step to begin →")
