import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

import streamlit as st

# --------------------------------------------------
# Usage / Plan Management
# --------------------------------------------------
from pipeline.usage_manager import init_plan_and_usage

# Initialize plan + usage (safe to call every run)
init_plan_and_usage()

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="CortexAI AutoML",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Main title
# --------------------------------------------------
st.title("🤖 CortexAI — AutoML Platform")
st.write("Welcome! Use the sidebar to navigate through the pipeline.")

# --------------------------------------------------
# Sidebar: Plan Selector
# --------------------------------------------------
st.sidebar.subheader("💳 Plan")

current_plan = st.session_state["plan"]

selected_plan = st.sidebar.radio(
    "Select your plan",
    ["free", "pro"],
    index=0 if current_plan == "free" else 1
)

st.session_state["plan"] = selected_plan

# Sidebar message
st.sidebar.success("Choose a step to begin →")
