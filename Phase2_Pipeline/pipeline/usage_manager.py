"""
usage_manager.py

Central plan & usage tracking for CortexAI.
Used to enforce Free / Pro feature limits.
"""

import streamlit as st


# --------------------------------------------------
# PLAN DEFINITIONS
# --------------------------------------------------

PLAN_LIMITS = {
    "free": {
        "uploads_per_session": 2,
        "pipeline_runs_per_session": 2,
        "llm_calls_per_session": 1,
        "max_rows": 50_000
    },
    "pro": {
        "uploads_per_session": float("inf"),
        "pipeline_runs_per_session": float("inf"),
        "llm_calls_per_session": float("inf"),
        "max_rows": 500_000
    }
}


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

def init_plan_and_usage():
    """
    Initialize plan and usage counters in session state.
    Safe to call on every page.
    """
    if "plan" not in st.session_state:
        st.session_state["plan"] = "free"

    if "usage" not in st.session_state:
        st.session_state["usage"] = {
            "uploads": 0,
            "pipeline_runs": 0,
            "llm_calls": 0
        }


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_current_plan():
    return st.session_state.get("plan", "free")


def get_plan_limits():
    plan = get_current_plan()
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


def increment_usage(key: str):
    if "usage" not in st.session_state:
        init_plan_and_usage()

    st.session_state["usage"][key] += 1


def check_limit(key: str):
    """
    Returns True if usage is allowed, False otherwise.
    """
    limits = get_plan_limits()
    usage = st.session_state.get("usage", {})

    return usage.get(key, 0) < limits.get(f"{key}_per_session", 0)


def enforce_limit(key: str, message: str):
    """
    Hard stop if usage exceeded.
    """
    if not check_limit(key):
        st.error(message)
        st.stop()
