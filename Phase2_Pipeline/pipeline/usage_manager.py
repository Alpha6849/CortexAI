"""
usage_manager.py

Central plan & usage tracking for CortexAI.
Supports Free / Pro plans and Developer override.
"""

import streamlit as st

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEVELOPER_MODE = True   # 🔥 SET TRUE FOR YOU, FALSE IN PRODUCTION

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
# INIT
# --------------------------------------------------

def init_plan_and_usage():
    if "plan" not in st.session_state:
        st.session_state["plan"] = "free"


    if "usage" not in st.session_state:
        st.session_state["usage"] = {
            "uploads": 0,
            "pipeline_runs": 0,
            "llm_calls": 0
        }

    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = DEVELOPER_MODE

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_current_plan():
    return "pro" if st.session_state.get("is_admin") else st.session_state.get("plan", "free")


def get_plan_limits():
    return PLAN_LIMITS[get_current_plan()]


def increment_usage(key: str):
    if st.session_state.get("is_admin"):
        return  # 👑 admin bypass

    st.session_state["usage"][key] += 1


def check_limit(key: str):
    if st.session_state.get("is_admin"):
        return True  # 👑 unlimited

    limits = get_plan_limits()
    usage = st.session_state.get("usage", {})
    return usage.get(key, 0) < limits.get(f"{key}_per_session", 0)


def enforce_limit(key: str, message: str):
    if not check_limit(key):
        st.error(message)
        st.stop()


def get_usage_snapshot():
    limits = get_plan_limits()
    usage = st.session_state.get("usage", {})

    return {
        "uploads": (usage.get("uploads", 0), limits["uploads_per_session"]),
        "pipeline_runs": (usage.get("pipeline_runs", 0), limits["pipeline_runs_per_session"]),
        "llm_calls": (usage.get("llm_calls", 0), limits["llm_calls_per_session"]),
    }
