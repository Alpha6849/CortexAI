import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

import streamlit as st


# Page configuration
st.set_page_config(
    page_title="CortexAI AutoML",
    page_icon="🤖",
    layout="wide"
)

# Main title
st.title("🤖 CortexAI — AutoML Platform")
st.write("Welcome! Use the sidebar to navigate through the pipeline.")

# Sidebar message
st.sidebar.success("Choose a step to begin →")
