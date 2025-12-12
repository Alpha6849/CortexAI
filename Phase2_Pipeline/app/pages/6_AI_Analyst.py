import os
from dotenv import load_dotenv

# Load .env (local) or Streamlit secrets (deployment)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if in Streamlit Cloud, fallback to secrets
if GROQ_API_KEY is None:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
