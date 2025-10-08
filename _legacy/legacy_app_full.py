"""LEGACY MONOLITH SNAPSHOT

Moved from repository root (legacy_app_full.py) on 2025-10-08 to `_legacy/`.
Kept for historical reference only. Not part of active runtime.

To run (debug / comparison only):
    streamlit run _legacy/legacy_app_full.py

The full original body has been truncated in this snapshot to reduce maintenance.
Use git history for the complete prior implementation if required.
"""

from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Legacy Water Balance (Monolith)")
st.title("Legacy Monolithic App (Snapshot)")
st.warning("Legacy snapshot only. Use the modular app (app.py) for current features.")
