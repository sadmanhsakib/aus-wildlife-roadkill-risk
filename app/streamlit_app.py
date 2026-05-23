from components.map_view import create_national_map
import streamlit as st


st.set_page_config(
    page_title="Roadkill Risk Dashboard",
    page_icon="🦘",
    layout="wide",
)
    
col1, col2 = st.columns([3, 1])
with col1:
    create_national_map()