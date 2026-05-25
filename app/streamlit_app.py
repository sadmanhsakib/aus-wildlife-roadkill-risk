from components.map_view import create_national_map
from components.shap_panel import render_shap_panel
from streamlit_folium import st_folium
from pathlib import Path
import streamlit as st
import os

@st.cache_data
def load_css(path: str = "app/assets/style.css") -> str:
    css = Path(path).read_text(encoding="utf-8")
    return f"<style>{css}</style>"


@st.cache_data
def load_html_data(path: str = "app/assets/") -> str:
    html_data = {}
    for filename in os.listdir(path):
        if filename.endswith(".html"):
            filepath = os.path.join(path, filename)
            html_data.update(
                {
                    filename.removesuffix(".html"): 
                        Path(filepath).read_text(encoding="utf-8")
                }
            )
    return html_data


st.set_page_config(
    page_title="Roadkill Risk Dashboard",
    page_icon="🦘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

html_data = load_html_data()

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(load_css(), unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(html_data["header"], unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Road Segments Scored", "99,739")
col2.metric("Wildlife Sightings", "413,000+")
col3.metric("Critical Segments", "1,189", ">0.98 risk score")
col4.metric("Species Covered", "11")
col5.metric("States Covered", "8")

# ── Map + SHAP layout ─────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Risk Map</p>', unsafe_allow_html=True)

map_col, shap_col = st.columns([3, 1], gap="medium")

with map_col:
    st.markdown('<div class="map-wrapper">', unsafe_allow_html=True)
    m, placements = create_national_map()
    map_output = st_folium(
        m, width="100%", height=580, returned_objects=["last_clicked"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if map_output["last_clicked"]:
        lat = map_output["last_clicked"]["lat"]
        lon = map_output["last_clicked"]["lng"]
        distances = [
            ((lat - p_lat) ** 2 + (lon - p_lon) ** 2, int(props["road_segment_id"]))
            for p_lat, p_lon, props in placements
        ]
        _, segment_id = min(distances)
        st.session_state.selected_segment = segment_id

with shap_col:
    st.markdown(
        '<p class="section-label">Feature Attribution</p>', unsafe_allow_html=True
    )
    render_shap_panel(st.session_state.get("selected_segment"))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(html_data["footer"], unsafe_allow_html=True)
