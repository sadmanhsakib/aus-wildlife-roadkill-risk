from components.map_view import (
    create_national_map,
    parse_clicked_segment,
    render_layer_controls,
    warmup_map_caches,
)
from components.shap_panel import render_shap_panel, warmup_shap_caches
from streamlit_folium import st_folium
from pathlib import Path
import streamlit as st
import os

@st.cache_data
def load_css(path: str = "app/assets/style.css") -> str:
    css = Path(path).read_text(encoding="utf-8")
    return f"<style>{css}</style>"


@st.cache_data
def load_html_data(path: str = "app/assets/") -> dict[str, str]:
    html_data = {}
    for filename in os.listdir(path):
        if filename.endswith(".html"):
            filepath = os.path.join(path, filename)
            html_data[filename.removesuffix(".html")] = Path(filepath).read_text(
                encoding="utf-8"
            )
    return html_data


st.set_page_config(
    page_title="Roadkill Risk Dashboard",
    page_icon="🦘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "selected_segment" not in st.session_state:
    st.session_state.selected_segment = None

# Warm caches once per browser session (speeds map toggles and SHAP clicks).
if not st.session_state.get("_caches_warm"):
    with st.spinner("Loading map data…"):
        warmup_map_caches()
        warmup_shap_caches()
    st.session_state._caches_warm = True

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


@st.fragment
def _map_panel() -> None:
    layers = render_layer_controls()
    m, map_key = create_national_map(layers)
    st.markdown('<div class="map-wrapper">', unsafe_allow_html=True)
    map_output = st_folium(
        m,
        width="100%",
        height=600,
        key=f"national-map-{map_key}",
        returned_objects=[
            "last_object_clicked_popup",
            "last_object_clicked_tooltip",
        ],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    segment_id = parse_clicked_segment(map_output)
    if (
        segment_id is not None
        and st.session_state.selected_segment != segment_id
    ):
        st.session_state.selected_segment = segment_id
        st.rerun()


with map_col:
    _map_panel()

with shap_col:
    st.markdown(
        """<p class="section-label">Feature Attribution</p>
            Click on any sign to get the feature attribution.
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.selected_segment:
        if st.button("✕ Clear", key="clear_shap"):
            st.session_state.selected_segment = None
            st.rerun()
    render_shap_panel(st.session_state.selected_segment)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(html_data["footer"], unsafe_allow_html=True)
