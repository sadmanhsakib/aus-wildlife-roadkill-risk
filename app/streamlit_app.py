from components.map_view import (
    create_national_map,
    parse_clicked_segment,
    render_layer_controls,
)
from components.shap_panel import render_shap_panel
from streamlit_folium import st_folium
from pathlib import Path
import streamlit as st

@st.cache_data
def load_css(path: str = "app/assets/style.css") -> str:
    """Load and cache CSS styles."""
    return f"<style>{Path(path).read_text(encoding='utf-8')}</style>"


@st.cache_data
def load_html_assets() -> dict[str, str]:
    """Load header and footer HTML files."""
    assets_path = Path("app/assets")
    return {
        "header": (assets_path / "header.html").read_text(encoding="utf-8"),
        "footer": (assets_path / "footer.html").read_text(encoding="utf-8"),
    }


st.set_page_config(
    page_title="Roadkill Risk Dashboard",
    page_icon="🦘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "selected_segment" not in st.session_state:
    st.session_state.selected_segment = None

html_data = load_html_assets()

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
    """Render the interactive map with layer controls."""
    layers = render_layer_controls()
    m, map_key = create_national_map(layers)
    st.markdown('<div class="map-wrapper">', unsafe_allow_html=True)
    map_output = st_folium(
        m,
        width="100%",
        height=600,
        key=f"national-map-{map_key}",
        returned_objects=["last_object_clicked_popup", "last_object_clicked_tooltip"],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Handle segment selection from map clicks
    segment_id = parse_clicked_segment(map_output)
    if segment_id and st.session_state.selected_segment != segment_id:
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
