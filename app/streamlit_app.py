from components.map_view import create_national_map
from components.shap_panel import render_shap_panel
from streamlit_folium import st_folium
import streamlit as st


st.set_page_config(
    page_title="Roadkill Risk Dashboard",
    page_icon="🦘",
    layout="wide",
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Road Segments Scored", "99,739")
col2.metric("Wildlife Sightings", "413,000+")
col3.metric("Critical Risk Segments", "1,189", ">0.98 score")
col4.metric("Species Covered", "11")
col5.metric("States Covered", "8")


m, placements = create_national_map()

map_output = st_folium(m, width="100%", height=600, returned_objects=["last_clicked"])

if map_output["last_clicked"]:
    lat = map_output["last_clicked"]["lat"]
    lon = map_output["last_clicked"]["lng"]

    # Find the nearest placement to the click coordinates
    distances = [
        ((lat - p_lat) ** 2 + (lon - p_lon) ** 2, int(props["road_segment_id"]))
        for p_lat, p_lon, props in placements
    ]
    _, segment_id = min(distances)
    st.session_state.selected_segment = segment_id

render_shap_panel(st.session_state.get("selected_segment"))