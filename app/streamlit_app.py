from components.map_view import create_national_map
import streamlit as st


st.set_page_config(
    page_title="Roadkill Risk Dashboard",
    page_icon="🦘",
    layout="wide",
)
    
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Road Segments Scored", "99,232")
col2.metric("Wildlife Sightings", "413,000+")
col3.metric("Critical Risk Segments", "1,189", ">0.98 score")
col4.metric("Species Covered", "11")
col5.metric("States Covered", "8")


create_national_map()