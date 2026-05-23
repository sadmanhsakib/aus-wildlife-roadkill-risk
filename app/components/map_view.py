import json
import folium
import folium.plugins
import streamlit as st
import geopandas as gpd
import branca.colormap as cm
from streamlit_folium import st_folium


@st.cache_data
def load_sign_placements() -> list[tuple]:
    """
    Load sign placements once per session using Streamlit's caching.
    
    Data structure:
    - GeoJSON file with features containing:
        * coordinates (longitude, latitude)
        * properties dictionary with metadata
    """
    with open("data/model/sign_placements.geojson", "r") as file:
        data = json.load(file)
    
    placements = []
    for feature in data["features"]:
        coods = feature["geometry"]["coordinates"]  # GeoJSON is [lon, lat]
        lon = coods[0]
        lat = coods[1]
        props = feature["properties"]
        placements.append((lat, lon, props))
    
    gdf = gpd.read_parquet("data/model/road_segments_scored.parquet")
    gdf = gdf.to_crs(epsg=4326)
    gdf["value"] = (gdf["sighting_count"] + gdf["species_richness"]) / 2
    point = gdf.geometry.representative_point()
    gdf["lon"] = point.x
    gdf["lat"] = point.y
    gdf = gdf[["lon", "lat", "value"]]
    return gdf, placements


def create_national_map():
    gdf, placements = load_sign_placements()

    m = folium.Map(
        location=[-25.0, 133.0],   # geographic center of Australia
        zoom_start=4,
        tiles="CartoDB positron",  # clean light basemap; no labels cluttering risk data
        prefer_canvas=True         # Canvas renderer is faster for many polygons
    )

    # ── Layer 1: Risk HeatMap ─────────────────────────────────────────────────
    if st.checkbox("Risk HeatMap", value=True):
        heat_data = []
        for index, row in gdf.iterrows():
            heat_data.append([row["lat"], row["lon"], row["value"]])

        folium.plugins.HeatMap(
            heat_data,
            min_opacity=0.4,
            radius=14,
            blur=9,
            max_zoom=13,
        ).add_to(m)
    
    # ── Layer 2: High-Risk Segments ───────────────────────────────────────────
    if st.checkbox("High-Risk Segments", value=False):
        colormap = cm.LinearColormap(
            colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"],
            vmin=0.985,
            vmax=1.0,
            caption="Predicted Risk Score"
        )

        for lat, lon, props in placements:
            risk = props.get("predicted_risk")
            if risk is None:
                continue

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=colormap(risk),
                fill=True,
                fill_color=colormap(risk),
                fill_opacity=0.7,
                tooltip=folium.Tooltip(f"Risk: {risk:.4f}")
            ).add_to(m)

        colormap.add_to(m)
    
    # ── Layer 3: Sign Placements ──────────────────────────────────────────────
    if st.checkbox("Sign Placements", value=False):
        for lat, lon, props in placements:
            tooltip_lines = ["<b>Proposed sign location</b>"]
            for key, val in props.items():
                if val is not None:
                    tooltip_lines.append(f"{key.replace('_', ' ').title()}: {val}")

            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(
                    icon="warning-sign",
                    prefix="glyphicon",
                    color="red",
                    icon_color="white"
                ),
                tooltip=folium.Tooltip("<br>".join(tooltip_lines))
            ).add_to(m)

    # ── Render ────────────────────────────────────────────────────────────────
    st_folium(m, width="100%", height=600, returned_objects=[])