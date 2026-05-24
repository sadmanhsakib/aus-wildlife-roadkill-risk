import json
import folium
import geopandas as gpd
import streamlit as st
import branca.colormap as cm
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium


@st.cache_data
def load_sign_placement_data() -> list[tuple]:
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
    return placements


@st.cache_data
def load_stored_gdf_data() -> list[tuple]:
    scored_gdf = gpd.read_parquet("data/model/road_segments_scored.parquet")
    scored_gdf = scored_gdf.to_crs(epsg=4326)
    return scored_gdf


@st.cache_data
def get_heatmap_data() -> list:
    heatmap_gdf = load_stored_gdf_data()
    heatmap_gdf["value"] = (
        heatmap_gdf["sighting_count"] + heatmap_gdf["species_richness"]
    ) / 2

    point = heatmap_gdf.geometry.representative_point()
    heatmap_gdf["lon"] = point.x
    heatmap_gdf["lat"] = point.y

    heat_data = []
    for _, row in heatmap_gdf.iterrows():
        heat_data.append([row["lat"], row["lon"], row["value"]])
    return heat_data


@st.cache_data
def get_highrisk_gdf() -> gpd.GeoDataFrame:
    scored_gdf = load_stored_gdf_data()
    return scored_gdf[scored_gdf["predicted_risk"] > 0.98]


def create_national_map():
    placements = load_sign_placement_data()

    m = folium.Map(
        location=[-25.0, 133.0],   # geographic center of Australia
        zoom_start=4,
        tiles="CartoDB positron",  # clean light basemap; no labels cluttering risk data
        prefer_canvas=True         # Canvas renderer is faster for many polygons
    )

    # ── Layer 1: Risk HeatMap ─────────────────────────────────────────────────
    if st.checkbox("Risk HeatMap", value=True):
        HeatMap(
            get_heatmap_data(),
            min_opacity=0.4,
            radius=12,
            blur=8,
            max_zoom=13,
        ).add_to(m)

    # ── Layer 2: High-Risk Segments ───────────────────────────────────────────
    if st.checkbox("High-Risk Segments", value=False):
        colormap = cm.LinearColormap(
            colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"],
            vmin=0.98,
            vmax=1.0,
            caption="Predicted Risk Score"
        )

        def style_fn(feature):
            risk = feature["properties"].get("predicted_risk", 0.98)
            return {
                "fillColor": colormap(risk),
                "color": colormap(risk),
                "weight": 2,
                "fillOpacity": 0.5,
                "opacity": 0.8,
            }

        def highlight_fn(feature):
            return {
                "fillOpacity": 0.9,
                "weight": 4,
            }
        folium.GeoJson(
            get_highrisk_gdf(),
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["road_segment_id", "predicted_risk", "state"],
                aliases=["Segment:", "Risk Score:", "State:"],
                localize=True,
                sticky=True,
            ),
        ).add_to(m)

        colormap.add_to(m)

    # ── Layer 3: Sign Placements ──────────────────────────────────────────────
    if st.checkbox("Sign Placements", value=False):
        fg = folium.FeatureGroup(name="Sign Placements", show=True)

        detail_cluster = MarkerCluster(show=True).add_to(fg)
        for lat, lon, props in placements:
            tooltip_lines = ["<b>Proposed sign location</b>"]
            for key, val in props.items():
                if val is not None:
                    tooltip_lines.append(f"{key.replace('_', ' ').title()}: {val}")

            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(icon="warning-sign", prefix="glyphicon", color="red", icon_color="white"),
                tooltip=folium.Tooltip("<br>".join(tooltip_lines))
            ).add_to(detail_cluster)

        fg.add_to(m)

    title_html = """
    <div style="position:fixed; top:12px; left:50%; transform:translateX(-50%);
        background:white; padding:8px 18px; border-radius:8px;
        box-shadow:0 2px 8px rgba(0,0,0,0.2); font-family:sans-serif;
        font-size:15px; font-weight:600; z-index:9999; color:#1a1a1a;">
        🦘 Australian Wildlife-Vehicle Collision Risk
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.plugins.Fullscreen(position="bottomright").add_to(m)
    folium.plugins.MeasureControl(position="bottomright").add_to(m)
    folium.plugins.MiniMap(position="bottomleft", toggle_display=True).add_to(m)

    # ── Render ────────────────────────────────────────────────────────────────
    st_folium(m, width="100%", height=600, returned_objects=[])
