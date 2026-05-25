import json
import folium
import pandas as pd
import geopandas as gpd
import streamlit as st
import branca.colormap as cm
from folium.plugins import HeatMap, MarkerCluster


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


@st.cache_data
def load_state_stats(path: str = "data/model/road_segments_scored.parquet") -> dict:
    df = pd.read_parquet(path, columns=["state", "predicted_risk"])
    stats = {}
    for state, group in df.groupby("state"):
        stats[state] = {
            "total_segments": len(group),
            "critical_segments": int((group["predicted_risk"] > 0.98).sum()),
            "mean_risk": round(group["predicted_risk"].mean(), 4),
            "max_risk": round(group["predicted_risk"].max(), 4),
        }
    return stats


@st.cache_resource
def load_state_boundaries(
    path: str = "data/processed/state_boundaries_simplified.parquet",
) -> gpd.GeoDataFrame:
    return gpd.read_parquet(path)


def add_state_layer(m: folium.Map) -> gpd.GeoDataFrame:
    state_boundaries_gdf = load_state_boundaries()
    state_stats = load_state_stats()

    state_boundaries_gdf["total_segments"] = state_boundaries_gdf["state"].map(
        lambda s: state_stats.get(s, {}).get("total_segments", "N/A")
    )
    state_boundaries_gdf["critical_segments"] = state_boundaries_gdf["state"].map(
        lambda s: state_stats.get(s, {}).get("critical_segments", "N/A")
    )
    state_boundaries_gdf["mean_risk"] = state_boundaries_gdf["state"].map(
        lambda s: state_stats.get(s, {}).get("mean_risk", "N/A")
    )
    state_boundaries_gdf["max_risk"] = state_boundaries_gdf["state"].map(
        lambda s: state_stats.get(s, {}).get("max_risk", "N/A")
    )

    def state_style(feature):
        return {
            "fillColor": "#1a1a2e",
            "color": "#4a90d9",
            "weight": 1.5,
            "fillOpacity": 0.05,
            "opacity": 0.7,
        }

    def state_highlight(feature):
        return {
            "fillColor": "#4a90d9",
            "fillOpacity": 0.2,
            "weight": 2.5,
        }

    folium.GeoJson(
        state_boundaries_gdf,
        name="State Boundaries",
        style_function=state_style,
        highlight_function=state_highlight,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "state",
                "total_segments",
                "critical_segments",
                "mean_risk",
                "max_risk",
            ],
            aliases=[
                "State",
                "Road Segments Scored",
                "Critical Segments (≥0.98)",
                "Mean Risk Score",
                "Peak Risk Score",
            ],
            localize=True,
            sticky=True,
            style=(
                "background-color: #1a1a2e;"
                "color: white;"
                "font-family: monospace;"
                "font-size: 12px;"
                "padding: 8px;"
                "border-radius: 4px;"
                "border: 1px solid #4a90d9;"
            ),
        ),
        zoom_on_click=True,
    ).add_to(m)


def create_national_map() -> tuple:
    placements = load_sign_placement_data()

    m = folium.Map(
        location=[-25.0, 133.0],   # geographic center of Australia
        zoom_start=4,
        tiles="CartoDB positron",  # clean light basemap; no labels cluttering risk data
        prefer_canvas=True         # Canvas renderer is faster for many polygons
    )

    # ── Layer 0: State Boundaries ─────────────────────────────────────────────
    if st.checkbox("State Boundaries", value=True):
        add_state_layer(m)

    # ── Layer 1: Occurrence HeatMap ─────────────────────────────────────────────────
    if st.checkbox("Occurrence HeatMap", value=False):
        HeatMap(
            get_heatmap_data(),
            min_opacity=0.4,
            radius=12,
            blur=8,
            max_zoom=13,
        ).add_to(m)

    # ── Layer 2: High-Risk Segments ───────────────────────────────────────────
    if st.checkbox("High-Risk Road Segments", value=False):
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
    if st.checkbox("Sign Placements", value=True):
        fg = folium.FeatureGroup(name="Sign Placements", show=True)

        detail_cluster = MarkerCluster(show=True).add_to(fg)
        for lat, lon, props in placements:
            tooltip_lines = ["<b>Proposed sign location</b>"]
            for key, val in props.items():
                if val is not None:
                    tooltip_lines.append(f"{key.replace('_', ' ').title()}: {val}")

            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(
                    icon="exclamation-triangle",
                    prefix="fa",
                    color="red",
                    icon_color="white",
                ),
                tooltip=folium.Tooltip("<br>".join(tooltip_lines)),
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

    return m, placements