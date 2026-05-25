import copy
import json
import re

import folium
import pandas as pd
import geopandas as gpd
import streamlit as st
import branca.colormap as cm
from folium.plugins import HeatMap


# Max heatmap points sent to the browser (99k points is slow to render in Leaflet).
HEATMAP_MAX_POINTS = 20_000


@st.cache_data
def get_sign_placements_geojson(
    path: str = "data/model/sign_placements.geojson",
) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_sign_placements() -> list[tuple[float, float, int, dict]]:
    """(lat, lon, segment_id, properties) for each proposed sign."""
    placements = []
    for feature in get_sign_placements_geojson()["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        props = feature["properties"]
        placements.append((lat, lon, int(props["road_segment_id"]), props))
    return placements


def _add_sign_placements(m: folium.Map, placements: list[tuple]) -> None:
    """Circle markers with a plain tooltip streamlit-folium can return on click."""
    fg = folium.FeatureGroup(name="Sign Placements", show=True)
    for lat, lon, segment_id, props in placements:
        tooltip_html = "<br>".join(
            [
                f"<b>segment_id:{segment_id}</b>",
                f"State: {props.get('state', '')}",
                f"Risk: {props.get('predicted_risk', 0):.4f}",
            ]
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="#c0392b",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.9,
            weight=1,
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
        ).add_to(fg)
    fg.add_to(m)


@st.cache_data
def load_stored_gdf_data() -> gpd.GeoDataFrame:
    scored_gdf = gpd.read_parquet("data/model/road_segments_scored.parquet")
    if scored_gdf.crs is None or scored_gdf.crs.to_epsg() != 4326:
        scored_gdf = scored_gdf.to_crs(epsg=4326)
    return scored_gdf


@st.cache_data
def get_heatmap_data(max_points: int = HEATMAP_MAX_POINTS) -> list[list[float]]:
    heatmap_gdf = load_stored_gdf_data()
    values = (
        heatmap_gdf["sighting_count"] + heatmap_gdf["species_richness"]
    ).to_numpy() / 2
    points = heatmap_gdf.geometry.representative_point()
    coords = pd.DataFrame({"lat": points.y, "lon": points.x, "value": values})

    if len(coords) > max_points:
        coords = coords.sample(n=max_points, random_state=42)

    return coords[["lat", "lon", "value"]].values.tolist()


@st.cache_data
def get_highrisk_geojson() -> str:
    scored_gdf = load_stored_gdf_data()
    highrisk = scored_gdf[scored_gdf["predicted_risk"] > 0.98]
    return highrisk.to_json()


@st.cache_data
def load_state_stats() -> dict:
    gdf = load_stored_gdf_data()
    df = gdf[["state", "predicted_risk"]]
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


@st.cache_data
def get_state_boundaries_geojson() -> str:
    state_boundaries_gdf = load_state_boundaries().copy()
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
    return state_boundaries_gdf.to_json()


def _map_cache_key(
    show_state: bool,
    show_heatmap: bool,
    show_highrisk: bool,
    show_signs: bool,
) -> str:
    return f"{show_state}-{show_heatmap}-{show_highrisk}-{show_signs}"


def _state_style(_feature) -> dict:
    return {
        "fillColor": "#1a1a2e",
        "color": "#4a90d9",
        "weight": 1.5,
        "fillOpacity": 0.05,
        "opacity": 0.7,
    }


def _state_highlight(_feature) -> dict:
    return {
        "fillColor": "#4a90d9",
        "fillOpacity": 0.2,
        "weight": 2.5,
    }


def _highrisk_highlight(_feature) -> dict:
    return {"fillOpacity": 0.9, "weight": 4}


@st.cache_resource(show_spinner="Building map…")
def _build_national_map(
    show_state: bool,
    show_heatmap: bool,
    show_highrisk: bool,
    show_signs: bool,
) -> folium.Map:
    m = folium.Map(
        location=[-25.0, 133.0],
        zoom_start=4,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    if show_state:
        folium.GeoJson(
            get_state_boundaries_geojson(),
            name="State Boundaries",
            style_function=_state_style,
            highlight_function=_state_highlight,
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

    if show_heatmap:
        HeatMap(
            get_heatmap_data(),
            min_opacity=0.4,
            radius=12,
            blur=8,
            max_zoom=13,
        ).add_to(m)

    if show_highrisk:
        colormap = cm.LinearColormap(
            colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"],
            vmin=0.98,
            vmax=1.0,
            caption="Predicted Risk Score",
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

        folium.GeoJson(
            get_highrisk_geojson(),
            style_function=style_fn,
            highlight_function=_highrisk_highlight,
            tooltip=folium.GeoJsonTooltip(
                fields=["road_segment_id", "predicted_risk", "state"],
                aliases=["Segment:", "Risk Score:", "State:"],
                localize=True,
                sticky=True,
            ),
        ).add_to(m)
        colormap.add_to(m)

    if show_signs:
        _add_sign_placements(m, load_sign_placements())

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

    return m


def render_layer_controls() -> dict[str, bool]:
    """Layer toggles live outside the cached map builder so only flags are inputs."""
    return {
        "show_state": st.checkbox("State Boundaries", value=True, key="layer_state"),
        "show_heatmap": st.checkbox(
            "Occurrence HeatMap", value=False, key="layer_heatmap"
        ),
        "show_highrisk": st.checkbox(
            "High-Risk Road Segments", value=False, key="layer_highrisk"
        ),
        "show_signs": st.checkbox("Sign Placements", value=True, key="layer_signs"),
    }


def create_national_map(layers: dict[str, bool]) -> tuple[folium.Map, str]:
    """
    Return a fresh Folium map for st_folium plus a stable key for layer state.
    Deep-copies the cached map so st_folium cannot mutate the cached instance.
    """
    key = _map_cache_key(**layers)
    return copy.deepcopy(_build_national_map(**layers)), key


def _extract_segment_id(text: str) -> int | None:
    """Parse road_segment_id from Folium popup/tooltip HTML or plain text."""
    if not text:
        return None

    patterns = [
        r"segment_id:\s*(\d+)",
        r"segment[_\s]*id[:\s]*</[^>]+>\s*<td[^>]*>\s*(\d+)",
        r"segment[_\s]*id[:\s]*(\d+)",
        r"Id:</th>\s*<td[^>]*>\s*(\d+)",
        r"Id:\s*(\d+)",
        r"Road Segment Id:\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Last resort: first long integer in the payload
    match = re.search(r"\b(\d{5,})\b", text)
    if match:
        return int(match.group(1))

    return None


def parse_clicked_segment(map_output: dict | None) -> int | None:
    """
    Read segment id from st_folium click data.

    streamlit-folium does not return GeoJSON properties in last_object_clicked
    (only lat/lng). Use popup or tooltip text instead.
    """
    if not map_output:
        return None

    for key in ("last_object_clicked_popup", "last_object_clicked_tooltip"):
        segment_id = _extract_segment_id(map_output.get(key) or "")
        if segment_id is not None:
            return segment_id

    return None


@st.cache_resource
def warmup_map_caches() -> None:
    """Pre-build default map layers and data so first interaction is faster."""
    _build_national_map(True, False, False, True)
    get_state_boundaries_geojson()
    load_sign_placements()
