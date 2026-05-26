import copy
import json
import re

import folium
import pandas as pd
import geopandas as gpd
import streamlit as st
import branca.colormap as cm
from folium.plugins import HeatMap


# Performance tuning constants
HEATMAP_MAX_POINTS = 15_000  # Reduced for faster rendering
HIGHRISK_THRESHOLD = 0.98


@st.cache_data
def load_sign_placements() -> list[tuple[float, float, int, dict]]:
    """Load sign placement data efficiently from GeoJSON.
    
    Returns:
        List of (lat, lon, segment_id, properties) tuples for each sign.
    """
    with open("data/model/sign_placements.geojson", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    placements = []
    for feature in data["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        props = feature["properties"]
        placements.append((lat, lon, int(props["road_segment_id"]), props))
    
    return placements


def _add_sign_placements(m: folium.Map, placements: list[tuple]) -> None:
    """Add sign markers to the map with tooltips."""
    fg = folium.FeatureGroup(name="Sign Placements", show=True)
    
    for lat, lon, segment_id, props in placements:
        tooltip_html = f"""
        <div style="font-family: 'Inter', -apple-system, sans-serif; font-size: 13px;">
            <div style="font-weight: 600; margin-bottom: 6px; color: #1d1d1f;">
                Segment ID: {segment_id}
            </div>
            <div style="color: #6e6e73; margin-bottom: 4px;">
                State: {props.get('state', 'N/A')}
            </div>
            <div style="color: #ff3b30; font-weight: 500;">
                Risk Score: {props.get('predicted_risk', 0):.4f}
            </div>
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="#ff3b30",
            fill=True,
            fill_color="#ff3b30",
            fill_opacity=0.8,
            weight=2,
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
        ).add_to(fg)
    
    fg.add_to(m)


@st.cache_data
def load_road_segments_lite() -> pd.DataFrame:
    """Load only essential columns from road segments for better performance.
    
    Returns:
        DataFrame with geometry, risk scores, and basic attributes.
    """
    columns = [
        "road_segment_id", "predicted_risk", "state",
        "sighting_count", "species_richness", "geometry"
    ]
    df = gpd.read_parquet("data/model/road_segments_scored.parquet", columns=columns)
    
    # Ensure correct CRS
    if df.crs is None or df.crs.to_epsg() != 4326:
        df = df.to_crs(epsg=4326)
    
    return df


@st.cache_data
def get_heatmap_data(max_points: int = HEATMAP_MAX_POINTS) -> list[list[float]]:
    """Generate heatmap data from wildlife sightings and species richness.
    
    Args:
        max_points: Maximum number of points to include (for performance).
        
    Returns:
        List of [lat, lon, intensity] values for the heatmap.
    """
    df = load_road_segments_lite()
    
    # Calculate intensity from sightings and species richness
    df["intensity"] = (df["sighting_count"] + df["species_richness"]) / 2
    
    # Get representative points for line geometries
    points = df.geometry.representative_point()
    
    # Create coordinate dataframe
    coords = pd.DataFrame({
        "lat": points.y,
        "lon": points.x,
        "intensity": df["intensity"]
    })
    
    # Sample if too many points
    if len(coords) > max_points:
        coords = coords.sample(n=max_points, random_state=42)
    
    return coords[["lat", "lon", "intensity"]].values.tolist()


@st.cache_data
def get_highrisk_geojson() -> str:
    """Get GeoJSON for high-risk road segments (risk > 0.98)."""
    df = load_road_segments_lite()
    highrisk = df[df["predicted_risk"] > HIGHRISK_THRESHOLD]
    return highrisk.to_json()


@st.cache_data
def calculate_state_stats() -> dict:
    """Calculate statistics for each state.
    
    Returns:
        Dictionary mapping state names to their statistics.
    """
    df = load_road_segments_lite()
    
    stats = {}
    for state, group in df.groupby("state"):
        stats[state] = {
            "total_segments": len(group),
            "critical_segments": int((group["predicted_risk"] > HIGHRISK_THRESHOLD).sum()),
            "mean_risk": round(group["predicted_risk"].mean(), 4),
            "max_risk": round(group["predicted_risk"].max(), 4),
        }
    
    return stats


@st.cache_data
def load_state_boundaries() -> gpd.GeoDataFrame:
    """Load simplified state boundaries for faster rendering."""
    return gpd.read_parquet("data/processed/state_boundaries_simplified.parquet")


@st.cache_data
def get_state_boundaries_geojson() -> str:
    """Get state boundaries with statistics as GeoJSON."""
    boundaries = load_state_boundaries()
    stats = calculate_state_stats()
    
    # Add statistics to boundaries
    boundaries["total_segments"] = boundaries["state"].map(
        lambda s: stats.get(s, {}).get("total_segments", "N/A")
    )
    boundaries["critical_segments"] = boundaries["state"].map(
        lambda s: stats.get(s, {}).get("critical_segments", "N/A")
    )
    boundaries["mean_risk"] = boundaries["state"].map(
        lambda s: stats.get(s, {}).get("mean_risk", "N/A")
    )
    boundaries["max_risk"] = boundaries["state"].map(
        lambda s: stats.get(s, {}).get("max_risk", "N/A")
    )
    
    return boundaries.to_json()


def _create_map_key(show_state: bool, show_heatmap: bool, 
                    show_highrisk: bool, show_signs: bool) -> str:
    """Generate a unique key for map caching based on visible layers."""
    return f"{show_state}-{show_heatmap}-{show_highrisk}-{show_signs}"


def _state_style(_feature) -> dict:
    """Style function for state boundaries."""
    return {
        "fillColor": "#1a1a2e",
        "color": "#4a90d9",
        "weight": 1.5,
        "fillOpacity": 0.05,
        "opacity": 0.7,
    }


def _state_highlight(_feature) -> dict:
    """Highlight style for state boundaries on hover."""
    return {
        "fillColor": "#4a90d9",
        "fillOpacity": 0.2,
        "weight": 2.5,
    }


def _highrisk_highlight(_feature) -> dict:
    """Highlight style for high-risk segments on hover."""
    return {"fillOpacity": 0.9, "weight": 4}


@st.cache_resource(show_spinner="Building map…")
def _build_national_map(show_state: bool, show_heatmap: bool,
                        show_highrisk: bool, show_signs: bool) -> folium.Map:
    """Build the Folium map with requested layers.
    
    Args:
        show_state: Show state boundaries layer
        show_heatmap: Show wildlife occurrence heatmap
        show_highrisk: Show high-risk road segments
        show_signs: Show proposed sign placements
        
    Returns:
        Configured Folium map object
    """
    # Create base map
    m = folium.Map(
        location=[-25.0, 133.0],
        zoom_start=4,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # Add state boundaries layer
    if show_state:
        folium.GeoJson(
            get_state_boundaries_geojson(),
            name="State Boundaries",
            style_function=_state_style,
            highlight_function=_state_highlight,
            tooltip=folium.GeoJsonTooltip(
                fields=["state", "total_segments", "critical_segments", "mean_risk", "max_risk"],
                aliases=["State", "Road Segments", "Critical Segments", "Mean Risk", "Peak Risk"],
                localize=True,
                sticky=True,
                style=(
                    "background: rgba(255, 255, 255, 0.98); "
                    "backdrop-filter: blur(10px); "
                    "color: #1d1d1f; "
                    "font-family: 'Inter', -apple-system, sans-serif; "
                    "font-size: 13px; "
                    "padding: 12px 16px; "
                    "border-radius: 8px; "
                    "border: 1px solid #e5e5ea; "
                    "box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);"
                ),
            ),
            zoom_on_click=True,
        ).add_to(m)

    # Add heatmap layer
    if show_heatmap:
        HeatMap(
            get_heatmap_data(),
            min_opacity=0.4,
            radius=12,
            blur=8,
            max_zoom=13,
        ).add_to(m)

    # Add high-risk segments layer
    if show_highrisk:
        colormap = cm.LinearColormap(
            colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"],
            vmin=HIGHRISK_THRESHOLD,
            vmax=1.0,
            caption="Predicted Risk Score",
        )

        def style_fn(feature):
            risk = feature["properties"].get("predicted_risk", HIGHRISK_THRESHOLD)
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
                aliases=["Segment ID", "Risk Score", "State"],
                localize=True,
                sticky=True,
                style=(
                    "background: rgba(255, 255, 255, 0.98); "
                    "backdrop-filter: blur(10px); "
                    "color: #1d1d1f; "
                    "font-family: 'Inter', -apple-system, sans-serif; "
                    "font-size: 13px; "
                    "padding: 12px 16px; "
                    "border-radius: 8px; "
                    "border: 1px solid #e5e5ea; "
                    "box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);"
                ),
            ),
        ).add_to(m)
        colormap.add_to(m)

    # Add sign placements layer
    if show_signs:
        _add_sign_placements(m, load_sign_placements())

    # Add map title overlay
    title_html = """
    <div style="position: fixed; 
                top: 16px; 
                left: 50%; 
                transform: translateX(-50%);
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 10px 24px;
                border-radius: 12px;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                font-family: 'Inter', -apple-system, sans-serif;
                font-size: 14px;
                font-weight: 600;
                color: #1d1d1f;
                z-index: 9999;
                letter-spacing: -0.01em;">
        🦘 Australian Wildlife-Vehicle Collision Risk Map
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add fullscreen control
    folium.plugins.Fullscreen(position="bottomright").add_to(m)

    return m


def render_layer_controls() -> dict[str, bool]:
    """Render checkboxes for map layer toggles.
    
    Returns:
        Dictionary of layer visibility flags
    """
    return {
        "show_state": st.checkbox("State Boundaries", value=True, key="layer_state"),
        "show_heatmap": st.checkbox("Occurrence HeatMap", value=False, key="layer_heatmap"),
        "show_highrisk": st.checkbox("High-Risk Road Segments", value=False, key="layer_highrisk"),
        "show_signs": st.checkbox("Sign Placements", value=True, key="layer_signs"),
    }


def create_national_map(layers: dict[str, bool]) -> tuple[folium.Map, str]:
    """Create a Folium map with the specified layers.
    
    Args:
        layers: Dictionary of layer visibility flags
        
    Returns:
        Tuple of (map object, cache key for the current layer configuration)
    """
    cache_key = _create_map_key(**layers)
    # Deep copy prevents st_folium from mutating the cached map
    return copy.deepcopy(_build_national_map(**layers)), cache_key


def _extract_segment_id(text: str) -> int | None:
    """Extract road segment ID from Folium tooltip/popup HTML.
    
    Args:
        text: HTML or plain text from map interaction
        
    Returns:
        Segment ID if found, None otherwise
    """
    if not text:
        return None

    # Try various patterns to extract segment ID
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

    # Fallback: find any long integer (likely a segment ID)
    match = re.search(r"\b(\d{5,})\b", text)
    if match:
        return int(match.group(1))

    return None


def parse_clicked_segment(map_output: dict | None) -> int | None:
    """Parse segment ID from st_folium click event.
    
    Args:
        map_output: Output dictionary from st_folium
        
    Returns:
        Segment ID if a sign was clicked, None otherwise
    """
    if not map_output:
        return None

    # Check both popup and tooltip for segment ID
    for key in ("last_object_clicked_popup", "last_object_clicked_tooltip"):
        segment_id = _extract_segment_id(map_output.get(key) or "")
        if segment_id:
            return segment_id

    return None
