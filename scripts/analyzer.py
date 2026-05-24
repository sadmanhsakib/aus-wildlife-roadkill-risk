"""
Wildlife Roadkill Risk Analyzer
===============================
This module processes wildlife sighting data and environmental rasters to compute 
composite risk scores for Australian road segments. It generates proxy labels 
for machine learning models by combining ecological suitability with road exposure.
"""

import gc, os, time
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import KNN, lag_spatial
from rasterio.sample import sample_gen
import fetcher


LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"


def main():
    visualize()
    
    return
    p = "backup/"

    for filename in os.listdir(p):
        if filename.endswith(".parquet"):
            df = pd.read_parquet(f"{p}{filename}")

            # Prepare spatial data (joins and projections)
            sightings_gdf = prepare_spatial_data(df)

            # Engineer features and risk labels
            sightings_gdf = engineer_features(sightings_gdf)

            sightings_gdf.to_parquet(f"sightings/{filename}", index=False)
            print(f"✅ sightings/{filename} processed successfully.\n")

    fetcher.merge(
        "sightings.parquet",
        [f"sightings/{f}" for f in os.listdir("sightings/")],
        has_geometry=True,
    )

    engineer_proxy_risk_labels()


def prepare_spatial_data(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Standardizes raw sighting data into a projected geospatial format 
    and performs spatial joins against administrative and road networks.

    Args:
        df: Input DataFrame containing raw sightings with lat/lon.

    Returns:
        GeoDataFrame enriched with state and nearest road segment metadata.
    """
    sightings_projected_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[LONGITUDE_COLUMN], df[LATITUDE_COLUMN]),
        crs="EPSG:4326",
    ).to_crs(epsg=32754)

    print("⏳ Loading state boundary data....")
    state_boundaries_projected_gdf = gpd.read_parquet(
        "data/processed/state_boundaries.parquet"
    ).to_crs(epsg=32754)
    print("⏳ Loading road network data....")
    road_networks_projected_gdf = gpd.read_parquet(
        "data/processed/road_networks.parquet"
    ).to_crs(epsg=32754)

    # Intersection with state boundaries for regional categorization
    sightings_joined = gpd.sjoin(
        sightings_projected_gdf,
        state_boundaries_projected_gdf,
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])

    # Nearest-neighbor join to associate sightings with specific road segments
    print("🔄 Calculating distance to the nearest road....")
    sightings_joined = gpd.sjoin_nearest(
        sightings_joined,
        road_networks_projected_gdf[
            [
                "road_segment_id",
                "road_class",
                "speed_limit",
                "traffic_proxy",
                "geometry",
            ]
        ],
        how="left",
        distance_col="distance_to_road",
    ).drop(columns=["index_right"])

    # Deduplicate in case of equidistant segments
    sightings_joined = sightings_joined[
        ~sightings_joined.index.duplicated(keep="first")
    ]

    del sightings_projected_gdf, state_boundaries_projected_gdf, road_networks_projected_gdf
    gc.collect()

    return sightings_joined


def engineer_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Performs feature engineering on sightings, including categorical encoding 
    and environmental data enrichment.

    Args:
        gdf: Spatially joined sightings GeoDataFrame.

    Returns:
        Enriched GeoDataFrame with normalized features.
    """
    gdf["state"] = gdf["state"].map(fetcher.STATE_CODES)

    # Overlay vegetation indices (NDVI) onto sighting points
    gdf = sample_raster_at_points(gdf, col_name="ndvi")

    # Filter invalid vegetation values (outside standard 0 to 1 range)
    gdf = gdf[gdf["ndvi"] <= 1.0]

    gdf["distance_to_road"] = round(gdf["distance_to_road"].astype(float), 2)

    return gdf


def sample_raster_at_points(gdf, col_name):
    """
    Extracts raster values at specific geographic coordinates.
    Handles coordinate transformation if raster CRS differs from input.

    Args:
        gdf: DataFrame containing longitude/latitude columns.
        col_name: Name of the resulting column in the output DataFrame.

    Returns:
        DataFrame with an additional column containing sampled raster values.
    """
    coords = list(zip(gdf["longitude"], gdf["latitude"]))

    with rasterio.open("data/processed/ndvi_median.tif") as src:
        # Align coordinate systems if necessary
        if src.crs.to_epsg() != 4326:
            from pyproj import Transformer

            transformer = Transformer.from_crs(
                "EPSG:4326", src.crs.to_epsg(), always_xy=True
            )
            coords = [transformer.transform(lon, lat) for lon, lat in coords]

        print("🔄 Fetching vegetation data for each coordinate....")
        sampled = list(sample_gen(src, coords, indexes=1))

    gdf[col_name] = np.array(sampled, dtype=float)
    return gdf


def minmax(series, q_hi=0.99):
    """Cap outliers at q_hi percentile before scaling."""
    hi = series.quantile(q_hi)
    clipped = series.clip(upper=hi)
    return (clipped - clipped.min()) / (clipped.max() - clipped.min())


def engineer_proxy_risk_labels():
    """
    Core algorithmic engine that aggregates point-level sightings into road-segment 
    risk labels. Employs a composite scoring model and spatial smoothing.

    The method:
    1. Aggregates ecological indicators (abundance, richness, habitat) by road segment.
    2. Computes a Road Exposure score (speed, traffic intensity, proximity).
    3. Blends raw risk with a Spatial Lag to prevent overfitting to specific patches.
    4. Generates a percentile-based proxy risk label.
    """
    sightings_gdf = gpd.read_parquet("sightings.parquet")

    # Aggregate species and habitat metrics at the road segment level
    road_segment_df = (
        sightings_gdf.groupby("road_segment_id")
        .agg(
            state=("state", "first"),
            sighting_count=("species", "count"),
            species_richness=("species", "nunique"),
            mean_body_mass_weight=("body_mass_weight", "mean"),
            mean_nocturnal_weight=("nocturnal_weight", "mean"),
            mean_peak_season_weight=("peak_season_weight", "mean"),
            mean_distance_to_road=("distance_to_road", "mean"),
            mean_ndvi=("ndvi", "mean"),
            road_class=("road_class", "first"),
            speed_limit=("speed_limit", "first"),
            traffic_proxy=("traffic_proxy", "first"),
        )
        .reset_index()
    )
    del sightings_gdf
    gc.collect()

    road_networks_gdf = gpd.read_parquet("data/processed/road_networks.parquet")

    # adding the geometry for each road
    road_segment_df = road_segment_df.merge(
        road_networks_gdf[["road_segment_id", "geometry"]],
        on="road_segment_id",
        how="left",
    )
    del road_networks_gdf
    gc.collect()

    road_segment_gdf = gpd.GeoDataFrame(
        road_segment_df, geometry="geometry", crs="EPSG:4326"
    )
    del road_segment_df
    gc.collect()

    # Ecological Score: Represents the biological likelihood of wildlife presence
    # Weighting prioritizes abundance and vegetation density
    road_segment_gdf["ecological_score"] = (
        0.30 * minmax(road_segment_gdf["sighting_count"])
        + 0.15 * minmax(road_segment_gdf["species_richness"])
        + 0.20 * minmax(road_segment_gdf["mean_ndvi"])
        + 0.15 * minmax(road_segment_gdf["mean_peak_season_weight"])
        + 0.10 * minmax(road_segment_gdf["mean_nocturnal_weight"])
        + 0.10 * minmax(road_segment_gdf["mean_body_mass_weight"])
    )

    # Road Exposure: Represents the physical risk factor of the road infrastructure
    road_segment_gdf["proximity"] = 1 - minmax(road_segment_gdf["mean_distance_to_road"])
    road_segment_gdf["road_exposure_score"] = (
        0.35 * minmax(road_segment_gdf["speed_limit"])
        + 0.35 * minmax(road_segment_gdf["proximity"])
        + 0.30 * minmax(road_segment_gdf["traffic_proxy"])
    )

    # Interaction model: Risk is defined by the intersection of presence and exposure
    road_segment_gdf["raw_risk"] = (
        road_segment_gdf["ecological_score"] * road_segment_gdf["road_exposure_score"]
    )

    # Spatial Regularization: Use K-Nearest Neighbors to smooth risk across contiguous segments.
    # This prevents the model from 'memorizing' outlier segments and encourages spatial generalization.
    w = KNN.from_dataframe(road_segment_gdf, k=5)
    w.transform = "r"  # Row-standardization for equal neighbor influence

    # Calculate spatial lag (average risk of 5 nearest neighbors)
    road_segment_gdf["spatial_lag"] = lag_spatial(
        w, road_segment_gdf["raw_risk"].values
    )

    # Harmonic blending: 70% direct signal, 30% neighborhood context
    road_segment_gdf["blended_risk"] = (
        0.7 * road_segment_gdf["raw_risk"] + 0.3 * road_segment_gdf["spatial_lag"]
    )

    # Final Proxy Label: Rank-transformed to a 0-1 probability scale
    road_segment_gdf["proxy_risk"] = road_segment_gdf["blended_risk"].rank(pct=True)

    road_segment_gdf.to_parquet("data/processed/road_segments.parquet", index=False)


def visualize(gdf: gpd.GeoDataFrame = None):
    """
    Renders road segment proxy risk as LineString geometries over
    state boundaries and the road network basemap.

    Args:
        gdf: GeoDataFrame of road segments with proxy_risk and LineString geometry.
    """
    if not gdf:
        gdf = gpd.read_parquet("data/processed/road_segments.parquet")

    print("⏳ Loading basemap layers....")
    state_boundaries_gdf = gpd.read_parquet("data/processed/state_boundaries.parquet")
    road_networks_gdf = gpd.read_parquet("data/processed/road_networks.parquet")

    sns.set_theme(style="whitegrid", palette="deep")
    _, ax = plt.subplots(figsize=(12, 10))

    # Basemap rendering
    state_boundaries_gdf.plot(ax=ax, color="green", alpha=0.2)
    road_networks_gdf.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5)

    plot_data = gdf.copy()
    plot_data["Risk"] = np.where(
        plot_data["proxy_risk"] > 0.8, "High risk", "Low risk"
    )

    risk_styles = [
        ("Low risk", "#3498db", 0.4),
        ("High risk", "#e74c3c", 1.0),
    ]
    for risk_label, color, linewidth in risk_styles:
        subset = plot_data[plot_data["Risk"] == risk_label]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            color=color,
            linewidth=linewidth,
            alpha=0.9,
            label=risk_label,
        )

    ax.legend(title="Risk")

    ax.set_title("Road Segment Proxy Risk across Australia", fontsize=14)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"✅ Execution completed in {time.time() - start_time:.2f} seconds")
