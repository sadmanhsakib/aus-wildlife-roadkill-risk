import fetcher
import time, gc, os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN, lag_spatial
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.sample import sample_gen


LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"

MAIN_STATES = (
    "New South Wales",
    "Victoria",
    "Queensland",
    "Western Australia",
    "South Australia",
    "Tasmania",
    "Australian Capital Territory",
    "Northern Territory",
)


def main():
    calculate_risk_score()

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
            print(f"✅ {filename} processed successfully.\n")


def prepare_spatial_data(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Handles GDF conversion, CRS projection, and core spatial joins (States & Roads)."""
    # converting pandas DataFrame to GeoDataFrame
    sightings = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[LONGITUDE_COLUMN], df[LATITUDE_COLUMN]),
        crs="EPSG:4326",
    )
    sightings_projected = sightings.to_crs("EPSG:32754")

    # freeing up memory space
    del sightings
    gc.collect()

    print("⏳ Loading boundary and network data....")
    states_projected = gpd.read_parquet("data/processed/states_projected.parquet")
    road_network_projected = gpd.read_parquet(
        "data/processed/australia_projected.parquet"
    )

    # Spatial join with States
    sightings_joined = gpd.sjoin(
        sightings_projected,
        states_projected,
        how="inner",
        predicate="within",
    )
    # dropping unnecessary column
    sightings_joined = sightings_joined.drop(columns=["index_right"])

    # spatial join with nearest Road
    print("🔄 Calculating distance to the nearest road....")
    sightings_joined = gpd.sjoin_nearest(
        sightings_joined,
        road_network_projected[
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
    )
    # dropping unnecessary column
    sightings_joined = sightings_joined.drop(columns=["index_right"])

    # dropping the duplicate sightings (in case of multiple nearest roads at same distance)
    sightings_joined = sightings_joined[
        ~sightings_joined.index.duplicated(keep="first")
    ]

    # freeing up memory space
    del sightings_projected, states_projected, road_network_projected
    gc.collect()

    return sightings_joined


def engineer_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Handles feature enrichment: state mapping, NDVI sampling, and risk labeling."""
    # replacing the state names with state codes
    gdf["state"] = gdf["state"].map(fetcher.STATE_CODES)

    # sampling vegetation (NDVI) data
    gdf = sample_raster_at_points(gdf, col_name="ndvi")

    # cleaning NDVI values
    gdf = gdf[gdf["ndvi"] <= 1.0]

    # rounding the road distance
    gdf["distance_to_road"] = round(gdf["distance_to_road"].astype(float), 2)

    return gdf


def sample_raster_at_points(df, col_name):
    """Samples a GeoTIFF raster at each lat/lon point in df."""

    # getting the coordinates
    coords = list(zip(df["longitude"], df["latitude"]))

    with rasterio.open("data/processed/ndvi_median_australia.tif") as src:
        # Reproject coords if raster CRS differs from WGS84
        if src.crs.to_epsg() != 4326:
            from pyproj import Transformer

            transformer = Transformer.from_crs(
                "EPSG:4326", src.crs.to_epsg(), always_xy=True
            )
            coords = [transformer.transform(lon, lat) for lon, lat in coords]

        print("🔄 Fetching vegetation data for each coordinate....")
        sampled = list(sample_gen(src, coords, indexes=1))

    values = np.array(sampled, dtype=float)

    df[col_name] = values
    return df


def minmax(series):
    return (series - series.min()) / (series.max() - series.min())


def calculate_risk_score():
    df = pd.read_parquet("sightings.parquet")

    # grouping the sightings by road segment
    road_segment_df = (
        df.groupby("road_segment_id")
        .agg(
            state=("state", "first"),
            sighting_count=("species", "count"),
            species_richness=("species", "nunique"),
            mean_body_mass_weight=("body_mass_weight", "mean"),
            mean_nocturnal_weight=("nocturnal_weight", "mean"),
            mean_peak_season_weight=("peak_season_weight", "mean"),
            mean_ndvi=("ndvi", "mean"),
            road_class=("road_class", "first"),
            speed_limit=("speed_limit", "first"),
            traffic_proxy=("traffic_proxy", "first"),
            distance_to_road=("distance_to_road", "mean"),
        )
        .reset_index()
    )

    roads_projected = gpd.read_parquet("data/processed/australia_projected.parquet")

    # getting the geometry for each road segment
    road_segment_df = road_segment_df.merge(
        roads_projected[["road_segment_id", "geometry"]],
        on="road_segment_id",
        how="left",
    )
    # freeing up memory space
    del roads_projected
    gc.collect()

    road_segment_gdf = gpd.GeoDataFrame(
        road_segment_df, geometry="geometry", crs="EPSG:32754"
    )
    # freeing up memory space
    del road_segment_df
    gc.collect()

    # calculating ecological score - based on species richness, abundance, and habitat quality
    road_segment_gdf["ecological_score"] = (
        0.30 * minmax(road_segment_gdf["sighting_count"])  # density matters most
        + 0.20 * minmax(road_segment_gdf["mean_ndvi"])  # vegetation = habitat
        + 0.15 * minmax(road_segment_gdf["species_richness"])
        + 0.15 * minmax(road_segment_gdf["mean_peak_season_weight"])
        + 0.10 * minmax(road_segment_gdf["mean_nocturnal_weight"])
        + 0.10 * minmax(road_segment_gdf["mean_body_mass_weight"])
    )

    # calculating proximity to road - based on distance to road
    road_segment_gdf["proximity"] = 1 - minmax(road_segment_gdf["distance_to_road"])

    # calculating road exposure score - based on speed limit, traffic proxy, and proximity to road
    road_segment_gdf["road_exposure_score"] = (
        0.35 * minmax(road_segment_gdf["speed_limit"])
        + 0.35 * minmax(road_segment_gdf["proximity"])
        + 0.30 * minmax(road_segment_gdf["traffic_proxy"])
    )
    # multiplying the ecological score and road exposure score
    # if either of the scores is zero, the raw risk will be zero
    road_segment_gdf["raw_risk"] = (
        road_segment_gdf["ecological_score"] * road_segment_gdf["road_exposure_score"]
    )

    """Injecting neighbourhood context into the label
        So that the model can't just memorize the training data
        and can generalize to unseen data"""

    # K=5 means each segment's 5 closest neighbours influence its label
    w = KNN.from_dataframe(road_segment_gdf, k=5)
    w.transform = "r"  # row-standardise so weights sum to 1

    # averages the 5 nearest neighbour's risk
    road_segment_gdf["spatial_lag"] = lag_spatial(
        w, road_segment_gdf["raw_risk"].values
    )

    # blending the raw risk with the spatial lag - to prevent the model from memorizing the training data
    road_segment_gdf["blended_risk"] = (
        0.7 * road_segment_gdf["raw_risk"] + 0.3 * road_segment_gdf["spatial_lag"]
    )

    # normalizing the data to a scale of 0 to 1
    road_segment_gdf["proxy_risk"] = road_segment_gdf["blended_risk"].rank(pct=True)

    road_segment_gdf = road_segment_gdf.drop(
        columns=[
            "ecological_score",
            "proximity",
            "road_exposure_score",
            "raw_risk",
            "spatial_lag",
            "blended_risk",
        ]
    )

    road_segment_gdf.to_parquet("road_segment_labels.parquet", index=False)


def visualize(gdf: gpd.GeoDataFrame):
    if not gdf:
        df = pd.read_parquet("sightings.parquet")

        # converting pandas DataFrame to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            # creating the geometry column
            # longitude is X and latitude is Y
            geometry=gpd.points_from_xy(df[LONGITUDE_COLUMN], df[LATITUDE_COLUMN]),
            crs="EPSG:4326",
        )
        gdf = gdf.to_crs("EPSG:32754")

    print("⏳ Loading the .parquet files....")
    # loading the gdfs for the background
    states_projected = gpd.read_parquet("data/processed/states_projected.parquet")
    roads_projected = gpd.read_parquet("data/processed/australia_projected.parquet")

    # setting the background theme
    sns.set_theme(style="whitegrid", palette="deep")

    # creating the figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # plotting the whole map
    print("🔄 Plotting the State Map....")
    states_projected.plot(ax=ax, color="green", alpha=0.2)

    # plotting the roads and roads with buffer
    print("🔄 Plotting the roads....")
    roads_projected.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5)

    # creating a copy of the modeling dataframe
    sightings_plot_data = gdf.copy()

    # adding x and y coordinates
    sightings_plot_data["x"] = sightings_plot_data.geometry.x
    sightings_plot_data["y"] = sightings_plot_data.geometry.y

    # adding risk labels
    sightings_plot_data["Risk"] = sightings_plot_data["risk_label"].map(
        {1: "High risk", 0: "Low risk"}
    )

    # plotting the sightings using seaborn for styled scatter points
    print("🔄 Plotting the sightings....")
    sns.scatterplot(
        data=sightings_plot_data,
        x="x",
        y="y",
        hue="Risk",
        hue_order=["High risk", "Low risk"],
        palette={"High risk": "#e74c3c", "Low risk": "#3498db"},
        size="Risk",
        sizes={"High risk": 20, "Low risk": 12},
        alpha=0.9,
        ax=ax,
    )

    # labeling the map
    ax.set_title("Sightings across Australia", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"✅ Time taken: {end_time - start_time} seconds")
