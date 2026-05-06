import time, gc, os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.sample import sample_gen
import fetcher


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
    for filename in os.listdir("backup"):
        if filename.endswith(".parquet"):
            df = pd.read_parquet(f"backup/{filename}")
            sightings_gdf = prepare_spatial_data(df)

            sightings_gdf.to_parquet(f"sightings/{filename}", index=False)

    print("Starting Visualization.....")
    visualize(None)


def prepare_spatial_data(df: pd.DataFrame) -> gpd.GeoDataFrame:
    # converting pandas DataFrame to GeoDataFrame
    sightings = gpd.GeoDataFrame(
        df,
        # creating the geometry column
        # longitude is X and latitude is Y
        geometry=gpd.points_from_xy(df[LONGITUDE_COLUMN], df[LATITUDE_COLUMN]),
        crs="EPSG:4326",
    )
    sightings_projected = sightings.to_crs("EPSG:32754")
    # freeing up memory space
    del sightings
    gc.collect()

    print("Loading state boundaries from the parquet file....")
    # loading the state data
    states_projected = gpd.read_parquet("data/states_projected.parquet")
    sightings_projected = gpd.sjoin(
        sightings_projected,
        states_projected,
        how="inner",
        predicate="within",
    )
    # freeing up memory space
    del states_projected
    gc.collect()

    # dropping unnecessary column
    sightings_projected = sightings_projected.drop(columns=["index_right"])

    # replacing the state names with state codes
    sightings_projected["state"] = sightings_projected["state"].map(fetcher.STATE_CODES)

    # getting the vegetation column
    sightings_projected = sample_raster_at_points(sightings_projected, col_name="ndvi")

    # dropping data with wrong vegetation data
    sightings_projected = sightings_projected[sightings_projected["ndvi"] <= 1.0]

    print("Loading road network from the parquet file....")
    # loading the roads data
    road_network_projected = gpd.read_parquet("data/australia_projected.parquet")

    print("Calculating distance to the nearest road....")
    # spatial join sightings to nearest road
    sightings_with_road_data = gpd.sjoin_nearest(
        sightings_projected,
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
    # freeing up memory space
    del sightings_projected, road_network_projected
    gc.collect()
    # dropping the duplicate sightings (in case of multiple nearest roads at same distance)
    sightings_with_road_data = sightings_with_road_data[
        ~sightings_with_road_data.index.duplicated(keep="first")
    ]
    # dropping unnecessary column
    sightings_with_road_data = sightings_with_road_data.drop(columns=["index_right"])

    # rounding the distance
    sightings_with_road_data["distance_to_road"] = round(
        sightings_with_road_data["distance_to_road"].astype(float), 2
    )

    print("Adding risk labels....")
    # adding risk labels
    sightings_with_road_data["risk_label"] = sightings_with_road_data[
        "distance_to_road"
    ].apply(lambda x: 1 if x <= 500 else 0)

    return sightings_with_road_data


def sample_raster_at_points(df, col_name):
    """Samples a GeoTIFF raster at each lat/lon point in df."""

    # getting the coordinates
    coords = list(zip(df["longitude"], df["latitude"]))

    with rasterio.open("data/ndvi_median_australia.tif") as src:
        # Reproject coords if raster CRS differs from WGS84
        if src.crs.to_epsg() != 4326:
            from pyproj import Transformer

            transformer = Transformer.from_crs(
                "EPSG:4326", src.crs.to_epsg(), always_xy=True
            )
            coords = [transformer.transform(lon, lat) for lon, lat in coords]

        print("Fetching vegetation data for each coordinate....")
        sampled = list(sample_gen(src, coords, indexes=1))

    values = np.array(sampled, dtype=float)

    df[col_name] = values
    return df


def visualize(df: gpd.GeoDataFrame):
    if not df:
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

    print("Loading the .parquet files....")
    # loading the gdfs for the background
    states_projected = gpd.read_parquet("data/states_projected.parquet")
    roads_projected = gpd.read_parquet("data/australia_projected.parquet")

    # setting the background theme
    sns.set_theme(style="whitegrid", palette="deep")

    # creating the figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # plotting the whole map
    print("Plotting the State Map....")
    states_projected.plot(ax=ax, color="green", alpha=0.2)

    # plotting the roads and roads with buffer
    print("Plotting the roads....")
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
    print("Plotting the sightings....")
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
    print(f"Time taken: {end_time - start_time} seconds")
