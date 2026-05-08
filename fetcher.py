import os, time, gc, glob
import httpx, asyncio, rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

# GBIF KEYS
KANGAROO_RED_KEY = 12019022
KANGAROO_GREY_KEY = 5219981
WALLABY_SWAMP_KEY = 2440149
WALLABY_RED_NECKED_KEY = 9589697
WOMBAT_KEY = 2440301
KOALA_KEY = 2440012
POSSUM_BRUSHTAIL_KEY = 2440254
POSSUM_RINGTAIL_KEY = 2440062
BANDICOOT_BROWN_KEY = 2435311
ECHIDNA_KEY = 2433378
PLATYPUS_KEY = 2433376
# Scientific Names for ALA
KANGAROO_RED_SCIENTIFIC_NAME = "Osphranter rufus"
KANGAROO_GREY_SCIENTIFIC_NAME = "Macropus giganteus"
WALLABY_SWAMP_SCIENTIFIC_NAME = "Wallabia bicolor"
WALLABY_RED_NECKED_SCIENTIFIC_NAME = "Notamacropus rufogriseus"
WOMBAT_SCIENTIFIC_NAME = "Vombatus ursinus"
KOALA_SCIENTIFIC_NAME = "Phascolarctos cinereus"
POSSUM_BRUSHTAIL_SCIENTIFIC_NAME = "Trichosurus vulpecula"
POSSUM_RINGTAIL_SCIENTIFIC_NAME = "Pseudocheirus peregrinus"
BANDICOOT_BROWN_SCIENTIFIC_NAME = "Isoodon obesulus"
ECHIDNA_SCIENTIFIC_NAME = "Tachyglossus aculeatus"
PLATYPUS_SCIENTIFIC_NAME = "Ornithorhynchus anatinus"
# Base URLs
GBIF_URL = "https://api.gbif.org/v1/occurrence/search"
ALA_URL = "https://biocache-ws.ala.org.au/ws/occurrences/search"

STATE_CODES = {
    "New South Wales": "NSW",
    "Queensland": "QLD",
    "Victoria": "VIC",
    "Tasmania": "TAS",
    "Australian Capital Territory": "ACT",
    "Northern Territory": "NT",
    "South Australia": "SA",
    "Western Australia": "WA",
}

SEASON_MAP = {
    1: "Summer",
    2: "Summer",
    3: "Autumn",
    4: "Autumn",
    5: "Autumn",
    6: "Winter",
    7: "Winter",
    8: "Winter",
    9: "Spring",
    10: "Spring",
    11: "Spring",
    12: "Summer",
}

PEAK_SEASON_MAP = {
    "Osphranter rufus": [9, 10, 11, 12],
    "Macropus rufus": [9, 10, 11, 12],
    "Macropus giganteus": [9, 10, 11, 12],
    "Wallabia bicolor": [9, 10, 11],
    "Notamacropus rufogriseus": [10, 11, 12, 1],
    "Vombatus ursinus": [3, 4, 5, 6, 7, 8],
    "Phascolarctos cinereus": [10, 11, 12, 1, 2],
    "Trichosurus vulpecula": [3, 4, 9, 10],
    "Pseudocheirus peregrinus": [4, 5, 9, 10],
    "Isoodon obesulus": [8, 9, 10, 11],
    "Tachyglossus aculeatus": [6, 7, 8, 9],
    "Ornithorhynchus anatinus": [9, 10, 11],
}

BODY_MASS_WEIGHT = {
    "Osphranter rufus": 1.00,  # Red kangaroo       ~85kg
    "Macropus giganteus": 0.90,  # Eastern grey       ~66kg
    "Wallabia bicolor": 0.65,  # Swamp wallaby      ~20kg
    "Notamacropus rufogriseus": 0.60,  # Red-necked wallaby ~17kg
    "Vombatus ursinus": 0.70,  # Common wombat      ~35kg
    "Phascolarctos cinereus": 0.55,  # Koala              ~12kg
    "Trichosurus vulpecula": 0.30,  # Brushtail possum   ~4kg
    "Pseudocheirus peregrinus": 0.25,  # Ringtail possum    ~1kg
    "Isoodon obesulus": 0.35,  # S. brown bandicoot ~1.5kg
    "Tachyglossus aculeatus": 0.40,  # Echidna            ~6kg
    "Ornithorhynchus anatinus": 0.45,  # Platypus           ~2kg
}

NOCTURNAL = {
    "Osphranter rufus": 0.7,
    "Macropus giganteus": 0.7,
    "Wallabia bicolor": 0.9,
    "Notamacropus rufogriseus": 0.9,
    "Vombatus ursinus": 0.95,
    "Phascolarctos cinereus": 0.85,
    "Trichosurus vulpecula": 0.95,
    "Pseudocheirus peregrinus": 0.95,
    "Isoodon obesulus": 0.90,
    "Tachyglossus aculeatus": 0.30,
    "Ornithorhynchus anatinus": 0.80,
}


def main():
    files = os.listdir("sightings")
    files = [os.path.join("sightings", file) for file in files]

    merge("sightings.parquet", files, shouldDelete=False)


async def get_gbif_data(species_key: int, state: str) -> str:
    offset = 0
    results = []

    async with httpx.AsyncClient() as client:
        while True:
            params = {
                "taxonKey": species_key,
                "country": "AU",
                "hasCoordinate": "true",
                "stateProvince": f"{state}",
                "year": [2026, 2025, 2024, 2023, 2022, 2021, 2020],
                "limit": 300,
                "offset": offset,
            }

            response = await client.get(GBIF_URL, params=params)

            # checking if the request was successful
            if response.status_code == 200:
                data = response.json()
                results.extend(data["results"])

                print(f"Data Pulled: {offset}")
                # stopping if it's the end of the dataset
                if data["endOfRecords"] or offset > 100:
                    break
                offset += 300
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return 1
        # for avoiding HTTP 429 error
        await asyncio.sleep(1.0)

    if results:
        file_name = (
            f"{results[0]['species'].lower().replace(' ', '_')}_sightings_gbif.csv"
        )
        # exporting the collected data to a csv file
        df = pd.DataFrame(results)
        # only keeping the required rows
        df = df[
            [
                "species",
                "month",
                "year",
                "decimalLatitude",
                "decimalLongitude",
            ]
        ]
        df.to_csv(file_name, index=False)
        print(f"✅Data exported to {file_name} successfully. ")

        return file_name
    else:
        print("No results found.")
        return None


async def get_ala_data(species_scientific_name: str, state: str) -> str:
    offset = 0
    results = []

    async with httpx.AsyncClient() as client:
        while True:
            params = {
                "q": species_scientific_name,
                "fq": [
                    "country:Australia",
                    "year:[2020 TO 2026]",
                    f"stateProvince:{state}",
                ],
                "pageSize": 1000,  # records per page (max 1000)
                "startIndex": offset,  # for pagination
                "fl": "scientificName,month,year,decimalLatitude,decimalLongitude",  # fields to return
            }
            headers = {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }

            # sending the requests
            response = await client.get(
                ALA_URL, params=params, headers=headers, follow_redirects=True
            )

            # checking if the request was successful
            if response.status_code == 200:
                data = response.json()
                results.extend(data["occurrences"])

                print(f"Data Pulled: {offset}")
                try:
                    # check if we've reached the end of the dataset to avoid data loss
                    total_records = data.get("totalRecords", 0)
                    if not data.get("occurrences") or offset + 1000 >= total_records:
                        break
                    # temporary break for testing purposes
                    elif offset > 10:
                        break
                    offset += 1000
                except (KeyError, TypeError):
                    # catch potential missing fields or invalid types to prevent infinite loops
                    break
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return 1
            # for avoiding HTTP 429 error
            await asyncio.sleep(1.0)

    if results:
        file_name = f"{results[0]['scientificName'].lower().replace(' ', '_')}_sightings_ala.csv"

        # exporting the collected data to a csv file
        df = pd.DataFrame(results)
        df.to_csv(file_name, index=False)
        print(f"✅Data exported to {file_name} successfully. ")

        return file_name
    else:
        print("No results found.")
        return None


def clean_DataFrame(file_name: str):
    column_schema = [
        "species",
        "month",
        "year",
        "decimalLatitude",
        "decimalLongitude",
    ]

    # reading the csv file
    df = pd.read_csv(file_name)

    # for ALA data
    try:
        # renmaing the ala specific column
        df = df.rename(
            columns={
                "scientificName": "species",
            }
        )
    # for GBIF data
    except KeyError:
        pass

    # ordering the column in correct schema
    df = df[column_schema]

    # renmaing the other columns
    df = df.rename(
        columns={
            "decimalLatitude": "latitude",
            "decimalLongitude": "longitude",
        }
    )

    # removing rows with missing values
    df = df.dropna(subset=["latitude", "longitude", "year", "month"])

    # removing rows with older sighting data
    df = df.drop(df[df["year"] < 2020].index)

    # dropping rows outside Australia
    df = df.drop(df[df["latitude"] < -44].index)
    df = df.drop(df[df["latitude"] > -10].index)
    df = df.drop(df[df["longitude"] < 113].index)
    df = df.drop(df[df["longitude"] > 154].index)

    # removing duplicates
    df = df.drop_duplicates(subset=["latitude", "longitude", "year", "month"])

    # exporting the csv file
    df.to_csv(f"{file_name}", index=False)
    print(f"✅Data exported to {file_name} successfully. ")


def merge(new_file_name: str, file_names: list, shouldDelete=True):
    df_list = []

    # loading all DataFrames in a single list
    for file_name in file_names:
        if file_name.endswith(".csv"):
            df_list.append(pd.read_csv(file_name))
        elif file_name.endswith(".parquet"):
            df_list.append(pd.read_parquet(file_name))

    # merging the dfs all together
    merged_df = pd.concat(df_list, ignore_index=True)

    # removing duplicates
    merged_df = merged_df.drop_duplicates(
        subset=["species", "month", "year", "latitude", "longitude"]
    )

    if shouldDelete:
        # removing the old files
        for file_name in file_names:
            os.remove(file_name)

    # exporting the file
    if new_file_name.endswith(".csv"):
        merged_df.to_csv(new_file_name, index=False)
    elif new_file_name.endswith(".parquet"):
        merged_df.to_parquet(new_file_name, index=False)

    print(f"✅{file_names} merged into {new_file_name} successfully. ")


def to_parquet(path: str):
    file_names = []

    try:
        for file_name in os.listdir(path):
            if not file_name.endswith(".csv"):
                continue

            file_name = os.path.join(path, file_name)
            file_names.append(file_name)
    except NotADirectoryError:
        file_names.append(path)

    for file_name in file_names:
        new_file_name = file_name.replace(".csv", ".parquet")

        df = pd.read_csv(file_name)
        df.to_parquet(new_file_name, index=False)

        os.remove(file_name)
        print(f"✅Data exported to {new_file_name} successfully. ")


def enrich(path: str):
    file_names = []

    try:
        for file_name in os.listdir(path):
            if not file_name.endswith(".parquet"):
                continue

            file_name = os.path.join(path, file_name)
            file_names.append(file_name)
    except NotADirectoryError:
        file_names.append(path)

    for file_name in file_names:
        # loading the df
        df = pd.read_parquet(file_name)

        # adding the season column
        df["season"] = df["month"].map(SEASON_MAP)
        # Adding species-specific weights
        df["body_mass_weight"] = df["species"].map(BODY_MASS_WEIGHT)
        df["nocturnal_weight"] = df["species"].map(NOCTURNAL)

        # Calculate peak season weight
        df["peak_season_weight"] = [
            1.3 if m in PEAK_SEASON_MAP.get(s, []) else 1.0
            for s, m in zip(df["species"], df["month"])
        ]

        df.to_parquet(file_name, index=False)
        print(f"{file_name} enriched successfully. ")


def prepare_road_network():
    print("Loading the road network...")

    # loading the roads data
    road_network = gpd.read_file(
        "data/raw/australia.gpkg",
        layer="gis_osm_roads_free",
        columns=["osm_id", "name", "fclass", "geometry"],
    )

    # speed limit and traffic volume by the road types
    # traffic proxy is a rating out of 5
    # : very busy traffic, 1: very light traffic
    FCLASS_DEFAULTS = {
        "motorway": (110, 5),
        "trunk": (100, 4),
        "primary": (100, 3),
        "secondary": (80, 2),
        "tertiary": (80, 2),
        "unclassified": (60, 1),
        "track": (40, 1),
        "residential": (50, 1),
    }

    # filtering to keep the relevant roads
    road_network = road_network[road_network["fclass"].isin(FCLASS_DEFAULTS.keys())]
    # adding the speed limit
    road_network["speed_zone"] = road_network["fclass"].map(
        lambda x: FCLASS_DEFAULTS[x][0]
    )
    # adding the traffic proxy
    road_network["traffic_proxy"] = road_network["fclass"].map(
        lambda x: FCLASS_DEFAULTS[x][1]
    )

    # renaming the column
    road_network.rename(
        columns={
            "osm_id": "road_segment_id",
            "fclass": "road_class",
            "name": "road_name",
            "speed_zone": "speed_limit",
        },
        inplace=True,
    )

    # converting it to projected system for accurate distance calculations
    road_network_projected = road_network.to_crs("EPSG:32754")
    # freeing up memory space
    del road_network
    gc.collect()

    road_network_projected.to_parquet("data/australia_projected.parquet", index=False)
    print("✅Road network parsed and saved to australia_projected.parquet")


def prepare_state_network():
    print("Loading the state data...")

    # loading the whole map
    states = gpd.read_file(
        "data/raw/SA1_2021_AUST_GDA2020.shp", columns=["STE_NAME21", "geometry"]
    )

    # filtering to just the main states
    states = states[states["STE_NAME21"].isin(STATE_CODES.keys())]
    states = states.rename(columns={"STE_NAME21": "state"})
    states_projected = states.to_crs("EPSG:32754")
    # freeing up memory space
    del states
    gc.collect()

    states_projected.to_parquet("data/states_projected.parquet", index=False)
    print("✅State network parsed and saved to states_projected.parquet")


def build_ndvi_median_composite():
    """
    Takes all monthly NDVI GeoTIFFs from AppEEARS and
    produces a single median composite raster using windowed processing
    to avoid memory errors.
    """
    print("Merging the .tif files (memory-efficient block-wise processing)...")

    tif_folder = "data/raw/vegetation/"
    output_path = "data/ndvi_median_australia.tif"

    # finding and storing the file_paths of all .tif files
    tif_files = sorted(glob.glob(os.path.join(tif_folder, "*.tif")))
    if not tif_files:
        print("No .tif files found in vegetation folder.")
        return

    print(f"Found {len(tif_files)} monthly files. ")

    # Open all source files
    srcs = [rasterio.open(f) for f in tif_files]

    try:
        # Use metadata from first file
        meta = srcs[0].meta.copy()
        meta.update(dtype="float32", count=1, nodata=np.nan)

        # Create output file
        with rasterio.open(output_path, "w", **meta) as dst:
            # Iterate through the destination file in blocks (windows)
            windows = [window for _, window in dst.block_windows(1)]
            num_windows = len(windows)

            for i, window in enumerate(windows):
                if i % 10 == 0:
                    print(f"Processing block {i+1}/{num_windows}...")

                block_arrays = []
                for src in srcs:
                    # Read the current window from each source file
                    data = src.read(1, window=window).astype(np.float32)
                    data[data == -28672] = np.nan  # MODIS nodata value
                    data = data * 0.0001  # MODIS scale factor
                    block_arrays.append(data)

                # Stack the small blocks (num_files, block_height, block_width)
                stack = np.stack(block_arrays, axis=0)

                # Compute median for this block ignoring NaNs
                block_median = np.nanmedian(stack, axis=0)

                # Write results to the output file window
                dst.write(block_median.astype(np.float32), 1, window=window)

                # Explicitly clear block memory
                del stack, block_arrays
                gc.collect()

    finally:
        # Close all source files
        for src in srcs:
            src.close()

    print(f"✅ Saved median composite → {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
