import os, time, gc
import httpx, asyncio
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


def main():
    asyncio.run(get_ala_data(WOMBAT_SCIENTIFIC_NAME, "New South Wales"))


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
                "fl": "scientificName,month,year,stateProvince,decimalLatitude,decimalLongitude",  # fields to return
            }
            headers = {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            # sending the requests
            response = await client.get(ALA_URL, params=params, headers=headers, follow_redirects=True)

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


def clean_data(file_name: str):
    column_schema = [
        "species",
        "month",
        "year",
        "stateProvince",
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
            "stateProvince": "state",
            "decimalLatitude": "latitude",
            "decimalLongitude": "longitude",
        }
    )

    # adding the state codes
    df["state"] = df["state"].map(STATE_CODES)

    # removing rows with missing values
    df = df.dropna(subset=["latitude", "longitude", "year", "month", "state"])

    # removing rows with older sighting data
    df = df.drop(df[df["year"] < 2020].index)

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
    for file_name in os.listdir(path):
        if not file_name.endswith(".csv"):
            continue

        file_name = os.path.join(path, file_name)
        new_file_name = file_name.replace(".csv", ".parquet")

        df = pd.read_csv(file_name)
        df.to_parquet(new_file_name, index=False)

        os.remove(file_name)
        print(f"✅Data exported to {new_file_name} successfully. ")


def enrich(path: str):
    for file_name in os.listdir(path):
        if not file_name.endswith(".parquet"):
            continue

        file_name = os.path.join(path, file_name)
        # loading the df
        df = pd.read_parquet(file_name)

        # adding the season column
        df["season"] = df["month"].map(SEASON_MAP)
        # adding the peak_season column
        df["is_peak_season"] = df.apply(
            lambda x: 1 if x["month"] in PEAK_SEASON_MAP[x["species"]] else 0, axis=1
        )

        df.to_parquet(file_name, index=False)


def prepare_road_network():
    print("Loading the road network...")

    # loading the roads data
    road_network = gpd.read_file(
        "road_data/australia.gpkg",
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
    # converting it to projected system for accurate distance calculations
    road_network_projected = road_network.to_crs("EPSG:32754")
    # freeing up memory space
    del road_network
    gc.collect()

    road_network_projected.to_parquet(
        "road_data/australia_projected.parquet", index=False
    )
    print("✅Road network parsed and saved to australia_projected.parquet")

    # adding buffer of 500m around the roads
    print("Adding buffer of 500m around the roads....")
    roads_with_buffer = gpd.GeoDataFrame(
        geometry=road_network_projected.buffer(500), crs="EPSG:32754"
    ).reset_index(drop=True)

    roads_with_buffer.to_parquet(
        "road_data/australia_projected_buffer.parquet", index=False
    )
    print("✅Road network parsed and saved to australia_projected_buffer.parquet")


def prepare_state_network():
    print("Loading the state data...")

    # loading the whole map
    states = gpd.read_file(
        "road_data/SA1_2021_AUST_GDA2020.shp", columns=["STE_NAME21", "geometry"]
    )

    # filtering to just the main states
    states = states[states["STE_NAME21"].isin(STATE_CODES.keys())]
    states_projected = states.to_crs("EPSG:32754")
    # freeing up memory space
    del states
    gc.collect()

    states_projected.to_parquet("road_data/states_projected.parquet", index=False)
    print("✅State network parsed and saved to states_projected.parquet")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
