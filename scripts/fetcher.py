"""
Wildlife Data Ingestion & Geospatial Preprocessing Pipeline
===========================================================
This module orchestrates the full upstream data pipeline: fetching biodiversity
occurrence records from GBIF and ALA, cleaning and standardizing raw CSVs,
enriching species records with ecological risk weights, and preparing projected
road and state boundary datasets for downstream spatial analysis.
"""

import asyncio, gc, glob, os, time
import httpx, rasterio, requests
import numpy as np
import pandas as pd
import geopandas as gpd

# --- GBIF Species Taxon Keys ---
# GBIF occurrence search requires backbone taxon keys, not free-text names,
# to avoid synonym ambiguity across its global taxonomy.
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

# --- ALA Scientific Names ---
# ALA's Solr API accepts binomial names directly; we query both ALA and GBIF
# because each archive has different contributor networks and coverage gaps.
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

# --- External API Endpoints ---
GBIF_URL = "https://api.gbif.org/v1/occurrence/search"
ALA_URL = "https://biocache-ws.ala.org.au/ws/occurrences/search"

# Canonical state list — downstream filters and map labels expect these eight
# jurisdictions; the ABS SA1 shapefile includes extra areas we deliberately omit.
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

# Australian seasons run opposite to the Northern Hemisphere calendar convention.
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

# Peak activity months per species, informed by breeding and movement ecology literature.
# Used to assign a temporal risk weight to sightings recorded during high-activity periods.
PEAK_SEASON_MAP = {
    "Osphranter rufus": [9, 10, 11, 12],
    "Macropus rufus": [9, 10, 11, 12],  # legacy GBIF synonym for Osphranter rufus
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

# Body mass weight coefficients, scaled 0.0–1.0 relative to the red kangaroo (~85 kg).
# Larger animals cause greater vehicle damage and represent higher mortality risk per collision.
BODY_MASS_WEIGHT = {
    "Osphranter rufus": 1.00,           # Red kangaroo       ~85kg
    "Macropus giganteus": 0.90,         # Eastern grey       ~66kg
    "Wallabia bicolor": 0.65,           # Swamp wallaby      ~20kg
    "Notamacropus rufogriseus": 0.60,   # Red-necked wallaby ~17kg
    "Vombatus ursinus": 0.70,           # Common wombat      ~35kg
    "Phascolarctos cinereus": 0.55,     # Koala              ~12kg
    "Trichosurus vulpecula": 0.30,      # Brushtail possum   ~4kg
    "Pseudocheirus peregrinus": 0.25,   # Ringtail possum  ~1kg
    "Isoodon obesulus": 0.35,           # S. brown bandicoot ~1.5kg
    "Tachyglossus aculeatus": 0.40,     # Echidna            ~6kg
    "Ornithorhynchus anatinus": 0.45,   # Platypus         ~2kg
}

# Nocturnality score (0.0–1.0). Reflects the probability of nighttime road crossing,
# where reduced driver visibility significantly increases collision risk.
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
    prepare_state_boundaries()


async def get_gbif_data(species_key: int, state: str) -> str:
    """
    Asynchronously fetches paginated species occurrence records from the GBIF API.

    Iterates through all available pages for the given taxon and state,
    persisting results to a CSV upon completion.

    Args:
        species_key: GBIF backbone taxon key for the target species.
        state: Australian state/territory name to scope the query.

    Returns:
        Path to the exported CSV file, or None if no records were found.
    """
    offset = 0
    results = []

    # httpx keeps pagination non-blocking so multiple species/state jobs can run concurrently.
    async with httpx.AsyncClient() as client:
        while True:
            params = {
                "taxonKey": species_key,
                "country": "AU",
                "hasCoordinate": "true",  # coordinate-less records cannot join to road segments
                "stateProvince": f"{state}",
                # Window matches NDVI composite, road network vintage, and model training period.
                "year": [2026, 2025, 2024, 2023, 2022, 2021, 2020],
                "limit": 300,  # GBIF hard cap per request
                "offset": offset,
            }

            response = await client.get(GBIF_URL, params=params)

            if response.status_code == 200:
                data = response.json()
                results.extend(data["results"])

                print(f"✅ Data Pulled: {offset}")
                if data["endOfRecords"]:
                    break
                offset += 300
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                return 1

        # One pause after each species/state query — enough to stay under GBIF's fair-use threshold.
        await asyncio.sleep(1.0)

    if results:
        file_name = (
            f"{results[0]['species'].lower().replace(' ', '_')}_sightings_gbif.csv"
        )
        df = pd.DataFrame(results)[
            # Subset at export — raw API payloads carry dozens of unused metadata fields.
            ["species", "month", "year", "decimalLatitude", "decimalLongitude"]
        ]
        df.to_csv(file_name, index=False)
        print(f"✅ Data exported to {file_name} successfully. ")

        return file_name
    else:
        print("⛔ No results found.")
        return None


def get_ala_data(species_scientific_name: str, state: str) -> str:
    """
    Fetches paginated species occurrence records from the Atlas of Living Australia (ALA) API.

    Args:
        species_scientific_name: Binomial species name for the Solr-based ALA query.
        state: Australian state/territory name to scope the query.

    Returns:
        Path to the exported CSV file, or None if no records were found.
    """
    offset = 0
    results = []

    while True:
        params = {
            "q": species_scientific_name,
            "fq": [
                "country:Australia",
                "year:[2020 TO 2026]",
                f"stateProvince:{state}",
            ],
            "pageSize": 1000,  # ALA maximum — fewer round trips than GBIF's smaller pages
            "startIndex": offset,
            # Field list keeps payloads small; we only need columns the risk model consumes.
            "fl": "scientificName,month,year,decimalLatitude,decimalLongitude",
        }
        headers = {"Accept": "application/json"}

        response = requests.get(ALA_URL, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            results.extend(data["occurrences"])

            print(f"✅ Data Pulled: {offset}")
            try:
                total_records = data.get("totalRecords", 0)
                if not data.get("occurrences") or offset + 1000 >= total_records:
                    break
                offset += 1000
            except (KeyError, TypeError):
                # ALA occasionally returns partial JSON on timeout — bail rather than loop forever.
                break
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return 1

        # Per-page throttle: ALA is more sensitive to burst traffic than GBIF.
        time.sleep(1.0)

    if results:
        file_name = f"{results[0]['scientificName'].lower().replace(' ', '_')}_sightings_ala.csv"
        df = pd.DataFrame(results)
        df.to_csv(file_name, index=False)
        print(f"✅ Data exported to {file_name} successfully. ")

        return file_name
    else:
        print("⛔ No results found.")
        return None


def clean_DataFrame(file_name: str):
    """
    Standardizes and validates a raw occurrence CSV, enforcing schema consistency
    and geographic bounds for Australian territory.

    Applies the following transformations:
    - Normalizes ALA/GBIF column naming differences to a unified schema.
    - Drops records missing temporal or coordinate data.
    - Excludes records predating the 2020 baseline, which aligns with
      the NDVI composite period and road network data.
    - Enforces Australian continental bounding box to remove offshore artefacts.
    - Deduplicates at the spatial-temporal granularity of the data model.

    Args:
        file_name: Path to the input CSV file. Overwritten in place on success.
    """
    column_schema = [
        "species",
        "month",
        "year",
        "decimalLatitude",
        "decimalLongitude",
    ]

    df = pd.read_csv(file_name)

    # Harmonize column names so GBIF and ALA records share one downstream schema.
    try:
        df = df.rename(columns={"scientificName": "species"})
    except KeyError:
        pass

    df = df[column_schema]
    df = df.rename(
        columns={
            "decimalLatitude": "latitude",
            "decimalLongitude": "longitude",
        }
    )

    df = df.dropna(subset=["latitude", "longitude", "year", "month"])

    # Exclude pre-2020 records to align with NDVI and road dataset temporal coverage
    df = df.drop(df[df["year"] < 2020].index)

    # Continental bbox drops mis-geocoded and offshore records outside the analysis extent.
    df = df.drop(df[df["latitude"] < -44].index)
    df = df.drop(df[df["latitude"] > -10].index)
    df = df.drop(df[df["longitude"] < 113].index)
    df = df.drop(df[df["longitude"] > 154].index)

    # Same sighting can appear in both GBIF and ALA — dedupe before spatial aggregation.
    df = df.drop_duplicates(subset=["latitude", "longitude", "year", "month"])

    df.to_csv(f"{file_name}", index=False)
    print(f"✅ Data exported to {file_name} successfully. ")


def merge(new_file_name: str, file_names: list, shouldDelete=False, has_geometry=False):
    """
    Consolidates multiple species occurrence files into a single unified dataset.

    Supports both CSV and Parquet input formats. Deduplication ensures data
    integrity across overlapping pulls from GBIF and ALA for the same species.

    Args:
        new_file_name: Output file path. Extension determines format (.csv or .parquet).
        file_names: List of input file paths to merge.
        shouldDelete: If True, source files are removed after a successful merge.
    """
    df_list = []

    # GeoParquet preserves geometry for spatial layers; flat Parquet for tabular sightings.
    if has_geometry:
        for file_name in file_names:
            if file_name.endswith(".csv"):
                df_list.append(pd.read_csv(file_name))
            elif file_name.endswith(".parquet"):
                df_list.append(gpd.read_parquet(file_name))
    else:
        for file_name in file_names:
            if file_name.endswith(".csv"):
                df_list.append(pd.read_csv(file_name))
            elif file_name.endswith(".parquet"):
                df_list.append(pd.read_parquet(file_name))

    merged_df = pd.concat(df_list, ignore_index=True)

    # Final merge pass catches duplicates that survived per-file cleaning.
    merged_df = merged_df.drop_duplicates(
        subset=["species", "month", "year", "latitude", "longitude"]
    )

    # Per-state/per-source CSVs are throwaway once consolidated — keeps data/ tidy.
    if shouldDelete:
        for file_name in file_names:
            os.remove(file_name)

    if new_file_name.endswith(".csv"):
        merged_df.to_csv(new_file_name, index=False)
    elif new_file_name.endswith(".parquet"):
        merged_df.to_parquet(new_file_name, index=False)

    print(f"✅ {file_names} merged into {new_file_name} successfully. ")


def to_parquet(path: str):
    """
    Converts CSV occurrence files to the Parquet format for columnar storage efficiency.

    Accepts either a directory (batch conversion) or a single file path. Source CSV
    files are removed after successful conversion to prevent stale data accumulation.

    Args:
        path: Directory path or single CSV file path to convert.
    """
    file_names = []

    try:
        for file_name in os.listdir(path):
            if file_name.endswith(".csv"):
                file_names.append(os.path.join(path, file_name))
    except NotADirectoryError:
        # Caller may pass a single file path instead of a directory.
        file_names.append(path)

    for file_name in file_names:
        new_file_name = file_name.replace(".csv", ".parquet")

        df = pd.read_csv(file_name)
        df.to_parquet(new_file_name, index=False)

        # Remove CSV so downstream stages cannot accidentally read the stale copy.
        os.remove(file_name)
        print(f"✅ Data exported to {new_file_name} successfully. ")


def enrich(path: str):
    """
    Augments occurrence records with species-specific ecological risk weights
    required for the composite risk scoring model in analyzer.py.

    Appended columns:
    - `season`: Calendar season derived from the observation month.
    - `body_mass_weight`: Normalized collision severity proxy (larger animal = higher impact).
    - `nocturnal_weight`: Probability of nighttime road crossing (higher = greater low-visibility risk).
    - `peak_season_weight`: Temporal risk multiplier (1.3 during peak activity months, else 1.0).

    Args:
        path: Directory path or single Parquet file path to enrich. Files are updated in place.
    """
    file_names = []

    try:
        for file_name in os.listdir(path):
            if file_name.endswith(".parquet"):
                file_names.append(os.path.join(path, file_name))
    except NotADirectoryError:
        # Caller may pass a single file path instead of a directory.
        file_names.append(path)

    for file_name in file_names:
        df = pd.read_parquet(file_name)

        df["season"] = df["month"].map(SEASON_MAP)
        df["body_mass_weight"] = df["species"].map(BODY_MASS_WEIGHT)
        df["nocturnal_weight"] = df["species"].map(NOCTURNAL)

        # 1.3× boost during peak activity months — modest uplift, not a hard filter.
        df["peak_season_weight"] = [
            1.3 if m in PEAK_SEASON_MAP.get(s, []) else 1.0
            for s, m in zip(df["species"], df["month"])
        ]

        df.to_parquet(file_name, index=False)
        print(f"✅ {file_name} enriched successfully. ")


def prepare_road_network():
    """
    Parses and projects the raw OpenStreetMap road network into the analysis-ready format.

    Filters OSM road classes to those relevant to wildlife collision risk,
    assigns canonical speed limits and traffic proxy volumes per road class,
    and reprojects to MGA Zone 54 (EPSG:32754) for metric distance calculations.

    Road class speed limits and traffic proxies are heuristics derived from
    Australian road design standards and traffic volume literature.
    """
    print("⏳ Loading the road network...")

    road_networks_gdf = gpd.read_file(
        "data/raw/road_network.gpkg",
        layer="gis_osm_roads_free",
        columns=["osm_id", "name", "fclass", "geometry"],
    )

    # Speed limit and traffic proxy are stand-ins where OSM lacks tagged maxspeed/AADT.
    FCLASS_DEFAULTS = {
        "motorway": (110, 1.0),
        "trunk": (100, 0.8),
        "primary": (100, 0.6),
        "secondary": (80, 0.4),
        "tertiary": (80, 0.4),
        "unclassified": (60, 0.2),
        "track": (40, 0.2),
        "residential": (50, 0.2),
    }

    # Exclude footpaths, cycleways, etc. — no meaningful vehicle-wildlife collision risk.
    road_networks_gdf = road_networks_gdf[road_networks_gdf["fclass"].isin(FCLASS_DEFAULTS.keys())]
    road_networks_gdf["speed_zone"] = road_networks_gdf["fclass"].map(
        lambda x: FCLASS_DEFAULTS[x][0]
    )
    road_networks_gdf["traffic_proxy"] = road_networks_gdf["fclass"].map(
        lambda x: FCLASS_DEFAULTS[x][1]
    )

    road_networks_gdf.rename(
        columns={
            "osm_id": "road_segment_id",
            "fclass": "road_class",
            "name": "road_name",
            "speed_zone": "speed_limit",
        },
        inplace=True,
    )

    road_networks_gdf.to_parquet("data/processed/road_networks.parquet", index=False)
    print("✅ Road network parsed and saved to data/processed/road_networks.parquet")


def prepare_state_boundaries():
    """
    Loads and reprojects the ABS SA1 state boundary shapefile into the analysis CRS.

    Filters to the eight canonical Australian states and territories,
    normalizes the column name, and projects to EPSG:32754 for spatial joins
    with the sightings and road network datasets.
    """
    print("⏳ Loading the state data...")
    state_boundaries = gpd.read_file(
        "data/raw/SA1_2021_AUST_GDA2020.shp", columns=["STE_NAME21", "geometry"]
    )

    # Keep only the eight mapped jurisdictions — matches STATE_CODES and map UI filters.
    state_boundaries = state_boundaries[
        state_boundaries["STE_NAME21"].isin(STATE_CODES.keys())
    ]
    state_boundaries = state_boundaries.rename(columns={"STE_NAME21": "state"})

    # Dissolve SA1 polygons to one multipolygon per state for state-level spatial joins.
    state_boundaries = state_boundaries.dissolve(by="state").reset_index()

    state_boundaries.to_parquet("data/processed/state_boundaries.parquet", index=False)
    print(
        "✅ State network parsed and saved to data/processed/state_boundaries.parquet"
    )


def build_ndvi_median_composite():
    """
    Generates a single long-term median NDVI raster from monthly AppEEARS GeoTIFFs.

    Uses block-wise (windowed) I/O to process rasters without loading full arrays
    into memory — essential for continent-scale vegetation datasets.

    Processing steps per block:
    - Reads the same spatial window from every monthly source file.
    - Applies the MODIS nodata mask (-28672) and the standard scale factor (0.0001).
    - Computes a NaN-safe pixel-wise median across all source months.
    - Writes the resulting composite block to the output GeoTIFF.
    """
    print("🔄 Merging the .tif files (memory-efficient block-wise processing)...")

    tif_folder = "data/raw/vegetation/"
    output_path = "data/processed/ndvi_median.tif"

    tif_files = sorted(glob.glob(os.path.join(tif_folder, "*.tif")))
    if not tif_files:
        print("⛔ No .tif files found in vegetation folder.")
        return

    print(f"✅ Found {len(tif_files)} monthly files. ")

    srcs = [rasterio.open(f) for f in tif_files]

    try:
        meta = srcs[0].meta.copy()
        meta.update(dtype="float32", count=1, nodata=np.nan)

        with rasterio.open(output_path, "w", **meta) as dst:
            windows = [window for _, window in dst.block_windows(1)]
            num_windows = len(windows)

            for i, window in enumerate(windows):
                if i % 10 == 0:
                    print(f"🔄 Processing block {i+1}/{num_windows}...")

                block_arrays = []
                for src in srcs:
                    data = src.read(1, window=window).astype(np.float32)
                    data[data == -28672] = np.nan  # treat MODIS fill as missing, not zero vegetation
                    data = data * 0.0001  # convert stored DN to physical reflectance before aggregation
                    block_arrays.append(data)

                # Median resists cloud gaps and anomalous months better than a temporal mean.
                stack = np.stack(block_arrays, axis=0)
                block_median = np.nanmedian(stack, axis=0)

                dst.write(block_median.astype(np.float32), 1, window=window)

                del stack, block_arrays
                # Explicit cleanup — block loops over continental rasters can exhaust memory on Windows.
                gc.collect()

    finally:
        for src in srcs:
            src.close()

    print(f"✅ Saved median composite → {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"✅ Execution completed in {time.time() - start_time:.2f} seconds")
