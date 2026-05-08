# 🦘 Australian Wildlife Roadkill Risk Mapper

## Overview

A Python-based geospatial data pipeline for analysing **wildlife-vehicle collision risk** across Australia. The project ingests wildlife sighting records from major biodiversity databases, cleans and standardises the data, enriches it with spatial and environmental features, and builds a training-ready dataset for a risk classification model.

Currently tracking **11 native species**:

| Common Name | Scientific Name |
|---|---|
| Red Kangaroo | *Osphranter rufus* |
| Eastern Grey Kangaroo | *Macropus giganteus* |
| Swamp Wallaby | *Wallabia bicolor* |
| Red-necked Wallaby | *Notamacropus rufogriseus* |
| Common Wombat | *Vombatus ursinus* |
| Koala | *Phascolarctos cinereus* |
| Common Brushtail Possum | *Trichosurus vulpecula* |
| Common Ringtail Possum | *Pseudocheirus peregrinus* |
| Southern Brown Bandicoot | *Isoodon obesulus* |
| Short-beaked Echidna | *Tachyglossus aculeatus* |
| Platypus | *Ornithorhynchus anatinus* |

> ⚠️ **This project is actively under development.** The full data pipeline (ingestion → cleaning → spatial analysis → feature engineering) is complete. Model training is the active next step. Contributions and feedback are welcome!

## Roadmap

| # | Phase | Status |
|---|-------|--------|
| 1 | **Data Ingestion** (ALA + GBIF) | ✅ Done |
| 2 | **Data Cleaning + GeoPandas** | ✅ Done |
| 3 | **Spatial Risk Analysis** (GeoPandas) | ✅ Done |
| 4 | **Feature Engineering** (NDVI + road features + seasonality) | ✅ Done |
| 5 | **Model Training** (Risk classification) | 🔄 In Progress |
| 6 | **Map Visualization** (Folium) | ⏳ Pending |
| 7 | **Dashboard UI** (Streamlit) | ⏳ Pending |
| 8 | **Output** (Risk zones + signage recommendations) | ⏳ Pending |

## How It Works

### 1. Data Ingestion ✅

Pulls wildlife occurrence records from two sources:

- **[GBIF](https://www.gbif.org/)** — Global Biodiversity Information Facility (paginated REST API, up to 10,000+ records per species)
- **[ALA](https://www.ala.org.au/)** — Atlas of Living Australia (paginated REST API with field selection)

Prepares the raw downloaded road network, state boundaries and vegetation data:

- **[GeoFabrik.de](https://download.geofabrik.de/australia-oceania/australia.html)** — Australian road network (OSM) -> `prepare_road_network()`
- **[Australian Bureau of Statistics (ABS)](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files)** — State boundaries -> `prepare_state_network()`
- **[NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/)** — Vegetation data (NDVI) -> `build_ndvi_median_composite()`

Both pipelines handle pagination, rate-limiting, and export raw results to per-species CSV files before processing.

### 2. Data Cleaning ✅

- Standardises column schemas across GBIF and ALA formats
- Filters to Australian records only (bounding-box check on lat/lon)
- Drops rows with missing coordinates, year, or month
- Removes sightings older than 2020
- Deduplicates on `(latitude, longitude, year, month)` per file
- Converts per-species CSVs to **Parquet** format and stores them in `backup/`
- Enriches the data with `season`, `body_mass_weight`, `nocturnal_weight`, `peak_season_weight`
- Final global deduplication on `(species, month, year, latitude, longitude)` when merging

### 3. Spatial Analysis ✅

Uses GeoPandas + Shapely to enrich each sighting with road and boundary context:

- Converts sightings to a projected CRS (`EPSG:32754`) for accurate metric calculations
- Loads pre-built Australian state boundaries (`states_projected.parquet`) and spatial-joins each sighting to assign its state
- State names are normalised to two-letter codes (e.g. `"New South Wales"` → `"NSW"`)
- Loads the OSM road network (`australia_projected.parquet`) — filtered to motorway, trunk, primary, secondary, tertiary, residential, unclassified, and track classes — and runs `sjoin_nearest` to attach:
  - `road_segment_id` — the nearest OSM road segment
  - `road_class` — road type (fclass)
  - `speed_limit` — mapped speed zone in km/h
  - `traffic_proxy` — relative traffic volume rating (1–5)
  - `distance_to_road` — Euclidean distance in metres (rounded to 2 dp)
- Loads a 500 m road buffer layer (`australia_projected_buffer.parquet`) and spatial-joins to classify each sighting as:
  - `risk_label = 1` → **High risk** (within 500 m of a road)
  - `risk_label = 0` → **Low risk** (further than 500 m)

### 4. Feature Engineering ✅

#### NDVI / Vegetation ✅

Raw monthly NDVI GeoTIFF files (from **NASA AppEEARS / MODIS MOD13A3**) are processed as follows:

- All monthly `.tif` files in `data/raw/vegetation/` are discovered and opened
- A **memory-efficient block-wise (windowed) median composite** is computed — reading only one raster tile at a time per block to avoid RAM exhaustion from loading 100+ large arrays simultaneously
- MODIS nodata values (`-28672`) are masked and the MODIS scale factor (`× 0.0001`) is applied
- The result is saved as `data/ndvi_median_australia.tif` — a single float32 raster representing long-run median NDVI across Australia
- Each sighting's coordinates are then used to **sample the median NDVI raster**, adding an `ndvi` column to the dataset

#### Season & Peak Season ✅

Two additional columns are derived per sighting:

| Column | Description |
|---|---|
| `season` | Meteorological season — `"Summer"`, `"Autumn"`, `"Winter"`, `"Spring"` |
| `is_peak_season` | `1` if the sighting month falls within the species' known breeding/peak-movement period, else `0` |

Peak seasons are defined per species in `PEAK_SEASON_MAP` (see `fetcher.py`).

### 5. Final Training Dataset Schema ✅

The processed output (per-species in `sightings/`, merged in `sightings.parquet`) has the following columns ready for model training:

| Column | Type | Description |
|---|---|---|
| `species` | str | Scientific name |
| `month` | int | Month of sighting (1–12) |
| `year` | int | Year of sighting |
| `latitude` | float | WGS84 decimal latitude |
| `longitude` | float | WGS84 decimal longitude |
| `season` | str | Meteorological season |
| `is_peak_season` | int | 1 = peak breeding/movement period |
| `state` | str | 2-letter state code |
| `ndvi` | float | Median NDVI at sighting location |
| `road_segment_id` | str | Nearest OSM road segment ID |
| `road_class` | str | Road type (motorway, primary, etc.) |
| `speed_limit` | int | Speed zone (km/h) |
| `traffic_proxy` | int | Traffic volume proxy (1–5) |
| `distance_to_road` | float | Distance to nearest road (metres) |
| `risk_label` | int | Target variable — 1 = high risk, 0 = low risk |
| `geometry` | geometry | Projected point geometry (EPSG:32754) |

### 6. Upcoming: Model Training 🔄

The completed dataset will be used to train a binary risk classifier. Planned approach:

- Feature selection and exploratory correlation analysis
- Train/test split (stratified by species and state)
- Baseline model (Logistic Regression / Random Forest)
- Evaluation via precision, recall, ROC-AUC
- Saved model artefact for use inside the Streamlit app

### 7. Visualisation (current: Matplotlib + Seaborn)

Plots the full spatial analysis on a single figure:

- Australian state boundaries (green fill)
- Road network (black lines)
- 500 m road buffer zones (grey fill)
- Sightings colour-coded: 🔴 High risk / 🔵 Low risk

> **Coming soon:** Migration from static Matplotlib plots to interactive **Folium** maps.

### 8. Dashboard & Outputs — ⏳ Planned

- **Streamlit dashboard** for interactive exploration of risk zones by species, state, and time period
- Exportable **risk zone maps** and **signage placement recommendations** using a greedy placement algorithm

## Project Structure

```
├── fetcher.py              # Data ingestion (GBIF & ALA APIs) + cleaning + merging
│                           #   + prepare_road_network(), prepare_state_network()
│                           #   + build_ndvi_median_composite() (memory-efficient NDVI composite)
│                           #   + enrich()  (season + is_peak_season columns)
├── analyzer.py             # Spatial risk analysis + NDVI sampling + visualisation
├── backup/                 # Per-species cleaned Parquet files (gitignored)
│   └── {species}.parquet
├── sightings/              # Per-species spatially-enriched Parquet files (gitignored)
│   └── {species}.parquet
├── sightings.parquet       # Final merged dataset — all species, all features
├── data/
│   ├── raw/
│   │   ├── SA1_2021_AUST_GDA2020.*    # ABS SA1 boundary shapefiles (gitignored)
│   │   ├── australia.gpkg             # OSM road network GeoPackage (gitignored)
│   │   └── vegetation/                # Monthly MODIS NDVI GeoTIFFs (gitignored)
│   ├── australia_projected.parquet    # Processed road network in EPSG:32754 (gitignored)
│   ├── australia_projected_buffer.parquet  # 500m road buffer (gitignored)
│   ├── states_projected.parquet       # State boundaries in EPSG:32754 (gitignored)
│   └── ndvi_median_australia.tif          # Median NDVI composite raster (gitignored)
├── blueprint.md            # Full technical blueprint & architecture plan
├── test.py                 # Scratch/testing utilities (gitignored)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT License
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | Pandas, PyArrow |
| Spatial Analysis | GeoPandas, Shapely |
| Raster Processing | Rasterio, NumPy |
| Road Data | OpenStreetMap (via Geofabrik GeoPackage) |
| Vegetation Data | NASA AppEEARS / MODIS MOD13A3 NDVI |
| Coordinate Systems | PyProj |
| HTTP / Async | HTTPX (async) |
| Visualisation | Matplotlib, Seaborn *(Folium planned)* |
| Dashboard (upcoming) | Streamlit |
| Model Training (upcoming) | scikit-learn |

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/sadmanhsakib/aus-wildlife-roadkill-risk-mapper.git
   cd aus-wildlife-roadkill-risk-mapper
   ```

2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   pip install -r requirements.txt
   ```

3. **Download geospatial data** and place in the `data/raw/` directory:
   - Australian SA1 boundary shapefiles from **[Australian Bureau of Statistics (ABS)](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files)**
   - `australia.gpkg` from **[Geofabrik](https://download.geofabrik.de/australia-oceania/australia.html)** OpenStreetMap exports
   - Monthly NDVI GeoTIFFs (MODIS MOD13A3) from **[NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/)** into `data/raw/vegetation/`

4. **Run the pipeline**
   ```bash
   # Step 1 — Pre-process road network, state boundaries, and NDVI raster
   # (only needed once; outputs saved as .parquet / .tif files)
   python fetcher.py

   # Step 2 — Run spatial risk analysis & feature engineering per species
   python analyzer.py
   ```

## Future Goals

- 🤖 **Risk classifier** — Train a binary model on the completed feature set to predict high/low risk zones
- 🗺️ **Interactive Folium maps** — Zoomable, layer-toggled web maps with popups
- 📊 **Streamlit dashboard** — Real-time exploration of risk zones by species, state, and time period
- 🪧 **Signage recommendations** — Auto-generate optimal locations for wildlife warning signs using a greedy placement algorithm

## License

See [LICENSE](LICENSE) for details.