# 🦘 Australian Wildlife Roadkill Risk Mapper

> A national-scale geospatial ML platform that scores **every road segment in Australia** for wildlife-vehicle collision risk — and recommends exactly where to place warning signs.

Built on 413,000+ biodiversity occurrence records across 11 native species, the platform combines ecological presence data, road network attributes, NDVI vegetation rasters, and a spatially-lagged proxy label to train an XGBoost risk regression model with full SHAP explainability. Output is served through a live Streamlit application.

The primary objective is to transform static, historically unreliable sign placement strategies into a dynamically updated, statistically rigorous decision-support tool. By modeling not just where animals have been seen, but the *ecological and infrastructural conditions* that promote a collision, the system identifies proactive and evidence-backed intervention zones.

---

## Why This Matters

Wildlife-vehicle collisions kill more than **[10 million animals](https://findanexpert.unimelb.edu.au/news/79342-10-million-animals-die-on-our-roads-each-year.-here%E2%80%99s-what-works-(and-what-doesn%E2%80%99t)-to-cut-the-toll)** annually in Australia alone and pose serious road safety risks. Current signage placement is largely static and not evidence-backed. This project provides road authorities with a **data-driven, reproducible pipeline** — from raw occurrence records to a deployable GeoJSON of prioritised sign locations — using 100% open data and open-source tooling.

---

## Architecture

```
ALA / GBIF occurrences                    → 413k sightings, 11 species
OSM road network (GeoFabrik)              → segment-level road attributes
State Map (ABS)                           → state boundaries
NDVI rasters (NASA MODIS MOD13A3)         → vegetation context
          │
          ▼
  Feature Store (sightings.parquet)
  (species · lat/lon · season · NDVI · road_class · speed · traffic · distance_to_road etc.)
          │
          ▼
  Proxy Label Construction
  ecological_score × road_exposure_score
  + spatial lag blending (PySAL)          → neighbourhood context
  + rank normalisation                    → proxy_risk ∈ [0, 1]
          │
          ▼
  XGBoost Regressor
  + Stochastic Spatial Block CV (5-fold)  → no autocorrelation leakage
  + Optuna hyperparameter tuning          → 50 trials
  + SHAP TreeExplainer                    → per-segment feature attribution
  + Moran's I on residuals                → spatial leakage audit
          │
          ▼
  Sign Placement Engine
  Sliding 5km window · top-K per state · 2km deduplication
          │
          ▼
  Streamlit App (Community Cloud)
  Folium map · risk heatmap · high-risk segments · sign placements · SHAP panel
```

---

## Current Status

| Phase | Description | Key Details | Status |
|---|---|---|---|
| 1 | Data ingestion (ALA + GBIF) | Fetching wildlife occurrences via REST APIs using **Requests** and **HTTPX (async)** | ✅ Done |
| 2 | Data cleaning + deduplication | Removing null/invalid coordinates and filtering spatial outliers (Pandas) | ✅ Done |
| 3 | Spatial join to road network + state boundaries | Applying accurate nearest-neighbour coordinate mapping to OSM road geometries | ✅ Done |
| 4 | Feature engineering (NDVI · season · ecological weights) | Producing memory-efficient raster composites and deriving biological proxies | ✅ Done |
| 5 | Proxy label construction + model training (XGBoost + SHAP) | Generating the spatially-lagged target variable and optimizing boosting hyperparameters | 🔄 In Progress |
| 6 | Sign placement engine | Operating a sliding-window heuristic to pinpoint maximum-impact sign locations | ⏳ Pending |
| 7 | Streamlit + Folium application | Creating an interactive and fully explainable visualization interface for stakeholders | ⏳ Pending |
| 8 | Model card + METHODOLOGY.md + screen recording | Detailing data provenance, analytical rationale, and final pipeline outputs | ⏳ Pending |

**Pipeline output:** `sightings.parquet` — 413,000 rows · 11 species · 17 features · ready for label construction.

---

## Tracked Species

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

---

## Feature Schema

| Column | Description |
|---|---|
| `species` | Scientific name |
| `month` / `year` | Temporal context |
| `latitude` / `longitude` | WGS84 coordinates |
| `season` | Meteorological season |
| `body_mass_weight` | Species body-mass collision severity proxy |
| `nocturnal_weight` | Nocturnal activity risk multiplier |
| `peak_season_weight` | Breeding / peak-movement period weight |
| `ndvi` | Median NDVI at sighting location (MODIS) |
| `road_segment_id` | Nearest OSM road segment |
| `road_class` | Road type (motorway → track) |
| `speed_limit` | Speed zone (km/h) |
| `traffic_proxy` | Relative traffic volume (1–5) |
| `distance_to_road` | Distance to nearest road (metres) |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | Pandas, PyArrow (Parquet) |
| Spatial | GeoPandas, Shapely, Rasterio, PyProj |
| Road / Raster data | GeoFabrik OSM, NASA AppEEARS MODIS |
| Spatial statistics | PySAL (libpysal, esda) |
| ML | XGBoost, scikit-learn, Optuna, SHAP |
| Experiment tracking | MLflow (local) |
| App | Streamlit, Folium |
| HTTP | HTTPX (async), Requests |

---

## Setup

```bash
git clone https://github.com/sadmanhsakib/aus-wildlife-roadkill-risk-mapper.git
cd aus-wildlife-roadkill-risk-mapper
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Required data** (place in `data/raw/`):
- ABS state boundary shapefiles — [ABS ASGS](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files)
- `australia.gpkg` — [GeoFabrik](https://download.geofabrik.de/australia-oceania/australia.html)
- Monthly NDVI GeoTIFFs — [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/) into `data/raw/vegetation/`

```bash
python scripts/fetcher.py     # Ingestion, cleaning, road/NDVI preprocessing
python scripts/analyzer.py    # Spatial enrichment, feature engineering
```

---

## Project Structure

```
├── scripts/
│   ├── fetcher.py              # ALA/GBIF ingestion · cleaning · road/NDVI prep
│   └── analyzer.py             # Spatial joins · NDVI sampling · feature output
├── data/
│   ├── raw/                    # Shapefiles, GeoPackage, NDVI rasters (gitignored)
│   ├── processed/              # Intermediate and output parquet/tif files (gitignored)
├── sightings.parquet           # Final merged feature store (gitignored)
├── requirements.txt
└── LICENSE
```

---

## License

MIT — see [LICENSE](LICENSE).