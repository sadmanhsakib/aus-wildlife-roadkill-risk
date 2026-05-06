# 🦘 Australian Wildlife Roadkill Risk Mapper

## Overview

A Python-based geospatial data pipeline for analysing **wildlife-vehicle collision risk** across Australia. The project ingests wildlife sighting records from major biodiversity databases, cleans and standardises the data, overlays it with real road network geometry, and identifies high-risk zones where animals are most likely to encounter traffic.

Currently tracking **10 native species**:

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

> ⚠️ **This project is actively under development.** Core data ingestion and spatial analysis are functional, but several major features are still in progress or pending. Contributions and feedback are welcome!

## Roadmap

| # | Phase | Status |
|---|-------|--------|
| 1 | **Data Ingestion** (ALA + GBIF) | ✅ Done |
| 2 | **Data Cleaning + GeoPandas** | ✅ Done |
| 3 | **Spatial Risk Analysis** (GeoPandas + Shapely) | ✅ Done |
| 4 | **Map Visualization** (Folium) | ⏳ Pending |
| 5 | **Dashboard UI** (Streamlit) | ⏳ Pending |
| 6 | **Output** (Risk zones + signage recommendations) | ⏳ Pending |

## How It Works

### 1. Data Ingestion ✅

Pulls wildlife occurrence records from two sources:

- **[GBIF](https://www.gbif.org/)** — Global Biodiversity Information Facility (paginated REST API, up to 10 000+ records per species)
- **[ALA](https://www.ala.org.au/)** — Atlas of Living Australia (paginated REST API with field selection)

Both pipelines handle pagination, rate-limiting, and export raw results to CSV before processing.

### 2. Data Cleaning ✅

- Standardises column schemas across GBIF and ALA formats
- Filters to Australian records only
- Drops rows with missing coordinates, year, month, or state
- Removes sightings older than 2020
- Deduplicates on location + time
- Merges multiple species/source CSVs into a single `sightings/sightings.csv`

### 3. Spatial Risk Analysis ✅

Uses GeoPandas + Shapely to:

- Convert sightings to a projected CRS (`EPSG:32754`)
- Load Australian state boundaries (SA1 2021 shapefiles)
- Load road network geometry from an OpenStreetMap GeoPackage (`australia.gpkg`)
- Filter to relevant road types (motorway, trunk, primary, secondary, tertiary, residential, unclassified)
- Buffer roads by **500 m** to define risk corridors
- Spatial-join sightings to nearest road (`sjoin_nearest`) and classify as **high-risk** (within buffer) or **low-risk**

### 4. Visualisation (current: Matplotlib + Seaborn)

Plots the full analysis on a single figure:

- Australian state boundaries (green fill)
- Road network (black lines)
- 500 m road buffer zones (grey fill)
- Sightings colour-coded: 🔴 High risk / 🔵 Low risk

> **Coming soon:** Migration from static Matplotlib plots to interactive **Folium** maps.

### 5. Dashboard & Outputs — ⏳ Planned

- **Streamlit dashboard** for interactive exploration of risk zones
- Exportable **risk zone maps** and **signage placement recommendations**

## Project Structure

```
├── fetcher.py          # Data ingestion from GBIF & ALA APIs + cleaning + merging
├── analyzer.py         # Spatial risk analysis & visualisation
├── sightings/
│   └── sightings.csv   # Merged & cleaned sighting dataset (gitignored)
├── data/               # Geospatial data — shapefiles & GeoPackage (gitignored)
│   ├── SA1_2021_AUST_GDA2020.*   # Australian SA1 boundary shapefiles
│   └── australia.gpkg             # OSM road network GeoPackage
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
├── project_blueprint.md # Full technical blueprint & architecture plan
├── test.py             # Scratch/testing utilities (gitignored)
└── LICENSE             # MIT License
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | Pandas |
| Spatial Analysis | GeoPandas, Shapely |
| Road Data | OpenStreetMap (via GeoPackage) |
| Visualisation | Matplotlib, Seaborn *(Folium planned)* |
| Dashboard (upcoming) | Streamlit |
| API Calls | Requests |

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

3. **Download geospatial data** and place in the `data/` directory:
   - Australian SA1 boundary shapefiles from **[Australian Bureau of Statistics (ABS)](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files)**
   - `australia.gpkg` from **[Geofabrik](https://download.geofabrik.de/australia-oceania/australia.html)** OpenStreetMap exports

4. **Run the pipeline**
   ```bash
   # Fetch & clean sighting data
   python fetcher.py

   # Run spatial risk analysis & visualisation
   python analyzer.py
   ```

## Future Goals

- 🗺️ **Interactive Folium maps** — Zoomable, layer-toggled web maps with popups
- 📊 **Streamlit dashboard** — Real-time exploration of risk zones by species, state, and time period
- 🪧 **Signage recommendations** — Auto-generate optimal locations for wildlife warning signs

## License

See [LICENSE](LICENSE) for details.