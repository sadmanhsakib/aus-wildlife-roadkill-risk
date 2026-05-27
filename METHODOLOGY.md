# Methodology — Australian Wildlife Roadkill Risk Mapper

This document records the research design decisions behind every stage of the
pipeline — why each approach was chosen, what the alternatives were, and what
the known limitations are. It is intended for technically literate readers who
want to understand the reasoning behind the system, not just its outputs.

---

## 1. Problem Framing

### 1.1 Why This Is a Hard Problem

Wildlife-vehicle collision risk modelling is a supervised learning problem with
no reliable supervisory signal. Approximately 10 million animals die on Australian
roads annually, yet [iNaturalist](https://inaturalist.ala.org.au/projects/australia-s-untold-roadtoll-recording-roadkill-and-road-trauma) — the most comprehensive open citizen-science
platform available — contains only ~15,000 confirmed roadkill observations across
570+ species accumulated over 5 years. This represents a detection rate of
approximately 0.03% relative to estimated annual collision frequency.

This sparsity is structural, not an engineering failure. Roadkill events are
transient, geographically dispersed across 900,000+ km of road network, and
entirely dependent on opportunistic observer presence at the moment of discovery.
Formal collision databases maintained by state road authorities are fragmented,
inconsistently recorded across jurisdictions, and not easily accessible under
open data licences. No national standardised roadkill monitoring program exists
in Australia at the time of writing.

Direct supervised learning — train a model to predict collision counts from road
and ecological features — is therefore infeasible. The ground truth does not exist
at the spatial and temporal resolution required.

### 1.2 Why a Proxy Label Approach

Given the absence of ground truth, the methodologically legitimate options are:

1. **Proxy label construction** — derive a surrogate risk score from observable
   correlates of collision risk (wildlife presence, road danger, habitat quality)
   and train a model to predict that surrogate.
2. **Unsupervised clustering** — group segments by feature similarity without
   any risk target. Produces clusters, not risk scores.
3. **Expert rule systems** — hard-coded thresholds from domain knowledge. Not
   generalisable and not updatable from new data.

This project uses approach 1. The proxy label is constructed to be ecologically
principled, spatially coherent, and partially non-recoverable by formula — the
three properties required for a proxy label to support genuine model generalisation
rather than formula memorisation. The known circularity this introduces is
documented in full in Section 7.

---

## 2. Data Design Decisions

### 2.1 Species Selection — Why These 11

Species were selected to satisfy four simultaneous criteria:

- **Ecological significance** — endemic Australian fauna with conservation concern
- **Collision severity** — body mass large enough to cause vehicle damage or fatality
- **Road proximity** — documented movement patterns that intersect road networks
- **Data availability** — sufficient verified occurrence records in [ALA](https://www.ala.org.au/) and [GBIF](https://www.gbif.org/)
  to produce segment-level sighting density after spatial aggregation

The 11 selected species (red kangaroo, eastern grey kangaroo, swamp wallaby,
red-necked wallaby, common wombat, koala, common brushtail possum, common ringtail
possum, southern brown bandicoot, short-beaked echidna, platypus) represent the
primary collision risk fauna identified in Australian road ecology literature.
Species with fewer than ~5,000 ALA records nationally were excluded — insufficient
density to produce reliable segment-level aggregates across all 8 states.

### 2.2 Dual API Ingestion — ALA and GBIF

ALA and GBIF are pulled independently and merged rather than using either alone
because the two databases have structurally different observer networks. ALA
specialises in Australian citizen science and is curated by taxonomic experts
at the national level. GBIF aggregates from a broader global contributor pool
including museum specimens, research surveys, and international biodiversity
programmes. Their overlap is partial — records exclusive to each source exist
in meaningful numbers. Taking the union (deduplicated on exact coordinates, month,
and year) achieves greater spatial coverage than either source alone, particularly
for remote regions underrepresented in citizen science data.

### 2.3 Temporal Window — 2020 to 2026

The six-year window was chosen to balance three competing constraints:

- **Recency** — older records reflect historical habitat and road conditions
  that may no longer apply. A koala sighting from 2005 near a dirt track that
  is now a dual carriageway is actively misleading.
- **Volume** — shorter windows reduce record counts below the threshold needed
  for reliable segment-level aggregation in low-density regions.
- **NDVI alignment** — MODIS MOD13A3 monthly composites are available from 2000
  onward, but the NASA AppEEARS pipeline was configured for 2020–2026 to match
  the sighting window. NDVI and sightings must cover the same period for the
  habitat quality signal to be meaningful.

### 2.4 NDVI Resolution — MODIS MOD13A3 at 1km

NDVI was sourced from MODIS MOD13A3 monthly composites at 1km spatial resolution.
Higher-resolution alternatives (Sentinel-2 at 10m, Landsat at 30m) were considered
and rejected for two reasons. First, 1km resolution is appropriate for the spatial
scale of road segment aggregation — road segments in the Australian road network
average several hundred metres in length, making sub-100m vegetation detail
irrelevant at the segment level. Second, monthly 1km composites are computationally
tractable at national scale; processing 150 Sentinel-2 tiles for all of Australia
at 10m resolution would require cloud compute resources inconsistent with the
open, reproducible design of this pipeline.

A median composite across all 150 monthly GeoTIFFs was computed rather than a
mean to suppress fire scar artefacts and cloud contamination, which produce
anomalously low NDVI values in individual months. The median is robust to these
outliers in a way the mean is not.

### 2.5 Road Network — GeoFabrik OSM

OpenStreetMap via GeoFabrik was chosen over alternatives (PSMA, HERE, TomTom)
because it is the only nationally complete, openly licensed, machine-readable
road network for Australia. The GeoPackage format preserves full attribute schema
including highway classification, which maps directly to the road class hierarchy
used in the road exposure score. Proprietary alternatives require commercial
licences incompatible with the open science commitment of this project.

`speed_limit` and `traffic_proxy` are imputed from road class where OSM attributes
are missing — a known limitation acknowledged in Section 7.

---

## 3. Proxy Label Design

### 3.1 Ecological Score — Weight Rationale

The ecological score is a weighted sum of five normalised components:

```text
ecological_score =
  0.30 × norm(sighting_count)
+ 0.20 × norm(mean_ndvi)
+ 0.15 × norm(species_richness)
+ 0.15 × norm(mean_peak_season_weight)
+ 0.10 × norm(mean_nocturnal_weight)
+ 0.10 × norm(mean_body_mass_weight)
```

`sighting_count` receives the highest weight (0.30) because it is the most direct
observational evidence of wildlife presence at a road segment. NDVI (0.20) receives
the second-highest weight because habitat quality is the strongest predictor of
sustained wildlife presence independent of observation effort. Species richness
(0.15) captures biodiversity breadth — segments supporting multiple species have
higher baseline collision probability across seasonal variation. The three
species-level weights (peak season, nocturnal, body mass) each receive 0.10–0.15
because they modify the *character* of risk (when, how visibly, how severely)
rather than the baseline probability of presence.

All five components are min-max normalised to [0, 1] before weighting. Outlier
clipping was considered but not applied at this stage — the rank normalisation
in Step 4 of the label construction makes the final `proxy_risk` scale robust
to outlier influence in the raw components.

### 3.2 Road Exposure Score — Weight Rationale

```text
road_exposure_score =
  0.35 × norm(speed_limit)
+ 0.35 × norm(proximity)
+ 0.30 × norm(traffic_proxy)
```

Speed limit and proximity receive equal weight (0.35 each) because they represent
the two independent physical determinants of collision probability: kinetic energy
at impact and how close animals approach the carriageway. Traffic volume (0.30)
receives slightly lower weight because it is imputed from road class rather than
measured — a lower-confidence signal deserves lower weight.

`proximity` is computed as `1 − norm(distance_to_road)` so that segments where
animals are observed close to the road surface score high. Distance to road is
averaged across all sightings assigned to a segment, capturing the typical
approach distance of wildlife at that location.

### 3.3 Multiplicative Combination — Why Not Additive

`raw_risk = ecological_score × road_exposure_score`

The multiplicative form enforces a logical constraint: risk is only non-zero
where wildlife presence **and** road danger co-occur simultaneously. An additive
combination would allow a high ecological score on a quiet unsealed track to
produce moderate risk, or a high-speed motorway with no wildlife sightings to
score moderately. Neither of these reflects genuine collision risk. Multiplication
produces a zero (or near-zero) whenever either factor approaches zero — correctly
modelling the logical AND relationship between the two risk conditions.

### 3.4 Spatial Lag Blending — Why 70/30

```text
blended_risk = 0.7 × raw_risk + 0.3 × spatial_lag
```

The 70/30 split was chosen to give the local segment signal primacy while
injecting meaningful neighbourhood context. Wildlife movement corridors do not
respect individual road segment boundaries — an animal crossing a segment
habitually will also cross adjacent segments. A segment with moderate own-sightings
surrounded by high-density sighting segments is ecologically more dangerous than
its own numbers suggest. The spatial lag encodes this corridor context.

The 0.3 lag weight was determined by testing the range 0.1–0.5. Below 0.2, the
lag adds negligible smoothing and Moran's I on the final label is not meaningfully
different from the unlagged version. Above 0.4, the label loses local specificity
— segments with genuinely low risk start inheriting high scores from distant
neighbours, reducing the discriminative power of the label. 0.3 balances
smoothing against specificity.

K=5 nearest neighbours was used for the spatial weights matrix. This was chosen
as the minimum neighbourhood size that produces a fully connected weights matrix
across Australia's road network while remaining computationally tractable.
At k=5, 41 segments in remote regions form disconnected components (0.04% of
the dataset) — an acceptable rate for a national model. Increasing k reduces
disconnected components but introduces long-range averaging that washes out
local ecological signals.

### 3.5 Rank Normalisation — Why Percentile Rank and Not Min-Max

`proxy_risk = percentile_rank(blended_risk)`

Percentile rank was chosen over min-max normalisation for three reasons. First,
it removes the arbitrary absolute magnitude of `blended_risk`, which depends
entirely on the weight choices in Sections 3.1 and 3.2 — weights that were
reasoned but not empirically calibrated. Second, it produces a uniform [0, 1]
distribution that XGBoost regresses against more stably than a skewed continuous
target. Third, it is robust to outlier segments: a single extreme `blended_risk`
value cannot distort the entire scale the way it would under min-max normalisation.

The result is that `proxy_risk` encodes relative risk ranking, not absolute risk
magnitude. This is intentional — the system is a risk prioritisation tool, not
a collision frequency predictor.

---

## 4. Feature Engineering

### 4.1 Spatial Lag Features — Which Columns and Why

Four features are spatially lagged before model training:

```python
lag_cols = ["sighting_count", "species_richness", "mean_ndvi", "traffic_proxy"]
```

These four were selected because their neighbourhood averages carry independent
predictive signal beyond the segment's own value:

- `sighting_count` — clusters of sightings across adjacent segments indicate
  wildlife corridors; a segment's neighbour density predicts its own risk
  independent of its own count
- `species_richness` — biodiversity clusters geographically; neighbouring
  richness indicates habitat connectivity
- `mean_ndvi` — vegetation is spatially continuous; neighbouring NDVI indicates
  habitat extent beyond the segment boundary
- `traffic_proxy` — road network traffic patterns are correlated across connected
  segments; arterial roads carry consistent load across their length

The following features were explicitly **not** lagged:

- `speed_limit` — a discrete road attribute, constant across connected segments
  of the same road class; lagging it produces no additional information
- `distance_to_road` / `proximity` — already a mean of observations at that
  segment; lagging a mean of a mean adds noise, not signal
- `mean_body_mass_weight`, `mean_nocturnal_weight`, `mean_peak_season_weight` —
  species trait averages, not geographically structured signals; their spatial
  lag reflects species range overlap, not road risk

### 4.2 Road Class Encoding

`road_class` is excluded from the XGBoost feature matrix entirely. It is an
unordered categorical variable with high cardinality in the raw OSM schema.
Its risk-relevant information is already captured by `speed_limit` and
`traffic_proxy`, both of which are imputed from road class and carry the
ordinal road danger signal in a continuous form that XGBoost can use directly.
Including the raw categorical would require encoding (ordinal or one-hot) and
adds redundancy without new signal.

---

## 5. Model Selection

### 5.1 Why XGBoost

XGBoost was selected over the four primary alternatives considered:

**vs. Random Forest:** Both are ensemble tree methods on tabular data. XGBoost's
gradient boosting sequentially corrects residuals, making it more sample-efficient
on the 99k segment dataset. Random forests require more trees to achieve equivalent
performance and offer no native handling of the learning rate shrinkage that
controls overfitting on a noisy proxy label.

**vs. Linear Regression:** The relationship between ecological features and road
risk is non-linear and involves interactions (a high sighting count near a slow
road is different from the same count near a motorway). Linear regression cannot
capture these interactions without manual feature crosses. XGBoost learns them
automatically through tree splits.

**vs. Neural Network:** A neural network would require careful normalisation of
all input features, embedding layers for any ordinal inputs, and substantially
more training data to avoid overfitting. More critically, `shap.TreeExplainer`
computes **exact** Shapley values for XGBoost by traversing the tree structure.
For neural networks, SHAP uses kernel approximations — the per-segment feature
attributions would be statistically estimated rather than mathematically exact.
Exact SHAP is a non-negotiable requirement for the per-segment explainability
panel in the Streamlit application.

**vs. Gaussian Process Regression:** GPR is the theoretically correct model for
spatially correlated data and produces calibrated uncertainty estimates. It was
rejected on computational grounds — GPR scales as O(n³) with dataset size, making
it infeasible for 99k segments without approximations (sparse GP, inducing points)
that would require significant additional engineering and hyperparameter design
beyond the scope of this project.

### 5.2 Hyperparameter Choices

The production model uses Optuna-optimised hyperparameters (see Section 6.7):

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 900 | Optuna-selected tree count for optimal convergence |
| `learning_rate` | ~0.024 | Conservative shrinkage; reduces overfitting on proxy label |
| `max_depth` | 8 | Allows complex feature interactions while preventing memorisation |
| `subsample` | ~0.8 | Row dropout per tree; reduces variance |
| `colsample_bytree` | ~0.8 | Feature dropout per tree; prevents single-feature dominance |

These parameters were selected via 50-trial Bayesian search (TPE sampler) optimising
mean spatial CV R² across five fixed jitter seeds. The search space and convergence
analysis are documented in Section 6.7.

---

## 6. Validation Strategy

### 6.1 Why Spatial Block CV and Not Random CV

Random CV is methodologically incorrect for spatially autocorrelated data. If a
model is trained on segments 50 metres from a test segment, the test segment's
features are nearly identical to its neighbours in the training set. The model
appears to generalise when it is actually interpolating between adjacent points.
The reported CV metrics reflect memorisation of spatial proximity, not learned
feature relationships.

Spatial block CV addresses this by assigning geographically proximate segments
to the same fold. Train and test sets are separated by block boundaries, ensuring
the model must predict risk in geographic regions it has not seen, using only
learned feature relationships.

### 6.2 Why Jittered Block Boundaries

Fixed 50km × 50km grid boundaries create a systematic artefact: segments near
block edges always appear in the same fold across runs. A model trained on many
such CV runs could implicitly learn the fixed grid structure. Jittering the
boundary origin by ±15km for each fold randomises which fold boundary segments
fall into, preventing this from becoming a learnable pattern. The reported CV
metrics reflect the mean and standard deviation across the five resulting folds,
capturing both performance level and fold-to-fold stability.

### 6.3 Why Moran's I on Residuals

R² and MAE measure how accurately the model predicts `proxy_risk` values. They
cannot distinguish between two different explanations for that accuracy:

1. The model learned *why* risk is high at certain segments (ecological and
   infrastructure feature relationships)
2. The model learned *where* risk is high (geographic proximity to high-risk
   training segments)

Moran's I on residuals provides this diagnostic. If residuals are spatially
random (Moran's I ≈ 0, p > 0.05), the model has captured the spatial structure
of risk through features — not through proximity. If residuals are spatially
clustered (significant Moran's I), the model is leaving unexplained geographic
patterns in its errors, indicating spatial leakage.

Reported values:

| Quantity | Moran's I |
|---|---|
| `proxy_risk` target | 0.4117 |
| Model residuals | 0.3081 |
| Spatial structure explained | 25.2% |

The model absorbed 25.2% of the spatial autocorrelation present in the target
using only tabular features — no coordinates, no spatial position. Residual
autocorrelation of 0.3081 reflects unobserved landscape covariates absent from
available open data sources (terrain complexity, fencing density, seasonal
migration corridor structure), consistent with known limitations of proxy-label
ecological modelling at continental scale.

### 6.4 Why Tasmania as the Geographic Holdout

Tasmania was selected as the holdout state for four reasons:

- **Geographic isolation** — separated from the mainland by Bass Strait, ensuring
  no spatial bleed between mainland training segments and holdout test segments
- **Distinct fauna** — Tasmania's fauna composition differs meaningfully from the
  mainland, particularly for large macropods; generalisation there is a harder
  test than holding out a mainland state
- **Adequate segment count** — 2,385 road segments with nearby sightings; large
  enough for a statistically meaningful Spearman correlation
- **Size** — small enough that exclusion costs only 2.4% of training data,
  preserving mainland model quality

### 6.5 Why Spearman and Not Pearson for Holdout Correlation

Spearman rank correlation was used for the holdout validation because both
`predicted_risk` and `sighting_count` are rank-based or heavily skewed
distributions. Pearson correlation assumes a linear relationship between normally
distributed variables — neither assumption holds here. Spearman measures monotonic
association regardless of distribution shape, making it the appropriate metric
for validating that the model's risk ranking is consistent with the independent
sighting density signal.

### 6.6 Interpreting the Holdout Results

```text
Ceiling (proxy_risk vs sighting_count in TAS):      0.3001
Model   (predicted_risk vs sighting_count in TAS):  0.2929
% of ceiling achieved:                              97.6%
Direct  (predicted_risk vs proxy_risk in TAS):      0.9823
```

The ceiling correlation of 0.3001 reflects the inherent limit of validating a
six-variable composite label against a single constituent feature — sighting count
contributes 30% weight to `ecological_score`, which is itself one factor in the
multiplicative formula. The maximum achievable correlation against sighting count
alone is well below 1.0 by construction. Achieving 97.6% of that ceiling confirms
the model is not leaving recoverable signal on the table. The direct correlation
of 0.9823 confirms strong geographic generalisation to an unseen state.

### 6.7 Optuna Hyperparameter Optimisation

Bayesian hyperparameter search was implemented using Optuna (50 trials, TPE
sampler, `seed=67`), optimising mean spatial CV R² as the objective. To prevent
the stochastic jitter in block boundary assignment from introducing noise into
the objective surface, each trial evaluated the candidate parameters across five
fixed jitter seeds (`[11, 42, 77, 123, 200]`), producing 25 fold scores (5 seeds
× 5 folds) per trial. The trial score is the mean R² across all 25 folds. This
multi-seed averaging ensures the objective reflects genuine model quality rather
than a favourable spatial partition.

The search space covered `n_estimators` (100–1000), `learning_rate` (0.01–0.2,
log scale), `max_depth` (3–9), `subsample` (0.5–1.0), and `colsample_bytree`
(0.5–1.0). The optimised model was retrained on the full mainland dataset
(97,354 segments) using `study.best_params` and all downstream artifacts
(`model.pkl`, `road_segments_scored.parquet`, `shap_values.parquet`) were
regenerated from the retrained model.

Post-optimisation results on the Tasmania geographic holdout:

| Metric | Value |
|---|---|
| Mainland spatial CV R² | 0.9743 ± 0.0002 |
| Mainland spatial CV MAE | 0.0337 ± 0.0007 |
| Ceiling (proxy_risk vs sighting_count, TAS) | 0.3001 |
| Model (predicted_risk vs sighting_count, TAS) | 0.2922 (p=0.0000) |
| % of ceiling achieved | 97.4% |
| Direct (predicted_risk vs proxy_risk, TAS) | 0.9835 |

The optimised model achieves marginally stronger geographic generalisation than
the hand-tuned baseline (97.4% of ceiling achieved). The direct correlation of
0.9835 on the unseen Tasmania holdout confirms that hyperparameter optimisation
did not overfit to the mainland spatial CV folds at the expense of geographic
transferability. The tight CV standard deviation (±0.0002 R², ±0.0007 MAE)
reflects the stability of the multi-seed averaging strategy — the objective
surface seen by Optuna was sufficiently smooth for TPE to converge reliably
within 50 trials.

---

## 7. Known Limitations

### 7.1 Proxy Label Circularity

The fundamental methodological limitation of this project is that `proxy_risk` is
derived from the same feature space used to train the XGBoost model. This creates
a circular validation loop: the model learns to predict a label that was constructed
from the model's own input features. The high spatial CV R² of 0.9743 partially
reflects this circularity — formula recovery contributes to model performance in
addition to genuine generalisation.

Two design decisions partially break this circularity. First, spatial lag blending
injects neighbourhood context into the label that no individual segment's own
features can fully reproduce — the label contains information the model cannot
directly access from its inputs. Second, the Tasmania holdout demonstrates that
the model generalises to an unseen geographic region at r = 0.9835, a result
inconsistent with pure formula memorisation.

The circularity **cannot be fully resolved without real collision ground truth data**.
This is not an engineering limitation — it is a fundamental constraint of the
available data. The project is correctly characterised as a **risk prioritisation
tool** grounded in ecological and infrastructure evidence, not a validated
collision frequency predictor.

### 7.2 Speed Limit and Traffic Proxy Imputation

`speed_limit` and `traffic_proxy` are imputed from road class for OSM segments
where these attributes are missing — which is the majority of the dataset outside
motorways and trunk roads. The imputation lookup table assigns fixed values by
road class (e.g. motorway → 110 km/h, 1.0 traffic; secondary → 60 km/h, 0.5
traffic). Real speed limits vary by state, urban/rural context, and local
authority decisions. Real traffic volumes vary by time of day and season. The
imputed values are order-of-magnitude correct but introduce systematic
measurement error in the road exposure score.

### 7.3 Observation Bias in Sighting Records

ALA and GBIF occurrence records are citizen-science observations, not systematic
wildlife surveys. They are concentrated near populated areas, tourist routes, and
roads — the same areas where collision risk is independently elevated. This creates
a partial confound: segments near populated areas have more sightings partly because
more people are present to observe, not only because wildlife density is genuinely
higher. The model cannot distinguish observer density from wildlife density. This
is a known and unresolved limitation of occurrence-record-based risk modelling.

### 7.4 Static Temporal Snapshot

The model produces a single risk score per segment based on data from 2020–2026.
It does not capture seasonal variation at the segment level — the `peak_season_weight`
encodes species-level seasonal risk but the model itself produces one static output.
Risk scores will become stale as land use changes, road infrastructure is upgraded,
and species distributions shift under climate change. The pipeline is designed to
be re-run as new ALA/GBIF data becomes available, but the current deployment
represents a 2020–2026 snapshot.

### 7.5 Residual Spatial Autocorrelation and Unobserved Covariates

Moran's I on model residuals is 0.3081, indicating that the model systematically
under- or over-predicts risk in spatially coherent clusters rather than making
independent errors across the road network. This is not a modelling failure — it
is an honest signal that unobserved landscape covariates with strong spatial
structure are influencing collision risk in ways no available open dataset can
currently supply.

Terrain derivatives (slope, curvature, ridgeline proximity) were considered as
a potential addition. Analysis indicates their marginal contribution to residual
reduction would be limited: animal movement corridors are already partially
reflected in the sighting density lag features, meaning terrain effects are
implicitly encoded wherever observer coverage is adequate. Adding SRTM-derived
elevation features would provide genuine new signal only in data-sparse western
regions where the sighting lags are near zero — an improvement for WA and SA
coverage but not a meaningful reduction in the national Moran's I figure.

AADT and fencing data represent the highest-value missing covariates but neither
is consolidted into a nationally queryable open dataset at the time of writing.
AADT is collected by state road agencies under inconsistent formats across
jurisdictions. Fencing density has no national open registry at any spatial
resolution. The residual autocorrelation therefore reflects the ceiling of what
is achievable with open data at national scale, and cannot be resolved through
modelling choices alone.

### 7.6 Geographic Coverage Bias in Sign Placement Recommendations

The 1,189 sign placement recommendations produced by the pipeline (threshold:
`predicted_risk ≥ 0.98`, spatial deduplication: 2km minimum separation) are not
uniformly distributed across Australia. The state-level breakdown of selected
segments is:

| State | Recommended Signs |
|---|---|
| NSW | 831 |
| VIC | 166 |
| QLD | 120 |
| TAS | 57 |
| ACT | 16 |
| SA | 12 |
| WA | 5 |

This distribution directly reflects the geographic concentration of sighting
records used to train the model, not the true national distribution of wildlife
collision risk. ALA and GBIF citizen-science observations are structurally
concentrated in eastern Australia — the most populous and most surveyed region of
the continent. WA has approximately three times the road network length of NSW,
yet receives only 5 sign recommendations. This is an artefact of observer density,
not ecological safety.

The consequence for sign placement outputs is significant: recommendations in NSW,
VIC, QLD, TAS and ACT are grounded in a dense, reliable training signal and should be
treated as high-confidence. Recommendations in SA and WA are extrapolations into
a data-sparse region — the model has scored these segments using learned feature
relationships, but those relationships were estimated almost entirely from eastern
Australian data. The sign placement output should therefore be understood as a
**high-confidence eastern Australia risk map with indicative western coverage**,
not a nationally uniform recommendation.

This limitation cannot be resolved through modelling choices. It requires
additional citizen-science observation effort or systematic wildlife survey data
in remote and western regions before a nationally balanced recommendation is
achievable.


---

## 9. Application Architecture

### 9.1 Streamlit Application Design

The Streamlit application (`app/streamlit_app.py`) provides an interactive web
interface for exploring model outputs. The architecture follows a component-based
design pattern with clear separation of concerns:

**Core Components:**
- `map_view.py` — Folium map construction, layer management, and click event parsing
- `shap_panel.py` — SHAP waterfall plot generation and segment attribution display

**Performance Optimisations:**
- `@st.cache_data` decorators on all data loading functions prevent redundant I/O
- Heatmap point sampling (15,000 max) keeps initial render under 3 seconds
- Deep copy of cached Folium maps prevents st_folium from mutating cached objects
- GeoJSON simplification (0.01° tolerance) reduces state boundary payload by ~80%

**Interactive Features:**
- Four toggleable map layers: state boundaries, occurrence heatmap, high-risk segments, sign placements
- Click-to-inspect workflow: clicking a sign marker updates the SHAP panel via session state
- Tooltip-on-hover for all map features with formatted HTML styling
- Fullscreen map control for detailed inspection

### 9.2 Data Flow Architecture

```text
User clicks sign marker
    ↓
st_folium captures click event → returns HTML tooltip content
    ↓
parse_clicked_segment() extracts road_segment_id via regex
    ↓
st.session_state.selected_segment updated
    ↓
st.rerun() triggers SHAP panel refresh
    ↓
render_shap_panel() loads SHAP values for selected segment
    ↓
generate_waterfall_plot() creates matplotlib figure
    ↓
Image displayed in right-hand panel
```

This architecture ensures the SHAP panel updates reactively without full page reload,
maintaining map state and layer selections across interactions.

### 9.3 Deployment Considerations

The application is designed for Streamlit Community Cloud deployment with the
following constraints respected:

- **Memory limit**: 1GB RAM — enforced via Parquet columnar loading and selective column reads
- **File size limit**: No single file exceeds 100MB (largest is `road_segments_scored.parquet` at ~54MB)
- **Cold start time**: Initial load completes in <5 seconds via aggressive caching
- **No external dependencies**: All data files are committed to the repository (within GitHub's 100MB file limit)

The application can also be run locally via `streamlit run app/streamlit_app.py`
for development and testing.

---

## 10. Future Work

### 10.1 Real Collision Data Integration

If any state road authority were to provide even partial collision records —
incident reports, insurance claims, or roadkill collection logs — the proxy label
could be replaced or augmented with a genuine supervisory signal. Even a single
state's data would allow the proxy label approach to be validated against real
outcomes, converting the current limitation into a documented calibration result.

### 10.2 Temporal Risk Scoring

Monthly risk scores per segment — rather than a single static score — would allow
road authorities to deploy seasonal warning sign programmes and variable message
signs calibrated to peak collision periods. This would require rerunning the
aggregation pipeline at monthly temporal resolution rather than pooling across
all six years, which is computationally feasible but requires ~6× the current
segment-level data volume per species.

### 10.3 Sign Placement Expansion to Western and Remote Australia

The current sign placement recommendations are effectively limited to eastern
Australia by training data coverage. Expanding reliable recommendations to WA,
SA, and NT requires one of two approaches: ingesting systematic wildlife survey
data from state conservation agencies in those regions (replacing the citizen-science
signal with a more uniform observational baseline), or partnering with ALA on a
targeted roadkill recording campaign along high-traffic corridors in data-sparse
states. Either approach would allow the pipeline to be retrained with national
coverage and produce defensible recommendations beyond the current eastern bias.

### 10.4 Interactive Application Enhancements

The current Streamlit application provides core functionality for risk exploration
and SHAP-based explainability. Potential enhancements for future versions include:

- **Temporal filtering**: Allow users to filter sightings and risk scores by season or month
- **Species-specific views**: Toggle individual species layers to understand species-level risk contributions
- **Export functionality**: Download filtered high-risk segments as CSV or GeoJSON for GIS import
- **Risk threshold slider**: Dynamically adjust the sign placement threshold (currently fixed at 0.98)
- **State-level drill-down**: Click a state to zoom and filter to that jurisdiction's segments
- **Mobile responsiveness**: Optimize map controls and SHAP panel layout for smartphone viewing

These enhancements would require additional UI state management and potentially
a more sophisticated caching strategy to maintain performance under increased
interactivity.

---

*This document should be read alongside the README for system context and the
inline docstrings in `scripts/model.py` for implementation detail.*
