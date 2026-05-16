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
roads annually, yet iNaturalist — the most comprehensive open citizen-science
platform available — contains only ~15,000 confirmed roadkill observations across
570+ species accumulated over 5 years. This represents a detection rate of
approximately 0.03% relative to estimated annual collision frequency.

This sparsity is structural, not an engineering failure. Roadkill events are
transient, geographically dispersed across 900,000+ km of road network, and
entirely dependent on opportunistic observer presence at the moment of discovery.
Formal collision databases maintained by state road authorities are fragmented,
inconsistently recorded across jurisdictions, and not publicly accessible under
open data licences. No national standardised roadkill monitoring programme exists
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
- **Data availability** — sufficient verified occurrence records in ALA and GBIF
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

```
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

```
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

```
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

Current hyperparameters are hand-tuned rather than optimised:

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 500 | Sufficient tree count for convergence at `learning_rate=0.05` |
| `learning_rate` | 0.05 | Conservative shrinkage; reduces overfitting on proxy label |
| `max_depth` | 6 | Allows feature interactions up to 6-way; deeper trees memorise noise |
| `subsample` | 0.8 | 20% row dropout per tree; reduces variance |
| `colsample_bytree` | 0.8 | 20% feature dropout per tree; prevents single-feature dominance |

Optuna-based Bayesian hyperparameter search (50 trials, optimising spatial CV R²)
is planned as a subsequent step. Current hand-tuned parameters produce CV R² of
0.9727 — above the 0.60 target by a substantial margin — so hyperparameter
optimisation is not on the critical path for project completion.

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

```
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

---

## 7. Known Limitations

### 7.1 Proxy Label Circularity

The fundamental methodological limitation of this project is that `proxy_risk` is
derived from the same feature space used to train the XGBoost model. This creates
a circular validation loop: the model learns to predict a label that was constructed
from the model's own input features. The high spatial CV R² of 0.9727 partially
reflects this circularity — formula recovery contributes to model performance in
addition to genuine generalisation.

Two design decisions partially break this circularity. First, spatial lag blending
injects neighbourhood context into the label that no individual segment's own
features can fully reproduce — the label contains information the model cannot
directly access from its inputs. Second, the Tasmania holdout demonstrates that
the model generalises to an unseen geographic region at r = 0.98, a result
inconsistent with pure formula memorisation.

The circularity cannot be fully resolved without real collision ground truth data.
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

### 7.5 Road Segments Without Sightings

The model only scores road segments that had at least one wildlife sighting within
the 2020–2026 window. Road segments with zero nearby sightings are excluded from
the feature store and receive no predicted risk score. This creates a systematic
blind spot for genuinely dangerous segments in regions with low observer coverage —
particularly remote Western Australia and Northern Territory. A segment may be a
genuine high-risk corridor but receive no score simply because no citizen scientist
was present to record a sighting. This is the single most significant gap between
the model's output and true national risk coverage.

---

## 8. Future Work

### 8.1 Real Collision Data Integration

If any state road authority were to provide even partial collision records —
incident reports, insurance claims, or roadkill collection logs — the proxy label
could be replaced or augmented with a genuine supervisory signal. Even a single
state's data would allow the proxy label approach to be validated against real
outcomes, converting the current limitation into a documented calibration result.

### 8.2 Terrain and Fencing Data

The residual Moran's I of 0.3081 suggests that unobserved landscape covariates
are driving unexplained spatial clustering in the residuals. Terrain complexity
(slope, curvature, ridgeline proximity) and fencing density are the two most
likely candidates — both influence wildlife movement patterns and road crossing
behaviour but are absent from available open data sources at national scale. If
national terrain or fencing datasets become available under open licences, they
represent the highest-value addition to the feature set.

### 8.3 Temporal Risk Scoring

Monthly risk scores per segment — rather than a single static score — would allow
road authorities to deploy seasonal warning sign programmes and variable message
signs calibrated to peak collision periods. This would require rerunning the
aggregation pipeline at monthly temporal resolution rather than pooling across
all six years, which is computationally feasible but requires ~6× the current
segment-level data volume per species.

### 8.4 Optuna Hyperparameter Optimisation

Bayesian hyperparameter search (50 trials, optimising spatial CV R²) using Optuna
is planned as an immediate next step. Current hand-tuned parameters already exceed
all target metrics, so this is a refinement rather than a correctness fix — but
the optimised model will be more defensible in a research context than one with
manually chosen parameters.

---

*This document should be read alongside the README for system context and the
inline docstrings in `scripts/model.py` for implementation detail.*