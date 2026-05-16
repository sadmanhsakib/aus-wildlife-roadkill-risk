import time
import shap
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from esda.moran import Moran
from libpysal.weights import KNN
from libpysal.weights.spatial_lag import lag_spatial


def main():
    lag_cols = [
        "sighting_count",
        "species_richness",
        "mean_ndvi",
        "traffic_proxy",
    ]

    segment_gdf = gpd.read_parquet("road_segment_labels.parquet")
    segment_gdf = add_spatial_lag_features(segment_gdf, cols=lag_cols)

    model, feature_cols = train_model(segment_gdf)

    X, _, _ = get_features_and_target(segment_gdf)
    evaluate_spatial_autocorrelation(
        gdf=segment_gdf, model=model, feature_cols=feature_cols
    )

    validate_geographic_holdout(segment_gdf, lag_cols=lag_cols, holdout_state="TAS")

    compute_shap(model=model, X=X, feature_cols=feature_cols, gdf=segment_gdf)

    score_all_segments(model=model, gdf=segment_gdf, feature_cols=feature_cols)

    joblib.dump(model, "data/model.pkl")
    joblib.dump(feature_cols, "data/feature_cols.pkl")


def add_spatial_lag_features(
    gdf: gpd.GeoDataFrame, cols: list, k: int = 5
) -> gpd.GeoDataFrame:
    """
    For each feature column, adds a 'lag_<col>' column = average of
    that feature across the 5 nearest neighbours.

    This gives XGBoost explicit spatial context as input features,
    so spatial structure doesn't get left in the residuals.
    """
    gdf = gdf.copy()
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "r"

    for col in cols:
        gdf[f"lag_{col}"] = lag_spatial(w, gdf[col].values)

    return gdf


def assign_jittered_blocks(
    gdf: gpd.GeoDataFrame, block_size: int = 50_000, jitter_range: int = 15_000
) -> pd.Series:
    """
    Assigns each road segment to a spatial block with jittered boundaries.

    Why jitter? Fixed grid boundaries let segments near edges consistently
    appear in the same fold. Jittering means boundary segments land in
    different folds across runs — preventing the model from implicitly
    learning the grid structure.

    Args:
        gdf: GeoDataFrame in EPSG:32754 (metres), so block_size is in metres
        block_size: side length of each block in metres (default 50km)
        jitter_range: max random offset applied to boundaries (default 15km)

    Returns:
        GeoDataFrame with block_x, block_y, and block_id columns
    """
    jitter_x = np.random.uniform(-jitter_range, jitter_range)
    jitter_y = np.random.uniform(-jitter_range, jitter_range)

    centroids = gdf.geometry.centroid
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Shift the origin by the jitter before dividing into blocks
    bx = ((centroids.x - bounds[0] + jitter_x) / block_size).astype(int)
    by = ((centroids.y - bounds[1] + jitter_y) / block_size).astype(int)

    return bx.astype(str) + "_" + by.astype(str)


def get_features_and_target(gdf: gpd.GeoDataFrame):
    """
    Separates the GeoDataFrame into feature matrix X and target vector y.
    Drops non-feature columns (geometry, IDs, intermediate scores).
    """
    drop_cols = [
        "road_segment_id",
        "state",
        "geometry",
        "ecological_score",
        "road_exposure_score",
        "raw_risk",
        "road_class",
        "spatial_lag",
        "blended_risk",
        "proxy_risk",
    ]
    feature_cols = [c for c in gdf.columns if c not in drop_cols]

    X = gdf[feature_cols].values
    y = gdf["proxy_risk"].values

    return X, y, feature_cols


def train_model(gdf: gpd.GeoDataFrame, n_folds: int = 5):
    """
    Trains XGBoost with stochastic spatial block cross-validation.
    Each fold uses a freshly jittered block assignment.
    """
    X, y, feature_cols = get_features_and_target(gdf)

    model = XGBRegressor(
        n_estimators=500,  # number of trees
        learning_rate=0.05,  # step size shrinkage
        max_depth=6,  # maximum depth of a tree
        subsample=0.8,  # fraction of samples used for fitting the tree
        colsample_bytree=0.8,  # fraction of features used for fitting the tree
        random_state=67,  # for reproducibility
        n_jobs=-1,  # allowing the model to use all available cores
    )

    cv_r2_scores = []
    cv_mae_scores = []

    block_ids = assign_jittered_blocks(gdf)
    groups = block_ids.values
    gkf = GroupKFold(n_splits=n_folds)

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        cv_r2_scores.append(r2_score(y_test, preds))
        cv_mae_scores.append(mean_absolute_error(y_test, preds))

    print(f"Spatial CV R²:  {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
    print(f"Spatial CV MAE: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")

    # Final fit on all data
    model.fit(X, y)

    return model, feature_cols


def evaluate_spatial_autocorrelation(gdf: gpd.GeoDataFrame, model, feature_cols: list):
    """
    Moran's I close to 0 (p > 0.05) = residuals are spatially random
    = model learned feature relationships, not just proximity
    """
    X = gdf[feature_cols].values
    gdf = gdf.copy()
    gdf["predicted_risk"] = model.predict(X)
    gdf["residual"] = gdf["predicted_risk"] - gdf["proxy_risk"]

    w = KNN.from_dataframe(gdf, k=5)
    w.transform = "r"

    mi = Moran(gdf["residual"].values, w)
    print(f"Moran's I on residuals: {mi.I:.4f}  (p={mi.p_sim:.4f})")
    print(
        "✅ No spatial leakage"
        if mi.p_sim > 0.05
        else "⚠️ Spatial autocorrelation detected in residuals"
    )

    # Confirm autocorrelation in the TARGET itself, not just residuals
    mi_target = Moran(gdf["proxy_risk"].values, w)
    print(
        f"Moran's I on proxy_risk target: {mi_target.I:.4f}  (p={mi_target.p_sim:.4f})"
    )

    return mi


def validate_geographic_holdout(
    gdf: gpd.GeoDataFrame,
    lag_cols: list,
    holdout_state: str = "TAS",
    n_folds: int = 5,
):
    """
    Retrains the model on all states except holdout_state, predicts on
    the holdout, then validates against raw sighting_count — an independent
    observational signal never directly used in proxy_risk construction.

    This partially breaks the proxy label circularity: the model has never
    seen the holdout region during training, so correlation with sighting
    density there constitutes independent geographic validation.
    """
    from scipy.stats import spearmanr

    # Split BEFORE adding lag features
    # Drop existing lag columns first — recompute on train only
    lag_feature_cols = [c for c in gdf.columns if c.startswith("lag_")]
    gdf_clean = gdf.drop(columns=lag_feature_cols)

    train_gdf = gdf_clean[gdf_clean["state"] != holdout_state].copy()
    holdout_gdf = gdf_clean[gdf_clean["state"] == holdout_state].copy()

    train_gdf = add_spatial_lag_features(train_gdf, cols=lag_cols)
    holdout_gdf = add_spatial_lag_features(holdout_gdf, cols=lag_cols)

    print(f"\nGeographic holdout validation — {holdout_state}")
    print(f"  Train segments : {len(train_gdf)}")
    print(f"  Holdout segments: {len(holdout_gdf)}")

    # Retrain on mainland only
    X_train, y_train, feature_cols = get_features_and_target(train_gdf)
    X_holdout, _, _ = get_features_and_target(holdout_gdf)

    holdout_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=67,
        n_jobs=-1,
    )

    # Spatial block CV on training data only
    cv_r2, cv_mae = [], []

    block_ids = assign_jittered_blocks(train_gdf)
    groups = block_ids.values
    gkf = GroupKFold(n_splits=n_folds)

    for train_idx, test_idx in gkf.split(X_train, y_train, groups):
        holdout_model.fit(X_train[train_idx], y_train[train_idx])
        preds = holdout_model.predict(X_train[test_idx])
        cv_r2.append(r2_score(y_train[test_idx], preds))
        cv_mae.append(mean_absolute_error(y_train[test_idx], preds))

    print(f"  Mainland CV R²:  {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
    print(f"  Mainland CV MAE: {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}")

    # Final fit on all mainland data
    holdout_model.fit(X_train, y_train)

    # Predict on Tasmania
    holdout_gdf["predicted_risk"] = holdout_model.predict(X_holdout)

    # After predicting on holdout_gdf

    # 1. Ceiling
    ceiling, _ = spearmanr(holdout_gdf["proxy_risk"], holdout_gdf["sighting_count"])

    # 2. Model vs sighting count
    corr, pval = spearmanr(holdout_gdf["predicted_risk"], holdout_gdf["sighting_count"])

    # 3. Direct generalisation
    direct, _ = spearmanr(holdout_gdf["predicted_risk"], holdout_gdf["proxy_risk"])

    print(f"\n  Ceiling (proxy_risk vs sighting_count):       {ceiling:.4f}")
    print(f"  Model  (predicted_risk vs sighting_count):    {corr:.4f}  (p={pval:.4f})")
    print(f"  % of ceiling achieved:                        {corr/ceiling*100:.1f}%")
    print(f"  Direct (predicted_risk vs proxy_risk):        {direct:.4f}")

    if corr / ceiling > 0.90:
        print("  ✅ Strong geographic generalisation")
    elif corr / ceiling > 0.70:
        print("  ⚠️ Moderate geographic generalisation")
    else:
        print("  ❌ Weak geographic generalisation")

    return holdout_gdf, corr, pval


def compute_shap(model, X: np.ndarray, feature_cols: list, gdf: gpd.GeoDataFrame):
    """
    Computes SHAP values for all segments and saves keyed by road_segment_id.
    TreeExplainer is exact (not sampled) for XGBoost — no approximation.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_segments, n_features)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, "road_segment_id", gdf["road_segment_id"].values)
    shap_df.to_parquet("data/shap_values.parquet", index=False)

    print(f"SHAP values saved — shape: {shap_df.shape}")
    return shap_df


def score_all_segments(model, gdf: gpd.GeoDataFrame, feature_cols: list):
    """
    Runs final predictions on all segments and saves the scored parquet.
    """
    X = gdf[feature_cols].values
    gdf = gdf.copy()
    gdf["predicted_risk"] = model.predict(X)
    gdf.to_parquet("data/road_segments_scored.parquet", index=False)
    print(f"Scored parquet saved — {len(gdf)} segments")
    return gdf


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"✅ Analysis pipeline completed in {time.time() - start_time:.2f} seconds")
