"""
Machine Learning Pipeline for Wildlife Roadkill Risk Mapping.

This module provides the end-to-end machine learning pipeline for predicting
wildlife roadkill risk along road segments. It includes spatial lag feature
generation, model training with spatial block cross-validation, spatial
autocorrelation evaluation, geographic holdout validation, and SHAP value
computation for model interpretability.
"""

import time
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
from xgboost import XGBRegressor
from esda.moran import Moran
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from libpysal.weights import KNN
from libpysal.weights.spatial_lag import lag_spatial


def main() -> None:
    """
    Executes the main machine learning pipeline.

    Steps:
    1. Load road segment labels data.
    2. Add spatial lag features to account for local spatial context.
    3. Train the XGBoost model using spatial block cross-validation.
    4. Evaluate spatial autocorrelation in model residuals.
    5. Perform geographic holdout validation (default: Tasmania).
    6. Compute SHAP values for model interpretability.
    7. Score all road segments and save the final predictions.
    8. Serialize the trained model and feature columns.
    """
    lag_cols = [
        "sighting_count",
        "species_richness",
        "mean_ndvi",
        "traffic_proxy",
    ]

    print("Loading road segment labels...")
    segment_gdf = gpd.read_parquet("road_segment_labels.parquet")

    print("Adding spatial lag features...")
    segment_gdf = add_spatial_lag_features(segment_gdf, cols=lag_cols)

    # Extract features and target once to avoid redundant computations
    X, y, feature_cols = get_features_and_target(segment_gdf)

    print("Training XGBoost model...")
    model = train_model(segment_gdf, X, y)
    
    print("Evaluating spatial autocorrelation...")
    evaluate_spatial_autocorrelation(gdf=segment_gdf, model=model, X=X)

    validate_geographic_holdout(segment_gdf, lag_cols=lag_cols, holdout_state="TAS")

    print("Computing SHAP values...")
    compute_shap(model=model, X=X, feature_cols=feature_cols, gdf=segment_gdf)

    print("Scoring all segments...")
    score_all_segments(model=model, gdf=segment_gdf, X=X)

    print("Saving model artifacts...")
    joblib.dump(model, "data/model.pkl")
    joblib.dump(feature_cols, "data/feature_cols.pkl")


def add_spatial_lag_features(
    gdf: gpd.GeoDataFrame, cols: List[str], k: int = 5
) -> gpd.GeoDataFrame:
    """
    Calculates and adds spatial lag features to the GeoDataFrame.

    For each specified feature column, this function calculates the 'spatial lag'
    (the average value of that feature across the `k` nearest neighbours) and 
    adds it as a new column prefixed with 'lag_'.

    This explicitly provides the XGBoost model with spatial context, reducing the
    likelihood of spatial structure being left in the model's residuals.

    Args:
        gdf: Input GeoDataFrame containing the road segments.
        cols: List of column names to compute the spatial lag for.
        k: Number of nearest neighbours to use for the spatial weights matrix.

    Returns:
        A new GeoDataFrame with the added 'lag_<col>' features.
    """
    gdf = gdf.copy()
    
    # Create K-Nearest Neighbours spatial weights matrix
    w = KNN.from_dataframe(gdf, k=k)
    # Row-standardise the weights matrix so the spatial lag is a true average
    w.transform = "r"

    for col in cols:
        gdf[f"lag_{col}"] = lag_spatial(w, gdf[col].values)

    return gdf


def assign_jittered_blocks(
    gdf: gpd.GeoDataFrame, block_size: int = 50_000, jitter_range: int = 15_000
) -> pd.Series:
    """
    Assigns each road segment to a spatial block with jittered boundaries.

    Spatial blocking is used for cross-validation to prevent data leakage.
    Jittering the boundaries ensures that segments near the edges of blocks
    do not consistently fall into the same fold across different runs. This
    prevents the model from inadvertently learning the arbitrary grid structure.

    Args:
        gdf: GeoDataFrame in a projected coordinate system (e.g., EPSG:32754),
             where units are in metres.
        block_size: Side length of each spatial block in metres (default 50km).
        jitter_range: Maximum random offset applied to grid boundaries 
                      (default 15km).

    Returns:
        pd.Series containing the unique block ID (string) for each segment.
    """
    # Generate random jitter offsets
    jitter_x = np.random.uniform(-jitter_range, jitter_range)
    jitter_y = np.random.uniform(-jitter_range, jitter_range)

    centroids = gdf.geometry.centroid
    bounds = gdf.total_bounds  # Format: [minx, miny, maxx, maxy]

    # Shift the origin by the jitter before dividing into discrete blocks
    bx = ((centroids.x - bounds[0] + jitter_x) / block_size).astype(int)
    by = ((centroids.y - bounds[1] + jitter_y) / block_size).astype(int)

    # Combine X and Y indices to create a unique block ID
    return bx.astype(str) + "_" + by.astype(str)


def get_features_and_target(
    gdf: gpd.GeoDataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Separates the GeoDataFrame into a feature matrix (X) and target vector (y).

    Filters out metadata, geometries, and intermediate score columns that 
    should not be used as predictive features.

    Args:
        gdf: GeoDataFrame containing all data.

    Returns:
        A tuple containing:
        - X: Feature matrix (NumPy array).
        - y: Target vector (NumPy array).
        - feature_cols: List of column names used as features.
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
    
    # Retain only columns that are not in the drop list
    feature_cols = [c for c in gdf.columns if c not in drop_cols]

    X = gdf[feature_cols].values
    y = gdf["proxy_risk"].values

    return X, y, feature_cols


def run_spatial_cv(
    gdf: gpd.GeoDataFrame,
    X: np.ndarray,
    y: np.ndarray,
    model: XGBRegressor,
    n_folds: int = 5,
) -> Tuple[List[float], List[float]]:
    """
    Runs spatial block cross-validation using GroupKFold with jittered boundaries.

    Ensures that spatial clusters of segments are kept intact in the same folds,
    preventing spatial data leakage during CV evaluation.

    Args:
        gdf: GeoDataFrame containing spatial coordinates for grouping.
        X: Feature matrix.
        y: Target vector.
        model: The model structure to train and evaluate.
        n_folds: Number of splits/folds for cross-validation.

    Returns:
        A tuple containing:
        - cv_r2_scores: List of R² scores for each fold.
        - cv_mae_scores: List of MAE scores for each fold.
    """
    cv_r2_scores = []
    cv_mae_scores = []

    # Assign spatial blocks to prevent spatial leakage during CV
    block_ids = assign_jittered_blocks(gdf)
    groups = block_ids.values
    
    # Use GroupKFold to ensure blocks are kept intact within folds
    gkf = GroupKFold(n_splits=n_folds)

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        cv_r2_scores.append(r2_score(y_test, preds))
        cv_mae_scores.append(mean_absolute_error(y_test, preds))

    return cv_r2_scores, cv_mae_scores


def train_model(
    gdf: gpd.GeoDataFrame, X: np.ndarray, y: np.ndarray, n_folds: int = 5
) -> XGBRegressor:
    """
    Trains an XGBoost regression model using stochastic spatial block cross-validation.

    Evaluates model performance across spatially distinct folds to ensure 
    generalisability, then refits the model on the entire dataset.

    Args:
        gdf: GeoDataFrame containing training data.
        X: Feature matrix.
        y: Target vector.
        n_folds: Number of cross-validation folds.

    Returns:
        The fully trained XGBoost model.
    """
    # Initialize XGBoost Regressor with tuned hyperparameters
    model = XGBRegressor(
        n_estimators=500,        # Number of boosting rounds (trees)
        learning_rate=0.05,      # Step size shrinkage to prevent overfitting
        max_depth=6,             # Maximum tree depth
        subsample=0.8,           # Fraction of samples used per tree
        colsample_bytree=0.8,    # Fraction of features used per tree
        random_state=67,         # Seed for reproducibility
        n_jobs=-1,               # Utilize all available CPU cores
    )

    cv_r2_scores, cv_mae_scores = run_spatial_cv(gdf, X, y, model, n_folds=n_folds)

    print(f"Spatial CV R²:  {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
    print(f"Spatial CV MAE: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")

    # Refit the model on the full dataset for deployment
    model.fit(X, y)

    return model


def evaluate_spatial_autocorrelation(
    gdf: gpd.GeoDataFrame, model: XGBRegressor, X: np.ndarray
) -> Moran:
    """
    Evaluates spatial autocorrelation in the model's residuals using Moran's I.

    If the model successfully captures the spatial patterns in the data, the 
    residuals should be spatially random (Moran's I close to 0, p > 0.05). 
    Significant autocorrelation in residuals indicates spatial leakage or 
    missing spatial predictors.

    Args:
        gdf: GeoDataFrame containing the data.
        model: Trained regression model.
        X: Feature matrix.

    Returns:
        Moran's I test results object.
    """
    gdf = gdf.copy()
    gdf["predicted_risk"] = model.predict(X)
    
    # Residual = Predicted - Actual
    gdf["residual"] = gdf["predicted_risk"] - gdf["proxy_risk"]

    # Compute spatial weights matrix
    w = KNN.from_dataframe(gdf, k=5)
    w.transform = "r"

    # Calculate Moran's I on the residuals
    mi = Moran(gdf["residual"].values, w)
    print(f"Moran's I on residuals: {mi.I:.4f}  (p={mi.p_sim:.4f})")
    print(
        "✅ No spatial leakage"
        if mi.p_sim > 0.05
        else "⚠️ Spatial autocorrelation detected in residuals"
    )

    # Calculate Moran's I on the target variable for comparison
    mi_target = Moran(gdf["proxy_risk"].values, w)
    print(
        f"Moran's I on proxy_risk target: {mi_target.I:.4f}  (p={mi_target.p_sim:.4f})"
    )

    return mi


def validate_geographic_holdout(
    gdf: gpd.GeoDataFrame,
    lag_cols: List[str],
    holdout_state: str = "TAS",
    n_folds: int = 5,
) -> Tuple[gpd.GeoDataFrame, float, float]:
    """
    Performs geographic holdout validation on a specified state.

    Retrains the model excluding the `holdout_state`, predicts on the holdout, 
    and validates against the raw `sighting_count`. This tests the model's 
    ability to generalise geographically to regions it has never seen, 
    validating against an independent observational signal.

    Args:
        gdf: GeoDataFrame containing all data.
        lag_cols: Columns used to compute spatial lag features.
        holdout_state: The state to exclude from training (default "TAS").
        n_folds: Number of folds for cross-validation on the training set.

    Returns:
        A tuple containing:
        - holdout_gdf: GeoDataFrame with holdout predictions.
        - corr: Spearman correlation between predictions and sightings.
        - pval: P-value of the correlation.
    """
    # Drop existing spatial lag features so they can be recomputed cleanly
    # ensuring no information leaks from the holdout to the training set.
    lag_feature_cols = [c for c in gdf.columns if c.startswith("lag_")]
    gdf_clean = gdf.drop(columns=lag_feature_cols)

    # Split into training (mainland) and holdout (target state)
    train_gdf = gdf_clean[gdf_clean["state"] != holdout_state].copy()
    holdout_gdf = gdf_clean[gdf_clean["state"] == holdout_state].copy()

    # Recompute spatial lag features independently
    train_gdf = add_spatial_lag_features(train_gdf, cols=lag_cols)
    holdout_gdf = add_spatial_lag_features(holdout_gdf, cols=lag_cols)

    print(f"\nGeographic holdout validation — {holdout_state}")
    print(f"  Train segments : {len(train_gdf)}")
    print(f"  Holdout segments: {len(holdout_gdf)}")

    X_train, y_train, _ = get_features_and_target(train_gdf)
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

    # Perform spatial block CV on training data to assess mainland performance
    cv_r2, cv_mae = run_spatial_cv(train_gdf, X_train, y_train, holdout_model, n_folds=n_folds)

    print(f"  Mainland CV R²:  {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
    print(f"  Mainland CV MAE: {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}")

    # Retrain model on all mainland data and predict on holdout
    holdout_model.fit(X_train, y_train)
    holdout_gdf["predicted_risk"] = holdout_model.predict(X_holdout)

    # Evaluate holdout performance using Spearman rank correlation
    
    # 1. Ceiling correlation: How well the proxy target itself correlates with 
    # the independent sighting count signal in the holdout region.
    ceiling, _ = spearmanr(holdout_gdf["proxy_risk"], holdout_gdf["sighting_count"])

    # 2. Model correlation: How well model predictions correlate with sightings.
    corr, pval = spearmanr(holdout_gdf["predicted_risk"], holdout_gdf["sighting_count"])

    # 3. Direct generalisation: How well predictions correlate with the proxy target.
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


def compute_shap(
    model: XGBRegressor, 
    X: np.ndarray, 
    feature_cols: List[str], 
    gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Computes SHAP (SHapley Additive exPlanations) values for model interpretability.

    Uses TreeExplainer to calculate exact SHAP values for all segments, revealing
    how much each feature contributed to the prediction for a given segment.

    Args:
        model: Trained XGBoost model.
        X: Feature matrix.
        feature_cols: List of feature names corresponding to columns in X.
        gdf: GeoDataFrame containing metadata (e.g., road_segment_id).

    Returns:
        DataFrame containing SHAP values merged with road_segment_id.
    """
    # TreeExplainer provides exact SHAP values for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_segments, n_features)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    # Insert road_segment_id at the first position for reference
    shap_df.insert(0, "road_segment_id", gdf["road_segment_id"].values)
    
    # Save SHAP values for downstream analysis or visualisation
    shap_df.to_parquet("data/shap_values.parquet", index=False)

    print(f"SHAP values saved — shape: {shap_df.shape}")
    return shap_df


def score_all_segments(
    model: XGBRegressor, gdf: gpd.GeoDataFrame, X: np.ndarray
) -> gpd.GeoDataFrame:
    """
    Scores all road segments using the trained model and saves the output.

    Args:
        model: Fully trained XGBoost model.
        gdf: GeoDataFrame containing all road segments.
        X: Feature matrix.

    Returns:
        GeoDataFrame updated with a 'predicted_risk' column.
    """
    gdf = gdf.copy()
    
    # Generate risk scores for the entire dataset
    gdf["predicted_risk"] = model.predict(X)
    
    # Save the final scored dataset
    gdf.to_parquet("data/road_segments_scored.parquet", index=False)
    
    print(f"Scored parquet saved — {len(gdf)} segments")
    return gdf


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"✅ Analysis pipeline completed in {time.time() - start_time:.2f} seconds")
