import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


@st.cache_data
def load_shap_values(path: str = "data/model/shap_values.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_feature_values(path: str = "data/model/road_segments_scored.parquet") -> pd.DataFrame:
    feature_cols = list(joblib.load("data/model/feature_cols.pkl"))
    return pd.read_parquet(path, columns=["road_segment_id"] + feature_cols)


def _render_waterfall(segment_id, shap_df: pd.DataFrame, feature_df: pd.DataFrame):
    # Non-feature columns to exclude from SHAP values
    exclude_cols = ["road_segment_id", "expected_value"]
    feature_cols = [c for c in shap_df.columns if c not in exclude_cols]

    # Filter to the single row and squeeze to 1D immediately
    shap_row = shap_df.loc[shap_df["road_segment_id"] == segment_id, feature_cols].iloc[
        0
    ]
    feature_row = feature_df.loc[feature_df["road_segment_id"] == segment_id].iloc[0]

    # Drop road_segment_id from feature_row — only actual feature values
    feature_row = feature_row.drop(labels=["road_segment_id"])

    expected_value = shap_df.loc[
        shap_df["road_segment_id"] == segment_id, "expected_value"
    ].iloc[0]

    explanation = shap.Explanation(
        values=shap_row.values,  # 1D array of shape (16,)
        base_values=float(expected_value),
        data=feature_row.values,  # 1D array of shape (16,)
        feature_names=list(shap_row.index),
    )

    fig, _ = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False)
    fig.patch.set_facecolor("white")

    st.pyplot(fig)
    plt.close(fig)


def render_shap_panel(segment_id=None):
    st.subheader("Feature Attribution")

    if segment_id is None:
        st.caption("Click a segment on the map to see why it was flagged.")
        return

    shap_df = load_shap_values()
    feature_df = load_feature_values()

    shap_row = shap_df.loc[shap_df["road_segment_id"] == segment_id]

    if segment_id not in shap_df["road_segment_id"].values:
        st.warning(f"No SHAP data found for segment `{segment_id}`.")
        return

    risk_score = shap_row.drop(columns=["road_segment_id"]).values.sum()

    st.caption(f"Segment `{segment_id}` — predicted risk **{risk_score:.4f}**")
    _render_waterfall(segment_id, shap_df, feature_df)
