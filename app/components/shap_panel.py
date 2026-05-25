import io
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


@st.cache_resource
def load_feature_cols(path: str = "data/model/feature_cols.pkl") -> list[str]:
    return list(joblib.load(path))


@st.cache_data
def load_shap_values(path: str = "data/model/shap_values.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.set_index("road_segment_id", drop=False)


@st.cache_data
def load_feature_values(path: str = "data/model/road_segments_scored.parquet") -> pd.DataFrame:
    feature_cols = load_feature_cols()
    df = pd.read_parquet(path, columns=["road_segment_id"] + feature_cols)
    return df.set_index("road_segment_id", drop=False)


@st.cache_resource
def warmup_shap_caches() -> None:
    load_shap_values()
    load_feature_values()


@st.cache_data
def _waterfall_image(segment_id: int) -> bytes:
    shap_df = load_shap_values()
    feature_df = load_feature_values()

    exclude_cols = ["road_segment_id", "expected_value"]
    feature_cols = [c for c in shap_df.columns if c not in exclude_cols]

    shap_row = shap_df.loc[segment_id, feature_cols]
    feature_row = feature_df.loc[segment_id, feature_cols]
    expected_value = shap_df.loc[segment_id, "expected_value"]

    explanation = shap.Explanation(
        values=shap_row.values,
        base_values=float(expected_value),
        data=feature_row.values,
        feature_names=list(shap_row.index),
    )

    fig, _ = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False)
    fig.patch.set_facecolor("white")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def render_shap_panel(segment_id: int | None = None) -> None:
    if segment_id is None:
        st.caption("Click a sign on the map to see why it was flagged.")
        return

    segment_id = int(segment_id)
    shap_df = load_shap_values()

    if segment_id not in shap_df.index:
        st.warning(f"No SHAP data found for segment `{segment_id}`.")
        return

    shap_row = shap_df.loc[segment_id]
    risk_score = shap_row.drop(labels=["road_segment_id", "expected_value"]).sum()

    st.caption(f"Segment `{segment_id}` — predicted risk **{risk_score:.4f}**")
    st.image(_waterfall_image(segment_id), width="stretch")
