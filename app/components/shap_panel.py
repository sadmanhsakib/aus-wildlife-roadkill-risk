import io
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Use non-interactive backend for faster rendering
matplotlib.use('Agg')


@st.cache_data
def load_feature_columns(path: str = "data/model/feature_cols.pkl") -> list[str]:
    """Load the list of feature column names used in the model."""
    return list(joblib.load(path))


@st.cache_data
def load_shap_values(path: str = "data/model/shap_values.parquet") -> pd.DataFrame:
    """Load pre-computed SHAP values for all road segments.
    
    Returns:
        DataFrame indexed by road_segment_id with SHAP values for each feature
    """
    df = pd.read_parquet(path)
    return df.set_index("road_segment_id", drop=False)


@st.cache_data
def load_feature_values(path: str = "data/model/road_segments_scored.parquet") -> pd.DataFrame:
    """Load feature values for all road segments (for SHAP display).
    
    Returns:
        DataFrame indexed by road_segment_id with feature values
    """
    feature_cols = load_feature_columns()
    df = pd.read_parquet(path, columns=["road_segment_id"] + feature_cols)
    return df.set_index("road_segment_id", drop=False)


@st.cache_data
def generate_waterfall_plot(segment_id: int) -> bytes:
    """Generate a SHAP waterfall plot for a specific road segment.
    
    Args:
        segment_id: The road segment ID to visualize
        
    Returns:
        PNG image bytes of the waterfall plot
    """
    shap_df = load_shap_values()
    feature_df = load_feature_values()

    # Get SHAP values and feature values for this segment
    exclude_cols = ["road_segment_id", "expected_value"]
    feature_cols = [c for c in shap_df.columns if c not in exclude_cols]

    shap_values = shap_df.loc[segment_id, feature_cols].values
    feature_values = feature_df.loc[segment_id, feature_cols].values
    expected_value = float(shap_df.loc[segment_id, "expected_value"])

    # Create SHAP explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=feature_values,
        feature_names=feature_cols,
    )

    # Generate waterfall plot
    fig, _ = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False)
    fig.patch.set_facecolor("white")

    # Save to bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    return buffer.getvalue()


def render_shap_panel(segment_id: int | None = None) -> None:
    """Render the SHAP feature attribution panel.
    
    Args:
        segment_id: Road segment ID to show attribution for, or None
    """
    if segment_id is None:
        st.caption("Click a sign on the map to see why it was flagged.")
        return

    segment_id = int(segment_id)
    shap_df = load_shap_values()

    # Check if segment exists in SHAP data
    if segment_id not in shap_df.index:
        st.warning(f"No SHAP data found for segment `{segment_id}`.")
        return

    # Calculate predicted risk from SHAP values
    shap_row = shap_df.loc[segment_id]
    exclude_cols = ["road_segment_id", "expected_value"]
    risk_score = shap_row.drop(labels=exclude_cols).sum()

    # Display segment info and waterfall plot
    st.caption(f"Segment `{segment_id}` — predicted risk **{risk_score:.4f}**")
    st.image(generate_waterfall_plot(segment_id), use_container_width=True)
