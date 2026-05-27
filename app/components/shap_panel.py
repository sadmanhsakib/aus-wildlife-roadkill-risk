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
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem 1rem; color: var(--color-text-tertiary);">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" 
                     stroke-width="1.5" style="margin-bottom: 1rem; opacity: 0.4;">
                    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
                    <circle cx="12" cy="10" r="3"/>
                </svg>
                <p style="font-size: 0.95rem; margin: 0; font-weight: 500; color: var(--color-text-secondary);">
                    No segment selected
                </p>
                <p style="font-size: 0.85rem; margin: 0.5rem 0 0 0; line-height: 1.5;">
                    Click on a red sign marker on the map to see detailed SHAP analysis
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    segment_id = int(segment_id)
    shap_df = load_shap_values()

    # Check if segment exists in SHAP data
    if segment_id not in shap_df.index:
        st.warning(f"⚠️ No SHAP data available for segment `{segment_id}`")
        return

    # Calculate predicted risk from SHAP values
    shap_row = shap_df.loc[segment_id]
    exclude_cols = ["road_segment_id"]
    risk_score = shap_row.drop(labels=exclude_cols).sum()

    # Display segment info
    st.markdown(
        f"""
        <div style="background: var(--color-bg-tertiary); 
                    padding: 1rem; 
                    border-radius: var(--radius-sm); 
                    margin-bottom: 1rem;
                    border-left: 3px solid var(--color-accent);">
            <div style="font-size: 0.75rem; 
                        color: var(--color-text-tertiary); 
                        text-transform: uppercase; 
                        letter-spacing: 0.05em; 
                        margin-bottom: 0.25rem;">
                Segment Analysis
            </div>
            <div style="font-size: 1.25rem; 
                        font-weight: 600; 
                        color: var(--color-text-primary);">
                ID: {segment_id}
            </div>
            <div style="font-size: 0.9rem; 
                        color: var(--color-text-secondary); 
                        margin-top: 0.5rem;">
                Predicted Risk: <strong style="color: var(--color-danger);">{risk_score:.4f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display waterfall plot
    st.image(generate_waterfall_plot(segment_id), width="stretch")

