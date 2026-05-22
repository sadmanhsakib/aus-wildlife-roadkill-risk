"""
Geospatial Sign Placement Engine for Wildlife Roadkill Risk Mapping.
===================================================================
This module processes high-risk road segments and performs spatial deduplication
(using a sliding-window spatial buffer check) to recommend optimal, strictly
distanced locations for wildlife warning signs (minimum 2km spacing between signs)
"""

import time
import geopandas as gpd


def main() -> None:
    get_sign_placement()


def get_sign_placement() -> None:
    """
    Identifies optimal road segments for placing wildlife warning signs.

    The algorithm follows these steps:
    1. Loads the model-scored road segments dataset.
    2. Projects the data to a metric CRS (EPSG:32754) to ensure distance operations are accurate.
    3. Filters for high-risk candidate segments (predicted_risk > 0.98).
    4. Iterates through candidate segments by state, sorted in descending order of predicted risk.
    5. Performs spatial deduplication: a candidate is only accepted if its centroid is
       not within 2km (2000m) of any already-selected sign placement's buffer.
    6. Reprojects the selected segments back to WGS 84 (EPSG:4326) and saves the
       output to sign_placements.geojson.
    """
    scored_segment_gdf = gpd.read_parquet("data/road_segments_scored.parquet")
    
    # Ensure EPSG:32754 so distance calculations are correct (units in metres)
    if scored_segment_gdf.crs != "EPSG:32754":
        scored_segment_gdf = scored_segment_gdf.to_crs(epsg=32754)
        
    # Retain only segments with a high predicted risk
    scored_segment_gdf = scored_segment_gdf[scored_segment_gdf["predicted_risk"] > 0.98]
    
    all_selected_rows = []
    
    # Process each state independently to ensure localized top-K recommendations
    for state in scored_segment_gdf["state"].unique():
        state_gdf = scored_segment_gdf[scored_segment_gdf["state"] == state]
        
        # Sort candidates descending by predicted risk to prioritize the most dangerous sites
        state_gdf = state_gdf.sort_values(by="predicted_risk", ascending=False).reset_index(drop=True)
        
        # Compute centroid and 2km spatial buffer in metric units (metres)
        state_gdf["centroid"] = state_gdf.geometry.centroid
        state_gdf["buffer"] = state_gdf.geometry.buffer(2000)
        
        selected_buffers = []
        selected_state_rows = []
        
        # Spatial deduplication loop: select the highest risk segments first
        for index, road_segment in state_gdf.iterrows():
            centroid = road_segment["centroid"]
            within_2000 = False
            
            # Check if this segment centroid falls inside any accepted sign placement's 2km buffer
            for buffer in selected_buffers:
                if centroid.within(buffer):
                    within_2000 = True
                    break
            
            if not within_2000:
                selected_buffers.append(road_segment["buffer"])
                selected_state_rows.append(road_segment)
        all_selected_rows.extend(selected_state_rows)
        
    if all_selected_rows:
        # Reconstruct the GeoDataFrame from the selected rows
        final_gdf = gpd.GeoDataFrame(all_selected_rows, crs="EPSG:32754")

        # Keep only the essential attributes required for application layers and GIS exports
        columns_to_keep = [
            "road_segment_id", "state", "road_class", 
            "speed_limit", "predicted_risk", "geometry"
        ]
        final_gdf = final_gdf[columns_to_keep]
        
        # Reproject back to WGS 84 (EPSG:4326) for interactive web maps / Streamlit integration
        final_gdf = final_gdf.to_crs(epsg=4326)
        
        # Save output to a GeoJSON file at the root directory
        final_gdf.to_file("data/sign_placements.geojson", driver="GeoJSON")
        print(f"✅ Successfully saved {len(final_gdf)} sign placements to data/sign_placements.geojson")
    else:
        print("⚠️ No sign placements matched the filtering criteria.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"✅ Execution completed in {time.time() - start_time:.2f} seconds")

