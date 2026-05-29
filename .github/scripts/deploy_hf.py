from huggingface_hub import HfApi
import os

hf_readme = """---
title: Aus Wildlife Roadkill Risk Mapper
emoji: 🦘
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: app/streamlit_app.py
pinned: true
---
"""

requirements = """streamlit
streamlit-folium
pandas
matplotlib
joblib
shap
geopandas
branca
folium
"""

with open("HF_README.md", "w") as f:
    f.write(hf_readme)

with open("requirements.txt", "w") as f:
    f.write(requirements)

token = os.environ["HF_TOKEN"]
api = HfApi()
repo_id = "sadmanhsakib/aus-wildlife-roadkill-risk-mapper"

# Upload only required folders
for folder, repo_path in [
    ("app", "app"),
    ("data/model", "data/model"),
    ("data/processed", "data/processed"),
]:
    api.upload_folder(
        folder_path=folder,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )

# Upload README and requirements
for local, remote in [
    ("HF_README.md", "README.md"),
    ("requirements.txt", "requirements.txt"),
]:
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
