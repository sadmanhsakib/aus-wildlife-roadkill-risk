from huggingface_hub import HfApi
import os

hf_readme = """---
title: Aus Wildlife Roadkill Risk Mapper
emoji: 🦘
colorFrom: green
colorTo: blue
sdk: docker
app_file: app/streamlit_app.py
pinned: true
---
"""

with open("HF_README.md", "w") as f:
    f.write(hf_readme)

token = os.environ["HF_TOKEN"]
api = HfApi()

api.upload_folder(
    folder_path=".",
    repo_id="sadmanhsakib/aus-wildlife-roadkill-risk-mapper",
    repo_type="space",
    token=token,
    ignore_patterns=[".git*", ".github*", "README.md"]
)

api.upload_file(
    path_or_fileobj="HF_README.md",
    path_in_repo="README.md",
    repo_id="sadmanhsakib/aus-wildlife-roadkill-risk-mapper",
    repo_type="space",
    token=token
)