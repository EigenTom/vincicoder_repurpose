#!/usr/bin/env python3
"""
Download VinciCoder-1.6M-SFT splits needed by generate_vl_cot_v2.py.

Downloads:
  - chart2code_refine.parquet
  - web2html_refine.parquet

Destination: ./vincicoder_sft/  (relative to this script's directory)

Usage:
  python download_vincicoder_sft.py
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID     = "DocTron-Hub/VinciCoder-1.6M-SFT"
SCRIPT_DIR  = Path(__file__).resolve().parent
LOCAL_DIR   = SCRIPT_DIR / "vincicoder_sft"
FILES       = ["chart2code_refine.parquet", "web2html_refine.parquet"]

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()
print(f"Listing files in {REPO_ID} ...")
available = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))

for fname in FILES:
    if fname in available:
        print(f"Downloading {fname} → {LOCAL_DIR}/")
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(LOCAL_DIR),
            repo_type="dataset",
            allow_patterns=[fname],
        )
        print(f"  ✓ {fname}")
    else:
        print(f"  ✗ {fname} not found in repo — skipping")

print("\nDownload complete.")
print(f"Files saved to: {LOCAL_DIR}")
