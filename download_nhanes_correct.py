"""
Download NHANES data with correct URLs

Files available at: https://wwwn.cdc.gov/Nchs/Nhanes/[YEAR]/[FILE].XPT
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import ssl

# Create data directory
Path("data/nhanes").mkdir(parents=True, exist_ok=True)

print("="*80)
print("DOWNLOADING NHANES DATA - CORRECT URLS")
print("="*80)

# Create unverified SSL context (CDC servers sometimes have certificate issues)
ssl_context = ssl._create_unverified_context()

CYCLES = [
    ("2007-2008", "E"),
    ("2009-2010", "F"),
    ("2011-2012", "G"),
    ("2013-2014", "H"),
]

def download_file(url, local_path):
    """Download a file from URL with SSL workaround"""
    try:
        print(f"    Downloading {url}...")
        with urllib.request.urlopen(url, context=ssl_context, timeout=30) as response:
            data = response.read()
            with open(local_path, 'wb') as f:
                f.write(data)
        print(f"    ✓ Saved to {local_path.name} ({len(data)/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False

for cycle, suffix in CYCLES:
    print(f"\n{cycle}:")

    # Demographics
    url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/DEMO_{suffix}.XPT"
    download_file(url, Path(f"data/nhanes/DEMO_{suffix}.XPT"))

    # Glucose & Insulin (combined in most cycles)
    url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/GLU_{suffix}.XPT"
    download_file(url, Path(f"data/nhanes/GLU_{suffix}.XPT"))

    # Insulin separate for 2013-2014
    if cycle == "2013-2014":
        url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/INS_{suffix}.XPT"
        download_file(url, Path(f"data/nhanes/INS_{suffix}.XPT"))

    # Biochemistry (LDH)
    url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/BIOPRO_{suffix}.XPT"
    download_file(url, Path(f"data/nhanes/BIOPRO_{suffix}.XPT"))

    # CRP
    url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/CRP_{suffix}.XPT"
    download_file(url, Path(f"data/nhanes/CRP_{suffix}.XPT"))

    # Medical Conditions Questionnaire (cancer)
    url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/MCQ_{suffix}.XPT"
    download_file(url, Path(f"data/nhanes/MCQ_{suffix}.XPT"))

print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
print("\nNext: python build_nhanes_rf_model.py")
