"""
Download NHANES data with Fasting Insulin, Glucose, and LDH

This script downloads NHANES data from cycles that have:
- Fasting insulin
- Fasting glucose
- LDH (lactate dehydrogenase)
- Cancer diagnosis information

We'll test if insulin resistance (HOMA-IR) is elevated in cancer patients
and if it correlates with LDH levels (potentially explaining LDH-lactate decorrelation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests

# Create data directory
Path("data/nhanes").mkdir(parents=True, exist_ok=True)

print("="*80)
print("DOWNLOADING NHANES DATA")
print("="*80)

# NHANES cycles with both insulin and LDH data
# Insulin: 2007-2014 cycles
# LDH: 2007-2018 cycles (measured as part of biochemistry panel)
# Overlap: 2007-2014

CYCLES = [
    ("2007-2008", "E"),
    ("2009-2010", "F"),
    ("2011-2012", "G"),
    ("2013-2014", "H"),
]

def download_file(url, local_path):
    """Download a file from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"  ✓ Downloaded {local_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return False

def download_nhanes_data():
    """Download NHANES data files"""

    for cycle, suffix in CYCLES:
        print(f"\nDownloading {cycle} data...")

        # Demographics (for age, gender, sampling weights)
        demo_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/DEMO_{suffix}.XPT"
        demo_path = Path(f"data/nhanes/DEMO_{suffix}.XPT")
        download_file(demo_url, demo_path)

        # Glucose & Insulin (fasting)
        glu_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/GLU_{suffix}.XPT"
        glu_path = Path(f"data/nhanes/GLU_{suffix}.XPT")
        download_file(glu_url, glu_path)

        # For 2013-2014, insulin is in separate file
        if cycle == "2013-2014":
            ins_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/INS_{suffix}.XPT"
            ins_path = Path(f"data/nhanes/INS_{suffix}.XPT")
            download_file(ins_url, ins_path)

        # Biochemistry profile (includes LDH)
        # Note: LDH variable name is LBXSLDSI in some cycles
        biopro_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/BIOPRO_{suffix}.XPT"
        biopro_path = Path(f"data/nhanes/BIOPRO_{suffix}.XPT")
        download_file(biopro_url, biopro_path)

        # Medical Conditions Questionnaire (cancer history)
        mcq_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/MCQ_{suffix}.XPT"
        mcq_path = Path(f"data/nhanes/MCQ_{suffix}.XPT")
        download_file(mcq_url, mcq_path)

        # CRP (C-Reactive Protein)
        crp_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/CRP_{suffix}.XPT"
        crp_path = Path(f"data/nhanes/CRP_{suffix}.XPT")
        download_file(crp_url, crp_path)

if __name__ == "__main__":
    download_nhanes_data()

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run: python analyze_nhanes_insulin_resistance.py")
    print("  2. Test hypothesis: Does insulin resistance explain LDH-lactate decorrelation?")
    print("\nExpected findings:")
    print("  - Cancer patients have higher HOMA-IR (insulin resistance)")
    print("  - Higher HOMA-IR → higher LDH (glycolysis)")
    print("  - But lactate production may be decoupled due to mitochondrial dysfunction")
