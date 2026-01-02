"""
Download NHANES data using the nhanes Python library

This approach uses the official nhanes package which handles
URL management and data downloading automatically.
"""

import nhanes
import pandas as pd
from pathlib import Path

Path("data/nhanes").mkdir(parents=True, exist_ok=True)

print("="*80)
print("DOWNLOADING NHANES DATA USING nhanes LIBRARY")
print("="*80)

# Survey years with both insulin and LDH
years = ['2007-2008', '2009-2010', '2011-2012', '2013-2014']

all_data = []

for year in years:
    print(f"\nProcessing {year}...")

    try:
        # Demographics
        print(f"  Downloading demographics...")
        demo = nhanes.load('DEMO', year=year)
        demo = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()

        # Glucose and Insulin
        print(f"  Downloading glucose/insulin...")
        glu = nhanes.load('GLU', year=year)

        # For some years, insulin is separate
        if year == '2013-2014':
            ins = nhanes.load('INS', year=year)
            glu = glu.merge(ins, on='SEQN', how='outer')

        # Biochemistry (LDH)
        print(f"  Downloading biochemistry...")
        biopro = nhanes.load('BIOPRO', year=year)

        # CRP
        print(f"  Downloading CRP...")
        crp = nhanes.load('CRP', year=year)

        # Medical conditions (cancer)
        print(f"  Downloading medical conditions...")
        mcq = nhanes.load('MCQ', year=year)

        # Merge all
        df = demo.copy()
        df = df.merge(glu, on='SEQN', how='left')
        df = df.merge(biopro, on='SEQN', how='left')
        df = df.merge(crp, on='SEQN', how='left')
        df = df.merge(mcq, on='SEQN', how='left')

        df['year'] = year
        all_data.append(df)

        print(f"  ✓ Loaded {len(df)} participants")

    except Exception as e:
        print(f"  ✗ Error: {e}")

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    output_path = Path("data/nhanes/nhanes_combined_raw.csv")
    combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(combined)} total participants to {output_path}")
    print(f"\nColumns available: {list(combined.columns)[:20]}...")
else:
    print("\n✗ No data downloaded")

print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
