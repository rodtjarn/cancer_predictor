"""
MIMIC-IV Demo Dataset Analysis
Analyze the structure and biomarker availability for cancer prediction model

Required biomarkers:
1. Lactate
2. Glucose
3. LDH (Lactate Dehydrogenase)
4. CRP (C-Reactive Protein)
5. Specific Gravity
6. Age
7. BMI
"""

import pandas as pd
import gzip
from pathlib import Path

# Base path
BASE_PATH = Path("/Users/per/work/claude/cancer_predictor_package/external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2")

def read_gz_csv(filename):
    """Read a gzipped CSV file"""
    with gzip.open(filename, 'rt') as f:
        return pd.read_csv(f)

print("="*80)
print("MIMIC-IV DEMO DATASET STRUCTURE ANALYSIS")
print("="*80)
print()

# 1. Lab Items Dictionary
print("1. LABORATORY BIOMARKERS AVAILABLE")
print("-" * 80)
d_labitems = read_gz_csv(BASE_PATH / "hosp/d_labitems.csv.gz")
print(f"Total lab tests available: {len(d_labitems)}")
print()

# Define biomarkers to search for
biomarkers = {
    'Lactate': ['lactate'],
    'Glucose': ['glucose'],
    'LDH': ['lactate dehydrogenase', 'ldh', ' ld'],
    'CRP': ['c-reactive', 'crp'],
    'Specific Gravity': ['specific gravity']
}

biomarker_items = {}
for biomarker, search_terms in biomarkers.items():
    # Search for each biomarker in blood/chemistry
    mask = d_labitems['label'].str.lower().str.contains('|'.join(search_terms), na=False)
    items = d_labitems[mask]

    # Filter to blood tests (most relevant for cancer)
    blood_items = items[items['fluid'] == 'Blood']

    print(f"\n{biomarker}:")
    if len(blood_items) > 0:
        print("  ✅ FOUND in blood")
        for _, row in blood_items.iterrows():
            print(f"     - Item {row['itemid']}: {row['label']} ({row['category']})")
        biomarker_items[biomarker] = blood_items['itemid'].tolist()
    else:
        # Show all matches if no blood match
        if len(items) > 0:
            print("  ⚠️  Found, but not in blood:")
            for _, row in items.head(3).iterrows():
                print(f"     - Item {row['itemid']}: {row['label']} ({row['fluid']}, {row['category']})")
        else:
            print("  ❌ NOT FOUND")

# Specific gravity (usually urine)
spec_grav = d_labitems[d_labitems['label'].str.contains('Specific Gravity', na=False)]
if len(spec_grav) > 0:
    print(f"\nSpecific Gravity (Urine):")
    print("  ✅ FOUND")
    for _, row in spec_grav.iterrows():
        print(f"     - Item {row['itemid']}: {row['label']} ({row['fluid']}, {row['category']})")
    biomarker_items['Specific Gravity'] = spec_grav['itemid'].tolist()

print()
print("="*80)

# 2. Patient Demographics
print("\n2. PATIENT DEMOGRAPHICS")
print("-" * 80)
patients = read_gz_csv(BASE_PATH / "hosp/patients.csv.gz")
print(f"Total patients in demo: {len(patients)}")
print(f"\nPatients table columns:")
for col in patients.columns:
    print(f"  - {col}")

print()
admissions = read_gz_csv(BASE_PATH / "hosp/admissions.csv.gz")
print(f"\nAdmissions table columns:")
for col in admissions.columns:
    print(f"  - {col}")

# 3. OMR (for BMI and vital signs)
print()
print("\n3. VITAL SIGNS & BMI (OMR table)")
print("-" * 80)
omr = read_gz_csv(BASE_PATH / "hosp/omr.csv.gz")
print(f"OMR records: {len(omr)}")
print(f"\nOMR columns:")
for col in omr.columns:
    print(f"  - {col}")

# Check what result_name values exist
if 'result_name' in omr.columns:
    print(f"\nUnique result types in OMR:")
    result_types = omr['result_name'].value_counts()
    for result_type, count in result_types.items():
        print(f"  - {result_type}: {count} records")

# 4. Sample Lab Events
print()
print("\n4. LABORATORY EVENTS DATA")
print("-" * 80)
labevents = read_gz_csv(BASE_PATH / "hosp/labevents.csv.gz")
print(f"Total lab events: {len(labevents)}")
print(f"\nLabevents columns:")
for col in labevents.columns:
    print(f"  - {col}")

print(f"\nSample lab events:")
print(labevents.head(10))

# Check for our biomarkers in the actual lab events
print()
print("\n5. BIOMARKER DATA AVAILABILITY")
print("-" * 80)
for biomarker, itemids in biomarker_items.items():
    if itemids:
        count = labevents[labevents['itemid'].isin(itemids)]['subject_id'].nunique()
        total_measurements = len(labevents[labevents['itemid'].isin(itemids)])
        print(f"\n{biomarker}:")
        print(f"  - Patients with measurements: {count}/{len(patients)}")
        print(f"  - Total measurements: {total_measurements}")

# 6. Diagnoses (Cancer)
print()
print("\n6. CANCER DIAGNOSES")
print("-" * 80)
diagnoses = read_gz_csv(BASE_PATH / "hosp/diagnoses_icd.csv.gz")
d_icd_diagnoses = read_gz_csv(BASE_PATH / "hosp/d_icd_diagnoses.csv.gz")

# Search for cancer diagnoses
cancer_codes = d_icd_diagnoses[
    d_icd_diagnoses['long_title'].str.contains('cancer|carcinoma|neoplasm|malignant',
                                                case=False, na=False)
]
print(f"Cancer-related ICD codes found: {len(cancer_codes)}")
print(f"\nSample cancer diagnoses:")
print(cancer_codes[['icd_code', 'long_title']].head(10))

# Check which patients have cancer diagnoses
cancer_patients = diagnoses[diagnoses['icd_code'].isin(cancer_codes['icd_code'])]['subject_id'].unique()
print(f"\nPatients with cancer diagnoses: {len(cancer_patients)}/{len(patients)}")

print()
print("="*80)
print("\n7. SUMMARY - CAN WE TEST THE CANCER PREDICTION MODEL?")
print("="*80)

coverage = {
    'Lactate': 'Lactate' in biomarker_items,
    'Glucose': 'Glucose' in biomarker_items,
    'LDH': 'LDH' in biomarker_items,
    'CRP': 'CRP' in biomarker_items,
    'Specific Gravity': 'Specific Gravity' in biomarker_items,
    'Age': 'anchor_age' in patients.columns or 'admittime' in admissions.columns,
    'BMI': 'BMI' in str(omr['result_name'].unique()) if 'result_name' in omr.columns else False
}

covered = sum(coverage.values())
total = len(coverage)

print(f"\nBiomarker Coverage: {covered}/{total} ({100*covered/total:.0f}%)")
print()
for biomarker, available in coverage.items():
    status = "✅" if available else "❌"
    print(f"  {status} {biomarker}")

print()
if covered >= 6:
    print("✅ EXCELLENT! Demo dataset has {}/{} biomarkers".format(covered, total))
    print("   This is sufficient to test the cancer prediction model.")
else:
    print(f"⚠️  Demo dataset has {covered}/{total} biomarkers")
    print("   May be sufficient for preliminary testing, but full MIMIC-IV access recommended.")

print()
print("Next Steps:")
print("  1. Extract patient-level data for patients with cancer diagnoses")
print("  2. Merge biomarker measurements with patient demographics")
print("  3. Create feature matrix matching model input format")
print("  4. Test cancer prediction model")
print()
print("="*80)
