"""
Analyze LDH-Lactate correlation in different datasets

The user observed that LDH-lactate correlation should be strong but isn't in cancer patients.
Let's investigate this across all available datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import gzip

print("="*80)
print("ANALYZING LDH-LACTATE CORRELATION")
print("="*80)

# ============================================================================
# ANALYSIS 1: V2 Synthetic Data (has designed correlation)
# ============================================================================
print("\n1. V2 SYNTHETIC DATA (Designed correlation: 0.6)")
print("-" * 80)

if Path("data/training_data_v2.npz").exists():
    data_v2 = np.load("data/training_data_v2.npz", allow_pickle=True)
    X_v2 = data_v2['X']
    y_v2 = data_v2['y']
    features_v2 = data_v2['feature_names']

    # Find indices
    feature_list = list(features_v2)
    ldh_idx = feature_list.index('LDH (U/L)')
    lactate_idx = feature_list.index('Lactate (mM)')

    # Overall correlation
    ldh_all = X_v2[:, ldh_idx]
    lactate_all = X_v2[:, lactate_idx]
    r_all, p_all = pearsonr(ldh_all, lactate_all)
    print(f"Overall (all patients): r = {r_all:.3f}, p = {p_all:.3e}")

    # Healthy controls
    ldh_healthy = X_v2[y_v2 == 0, ldh_idx]
    lactate_healthy = X_v2[y_v2 == 0, lactate_idx]
    r_healthy, p_healthy = pearsonr(ldh_healthy, lactate_healthy)
    print(f"Healthy controls: r = {r_healthy:.3f}, p = {p_healthy:.3e}")

    # Cancer patients
    ldh_cancer = X_v2[y_v2 == 1, ldh_idx]
    lactate_cancer = X_v2[y_v2 == 1, lactate_idx]
    r_cancer, p_cancer = pearsonr(ldh_cancer, lactate_cancer)
    print(f"Cancer patients: r = {r_cancer:.3f}, p = {p_cancer:.3e}")

    print(f"\nDifference: Cancer correlation is {abs(r_cancer - r_healthy):.3f} {'lower' if r_cancer < r_healthy else 'higher'} than healthy")
else:
    print("V2 data not found")

# ============================================================================
# ANALYSIS 2: V3 MIMIC-Matched Data
# ============================================================================
print("\n2. V3 MIMIC-MATCHED SYNTHETIC DATA")
print("-" * 80)

if Path("data/training_data_v3_mimic_matched.npz").exists():
    data_v3 = np.load("data/training_data_v3_mimic_matched.npz", allow_pickle=True)
    X_v3 = data_v3['X']
    y_v3 = data_v3['y']
    features_v3 = data_v3['feature_names']

    # Find indices
    feature_list = list(features_v3)
    ldh_idx = feature_list.index('LDH')
    lactate_idx = feature_list.index('Lactate')

    # Overall correlation
    ldh_all = X_v3[:, ldh_idx]
    lactate_all = X_v3[:, lactate_idx]
    r_all, p_all = pearsonr(ldh_all, lactate_all)
    print(f"Overall (all patients): r = {r_all:.3f}, p = {p_all:.3e}")

    # Healthy controls
    ldh_healthy = X_v3[y_v3 == 0, ldh_idx]
    lactate_healthy = X_v3[y_v3 == 0, lactate_idx]
    r_healthy, p_healthy = pearsonr(ldh_healthy, lactate_healthy)
    print(f"Healthy controls: r = {r_healthy:.3f}, p = {p_healthy:.3e}")

    # Cancer patients
    ldh_cancer = X_v3[y_v3 == 1, ldh_idx]
    lactate_cancer = X_v3[y_v3 == 1, lactate_idx]
    r_cancer, p_cancer = pearsonr(ldh_cancer, lactate_cancer)
    print(f"Cancer patients: r = {r_cancer:.3f}, p = {p_cancer:.3e}")

    print(f"\nDifference: Cancer correlation is {abs(r_cancer - r_healthy):.3f} {'lower' if r_cancer < r_healthy else 'higher'} than healthy")
else:
    print("V3 data not found")

# ============================================================================
# ANALYSIS 3: Real MIMIC-IV Data (if available)
# ============================================================================
print("\n3. REAL MIMIC-IV DEMO DATA")
print("-" * 80)

def read_gz_csv(filename):
    with gzip.open(filename, 'rt') as f:
        return pd.read_csv(f)

try:
    # Try multiple possible paths
    possible_paths = [
        Path("external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2"),
        Path("external_datasets/mimic_demo"),
        Path("mimic-iv-clinical-database-demo-2.2")
    ]

    mimic_path = None
    for path in possible_paths:
        if path.exists():
            mimic_path = path
            break

    if mimic_path:
        patients = read_gz_csv(mimic_path / "hosp/patients.csv.gz")
        labevents = read_gz_csv(mimic_path / "hosp/labevents.csv.gz")
        diagnoses = read_gz_csv(mimic_path / "hosp/diagnoses_icd.csv.gz")
        d_icd = read_gz_csv(mimic_path / "hosp/d_icd_diagnoses.csv.gz")

        # Identify cancer patients
        cancer_codes = d_icd[
            d_icd['long_title'].str.contains(
                'cancer|carcinoma|neoplasm|malignant|melanoma|lymphoma|leukemia',
                case=False, na=False
            )
        ]
        benign_codes = d_icd[d_icd['long_title'].str.contains('benign', case=False, na=False)]
        cancer_codes = cancer_codes[~cancer_codes['icd_code'].isin(benign_codes['icd_code'])]

        cancer_patient_ids = diagnoses[
            diagnoses['icd_code'].isin(cancer_codes['icd_code'])
        ]['subject_id'].unique()

        patients['cancer'] = patients['subject_id'].isin(cancer_patient_ids).astype(int)

        # Extract biomarkers
        biomarker_items = {
            'Glucose': [50809, 50931],
            'Lactate': [50813],
            'LDH': [50954],
        }

        patient_biomarkers = []
        for subject_id in patients['subject_id']:
            patient_labs = labevents[labevents['subject_id'] == subject_id]
            biomarker_values = {'subject_id': subject_id}

            for biomarker, item_ids in biomarker_items.items():
                values = patient_labs[patient_labs['itemid'].isin(item_ids)]['valuenum'].dropna()
                if len(values) > 0:
                    biomarker_values[biomarker] = values.median()
                else:
                    biomarker_values[biomarker] = np.nan

            patient_biomarkers.append(biomarker_values)

        biomarker_df = pd.DataFrame(patient_biomarkers)
        df = patients.merge(biomarker_df, on='subject_id')
        df_complete = df.dropna(subset=['Glucose', 'Lactate', 'LDH'])

        print(f"Found {len(df_complete)} patients with complete LDH and Lactate data")
        print(f"  Cancer: {df_complete['cancer'].sum()}")
        print(f"  Control: {(1 - df_complete['cancer']).sum()}")

        # Overall correlation
        r_all, p_all = pearsonr(df_complete['LDH'], df_complete['Lactate'])
        print(f"\nOverall (all patients): r = {r_all:.3f}, p = {p_all:.3e}")

        # Healthy controls
        healthy_df = df_complete[df_complete['cancer'] == 0]
        if len(healthy_df) > 1:
            r_healthy, p_healthy = pearsonr(healthy_df['LDH'], healthy_df['Lactate'])
            print(f"Healthy controls (n={len(healthy_df)}): r = {r_healthy:.3f}, p = {p_healthy:.3e}")

        # Cancer patients
        cancer_df = df_complete[df_complete['cancer'] == 1]
        if len(cancer_df) > 1:
            r_cancer, p_cancer = pearsonr(cancer_df['LDH'], cancer_df['Lactate'])
            print(f"Cancer patients (n={len(cancer_df)}): r = {r_cancer:.3f}, p = {p_cancer:.3e}")

            if len(healthy_df) > 1:
                print(f"\nDifference: Cancer correlation is {abs(r_cancer - r_healthy):.3f} {'lower' if r_cancer < r_healthy else 'higher'} than healthy")
                print(f"\n*** FINDING: {'Correlation breakdown in cancer!' if r_cancer < 0.3 and r_healthy > 0.5 else 'Normal correlation pattern'} ***")

except Exception as e:
    print(f"Could not load MIMIC data: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND BIOLOGICAL INTERPRETATION")
print("="*80)
print("""
EXPECTED: LDH (lactate dehydrogenase) catalyzes lactate <-> pyruvate
          So LDH and lactate SHOULD be strongly correlated (r > 0.6)

OBSERVATION: If cancer patients show WEAKER LDH-lactate correlation than healthy:

  Possible explanations:
  1. INSULIN RESISTANCE: High insulin + glucose → lactate (via glycolysis)
     But LDH may not increase proportionally if insulin resistance is present

  2. WARBURG EFFECT DYSREGULATION: Cancer cells may produce lactate through
     alternative pathways that don't proportionally increase LDH

  3. MITOCHONDRIAL DYSFUNCTION: Broken TCA cycle → lactate accumulation
     But LDH expression controlled by different regulatory pathways

  4. TUMOR HETEROGENEITY: Different cancer cells have different metabolic
     profiles, leading to decorrelation at population level

TO TEST INSULIN HYPOTHESIS:
  Need datasets with: Fasting Insulin, Glucose, LDH, Lactate
  Calculate HOMA-IR (insulin resistance index)
  Test if high HOMA-IR correlates with LDH-lactate decorrelation
""")

print("\nRECOMMENDATION:")
print("  Search for public datasets with insulin + metabolic markers")
print("  Options:")
print("    1. NHANES (US health survey) - has insulin, glucose, may have LDH/lactate")
print("    2. UK Biobank - large cohort with metabolic markers")
print("    3. Full MIMIC-IV (not demo) - check insulin itemid 51676")
print("    4. Published metabolic datasets from cancer metabolism papers")
