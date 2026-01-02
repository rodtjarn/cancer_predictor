"""
Generate Synthetic Data Matched to MIMIC-IV Demo Distribution (V3)

This approach:
1. Loads real MIMIC-IV demo data (100 patients)
2. Extracts distribution parameters (means, stds, correlations) for cancer/control
3. Generates synthetic data matching MIMIC-IV distribution
4. Should eliminate population mismatch problem

This is data augmentation - creating more samples that look like real MIMIC-IV patients.
"""

import pandas as pd
import numpy as np
import gzip
import pickle
from pathlib import Path
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING MIMIC-IV MATCHED SYNTHETIC DATA (V3)")
print("="*80)

# Paths
BASE_PATH = Path("external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2")

def read_gz_csv(filename):
    """Read a gzipped CSV file"""
    with gzip.open(filename, 'rt') as f:
        return pd.read_csv(f)

# ============================================================================
# STEP 1: Load Real MIMIC-IV Data
# ============================================================================
print("\nSTEP 1: Loading Real MIMIC-IV Demo Data")
print("-" * 80)

patients = read_gz_csv(BASE_PATH / "hosp/patients.csv.gz")
labevents = read_gz_csv(BASE_PATH / "hosp/labevents.csv.gz")
diagnoses = read_gz_csv(BASE_PATH / "hosp/diagnoses_icd.csv.gz")
d_icd_diagnoses = read_gz_csv(BASE_PATH / "hosp/d_icd_diagnoses.csv.gz")

print(f"✓ Loaded {len(patients)} patients")

# ============================================================================
# STEP 2: Identify Cancer Patients
# ============================================================================
print("\nSTEP 2: Identifying Cancer Patients")
print("-" * 80)

cancer_codes = d_icd_diagnoses[
    d_icd_diagnoses['long_title'].str.contains(
        'cancer|carcinoma|neoplasm|malignant|melanoma|lymphoma|leukemia',
        case=False, na=False
    )
]
benign_codes = d_icd_diagnoses[
    d_icd_diagnoses['long_title'].str.contains('benign', case=False, na=False)
]
cancer_codes = cancer_codes[~cancer_codes['icd_code'].isin(benign_codes['icd_code'])]

cancer_patient_ids = diagnoses[
    diagnoses['icd_code'].isin(cancer_codes['icd_code'])
]['subject_id'].unique()

patients['cancer'] = patients['subject_id'].isin(cancer_patient_ids).astype(int)

print(f"✓ Cancer patients: {patients['cancer'].sum()}")
print(f"✓ Control patients: {(1 - patients['cancer']).sum()}")

# ============================================================================
# STEP 3: Extract Biomarkers
# ============================================================================
print("\nSTEP 3: Extracting 4 Biomarkers from Real MIMIC-IV Patients")
print("-" * 80)

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
df['Age'] = df['anchor_age']

# Drop patients with missing biomarkers
df_complete = df.dropna(subset=['Glucose', 'Lactate', 'LDH', 'Age'])

# Convert glucose to mM
df_complete['Glucose_mM'] = df_complete['Glucose'] / 18.0

print(f"Complete data: {len(df_complete)} patients")
print(f"  Cancer: {df_complete['cancer'].sum()}")
print(f"  Control: {(1 - df_complete['cancer']).sum()}")

# ============================================================================
# STEP 4: Extract Distribution Parameters
# ============================================================================
print("\nSTEP 4: Extracting Distribution Parameters from Real Data")
print("-" * 80)

feature_cols = ['Glucose_mM', 'Age', 'Lactate', 'LDH']

# Separate cancer and control
cancer_data = df_complete[df_complete['cancer'] == 1][feature_cols].values
control_data = df_complete[df_complete['cancer'] == 0][feature_cols].values

print(f"\nReal data statistics:")
print(f"  Cancer patients: {len(cancer_data)}")
print(f"  Control patients: {len(control_data)}")

# Calculate statistics
cancer_mean = np.mean(cancer_data, axis=0)
cancer_cov = np.cov(cancer_data.T)
control_mean = np.mean(control_data, axis=0)
control_cov = np.cov(control_data.T)

print(f"\nCancer Distribution:")
for i, feat in enumerate(feature_cols):
    print(f"  {feat:15s}: mean={cancer_mean[i]:6.2f}, std={np.sqrt(cancer_cov[i,i]):6.2f}")

print(f"\nControl Distribution:")
for i, feat in enumerate(feature_cols):
    print(f"  {feat:15s}: mean={control_mean[i]:6.2f}, std={np.sqrt(control_cov[i,i]):6.2f}")

print(f"\nFeature Correlations (Cancer):")
cancer_corr = np.corrcoef(cancer_data.T)
for i, feat1 in enumerate(feature_cols):
    for j, feat2 in enumerate(feature_cols):
        if j > i:
            print(f"  {feat1:15s} - {feat2:15s}: {cancer_corr[i,j]:+.3f}")

print(f"\nFeature Correlations (Control):")
control_corr = np.corrcoef(control_data.T)
for i, feat1 in enumerate(feature_cols):
    for j, feat2 in enumerate(feature_cols):
        if j > i:
            print(f"  {feat1:15s} - {feat2:15s}: {control_corr[i,j]:+.3f}")

# ============================================================================
# STEP 5: Generate Synthetic Data Matching MIMIC-IV Distribution
# ============================================================================
print("\nSTEP 5: Generating Synthetic Data Matched to MIMIC-IV Distribution")
print("-" * 80)

# Generate more synthetic samples to augment small dataset
n_synthetic = 10000  # Generate 10,000 synthetic patients
cancer_ratio = len(cancer_data) / (len(cancer_data) + len(control_data))

n_cancer_synthetic = int(n_synthetic * cancer_ratio)
n_control_synthetic = n_synthetic - n_cancer_synthetic

print(f"Generating {n_synthetic} synthetic patients:")
print(f"  Cancer: {n_cancer_synthetic} ({cancer_ratio*100:.1f}%)")
print(f"  Control: {n_control_synthetic} ({(1-cancer_ratio)*100:.1f}%)")

# Generate cancer patients
print("\nGenerating cancer patients from multivariate normal...")
synthetic_cancer = multivariate_normal.rvs(
    mean=cancer_mean,
    cov=cancer_cov,
    size=n_cancer_synthetic,
    random_state=42
)

# Ensure positive values (biomarkers can't be negative)
synthetic_cancer = np.abs(synthetic_cancer)

# Generate control patients
print("Generating control patients from multivariate normal...")
synthetic_control = multivariate_normal.rvs(
    mean=control_mean,
    cov=control_cov,
    size=n_control_synthetic,
    random_state=43
)

# Ensure positive values
synthetic_control = np.abs(synthetic_control)

# Combine
X_synthetic = np.vstack([synthetic_cancer, synthetic_control])
y_synthetic = np.hstack([
    np.ones(n_cancer_synthetic),
    np.zeros(n_control_synthetic)
])

# Shuffle
shuffle_idx = np.random.permutation(len(y_synthetic))
X_synthetic = X_synthetic[shuffle_idx]
y_synthetic = y_synthetic[shuffle_idx]

print(f"\n✓ Generated {len(y_synthetic)} synthetic patients")
print(f"  Cancer: {y_synthetic.sum()} ({100*y_synthetic.sum()/len(y_synthetic):.1f}%)")
print(f"  Control: {(1-y_synthetic).sum()} ({100*(1-y_synthetic).sum()/len(y_synthetic):.1f}%)")

# Verify synthetic data matches real distribution
print(f"\nSynthetic Data Statistics (should match real data):")
synthetic_cancer_data = X_synthetic[y_synthetic == 1]
synthetic_control_data = X_synthetic[y_synthetic == 0]

print(f"\nSynthetic Cancer Distribution:")
for i, feat in enumerate(feature_cols):
    print(f"  {feat:15s}: mean={np.mean(synthetic_cancer_data[:, i]):6.2f}, std={np.std(synthetic_cancer_data[:, i]):6.2f}")

print(f"\nSynthetic Control Distribution:")
for i, feat in enumerate(feature_cols):
    print(f"  {feat:15s}: mean={np.mean(synthetic_control_data[:, i]):6.2f}, std={np.std(synthetic_control_data[:, i]):6.2f}")

# ============================================================================
# STEP 6: Split and Save
# ============================================================================
print("\nSTEP 6: Splitting and Saving Data")
print("-" * 80)

# Split synthetic data
X_train, X_test, y_train, y_test = train_test_split(
    X_synthetic, y_synthetic,
    test_size=0.3,
    random_state=42,
    stratify=y_synthetic
)

print(f"Training set: {len(y_train)} samples")
print(f"  Cancer: {y_train.sum()} ({100*y_train.sum()/len(y_train):.1f}%)")

print(f"\nTest set: {len(y_test)} samples")
print(f"  Cancer: {y_test.sum()} ({100*y_test.sum()/len(y_test):.1f}%)")

# Save synthetic data
np.savez(
    'data/training_data_v3_mimic_matched.npz',
    X=X_train,
    y=y_train,
    feature_names=np.array(feature_cols)
)

np.savez(
    'data/test_data_v3_mimic_matched.npz',
    X=X_test,
    y=y_test,
    feature_names=np.array(feature_cols)
)

print(f"\n✓ Saved training data to data/training_data_v3_mimic_matched.npz")
print(f"✓ Saved test data to data/test_data_v3_mimic_matched.npz")

# Save distribution parameters for reference
distribution_params = {
    'cancer_mean': cancer_mean,
    'cancer_cov': cancer_cov,
    'control_mean': control_mean,
    'control_cov': control_cov,
    'feature_names': feature_cols,
    'n_real_cancer': len(cancer_data),
    'n_real_control': len(control_data),
    'cancer_ratio': cancer_ratio
}

with open('data/mimic_distribution_params.pkl', 'wb') as f:
    pickle.dump(distribution_params, f)

print(f"✓ Saved distribution parameters to data/mimic_distribution_params.pkl")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Extracted distribution from {len(df_complete)} real MIMIC-IV patients")
print(f"✓ Generated {n_synthetic} synthetic patients matching MIMIC-IV distribution")
print(f"✓ Preserved real correlations between biomarkers")
print(f"✓ Maintained cancer/control ratio from real data")
print(f"\nNext steps:")
print(f"  1. Train model: python train_v3_model.py")
print(f"  2. Test on real MIMIC patients (held-out)")
print(f"  3. Compare V1 vs V2 vs V3 performance")
print(f"\nExpected: V3 should perform MUCH better than V1/V2 because it matches target distribution!")
