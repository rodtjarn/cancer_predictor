"""
Test V2 Model (Realistic Stochastics) on MIMIC-IV Real Data

This script:
1. Loads V2 model trained on realistic stochastics
2. Extracts same 100 MIMIC-IV patients used in original validation
3. Tests V2 model performance
4. Compares V1 vs V2 results

HYPOTHESIS:
More realistic synthetic data (skewed distributions, correlations, outliers)
should reduce the sim-to-real gap and improve real-world performance.
"""

import pandas as pd
import numpy as np
import gzip
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING V2 MODEL (REALISTIC STOCHASTICS) ON MIMIC-IV")
print("="*80)

# Paths
BASE_PATH = Path("external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2")

def read_gz_csv(filename):
    """Read a gzipped CSV file"""
    with gzip.open(filename, 'rt') as f:
        return pd.read_csv(f)

# ============================================================================
# STEP 1: Load MIMIC-IV Data (Same as V1 Validation)
# ============================================================================
print("\nSTEP 1: Loading MIMIC-IV Data")
print("-" * 80)

patients = read_gz_csv(BASE_PATH / "hosp/patients.csv.gz")
labevents = read_gz_csv(BASE_PATH / "hosp/labevents.csv.gz")
diagnoses = read_gz_csv(BASE_PATH / "hosp/diagnoses_icd.csv.gz")
d_icd_diagnoses = read_gz_csv(BASE_PATH / "hosp/d_icd_diagnoses.csv.gz")

print(f"✓ Loaded {len(patients)} patients")
print(f"✓ Loaded {len(labevents):,} lab events")

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
# STEP 3: Extract 4 Biomarkers (Same as V1 Validation)
# ============================================================================
print("\nSTEP 3: Extracting 4 Biomarkers (Glucose, Age, Lactate, LDH)")
print("-" * 80)

# Item IDs for biomarkers
biomarker_items = {
    'Glucose': [50809, 50931],
    'Lactate': [50813],
    'LDH': [50954],
}

# Extract biomarkers
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

# Merge with patient info
df = patients.merge(biomarker_df, on='subject_id')

# Add age
df['Age'] = df['anchor_age']

# Drop patients with missing biomarkers
print(f"Patients before filtering: {len(df)}")
df_complete = df.dropna(subset=['Glucose', 'Lactate', 'LDH', 'Age'])
print(f"Patients after filtering (complete data): {len(df_complete)}")

print(f"\nFinal dataset:")
print(f"  Cancer: {df_complete['cancer'].sum()}")
print(f"  Control: {(1 - df_complete['cancer']).sum()}")

# ============================================================================
# STEP 4: Prepare Feature Matrix (4 biomarkers - same as V1)
# ============================================================================
print("\nSTEP 4: Preparing Feature Matrix")
print("-" * 80)

# Convert glucose from mg/dL to mM (same as synthetic data)
df_complete['Glucose_mM'] = df_complete['Glucose'] / 18.0

# Create feature matrix (same order as V1 validated model)
feature_cols = ['Glucose_mM', 'Age', 'Lactate', 'LDH']
X = df_complete[feature_cols].values
y = df_complete['cancer'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {feature_cols}")

# ============================================================================
# STEP 5: Train 4-Biomarker V2 Model on V2 Synthetic Data
# ============================================================================
print("\nSTEP 5: Training 4-Biomarker V2 Model on Realistic Synthetic Data")
print("-" * 80)

# Load V2 synthetic training data
v2_data = np.load('data/training_data_v2.npz', allow_pickle=True)
X_train_v2 = v2_data['X']
y_train_v2 = v2_data['y']
feature_names_v2 = v2_data['feature_names']

print(f"V2 synthetic training data: {X_train_v2.shape}")
print(f"V2 features: {list(feature_names_v2)}")

# Extract only the 4 biomarkers we need (matching MIMIC data)
# V2 feature order: Lactate, CRP, SG, Glucose, LDH, Age, BMI
# We want: Glucose, Age, Lactate, LDH (matching MIMIC feature_cols)

v2_feature_indices = {
    'Lactate (mM)': 0,
    'CRP (mg/L)': 1,
    'Specific Gravity': 2,
    'Glucose (mM)': 3,
    'LDH (U/L)': 4,
    'Age': 5,
    'BMI': 6
}

# Select features in correct order: Glucose, Age, Lactate, LDH
selected_indices = [
    v2_feature_indices['Glucose (mM)'],
    v2_feature_indices['Age'],
    v2_feature_indices['Lactate (mM)'],
    v2_feature_indices['LDH (U/L)']
]

X_train_v2_4biomarkers = X_train_v2[:, selected_indices]

print(f"\nReduced to 4 biomarkers: {X_train_v2_4biomarkers.shape}")
print(f"Feature order: Glucose, Age, Lactate, LDH")

# Train V2 model
model_v2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_v2.fit(X_train_v2_4biomarkers, y_train_v2)

print(f"\n✓ Trained V2 model")
print("\nV2 Model Feature Importance:")
for feat, imp in zip(['Glucose', 'Age', 'Lactate', 'LDH'], model_v2.feature_importances_):
    print(f"  {feat:15s}: {imp*100:5.1f}%")

# ============================================================================
# STEP 6: Test V2 Model on MIMIC-IV
# ============================================================================
print("\nSTEP 6: Testing V2 Model on MIMIC-IV Real Patients")
print("-" * 80)

# Split MIMIC data (same as V1 validation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train set: {len(y_train)} patients ({y_train.sum()} cancer)")
print(f"Test set: {len(y_test)} patients ({y_test.sum()} cancer)")

# Predict
y_pred_v2 = model_v2.predict(X_test)

# Calculate metrics
acc_v2 = accuracy_score(y_test, y_pred_v2)
cm_v2 = confusion_matrix(y_test, y_pred_v2)
tn, fp, fn, tp = cm_v2.ravel()
sens_v2 = tp / (tp + fn) if (tp + fn) > 0 else 0
spec_v2 = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nV2 Model Performance on MIMIC-IV:")
print(f"  Accuracy: {acc_v2*100:.1f}%")
print(f"  Sensitivity: {sens_v2*100:.1f}%")
print(f"  Specificity: {spec_v2*100:.1f}%")

print("\nConfusion Matrix:")
print(cm_v2)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_v2, target_names=['Control', 'Cancer']))

# Cross-validation
cv_scores_v2 = cross_val_score(model_v2, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation:")
print(f"  Mean: {cv_scores_v2.mean()*100:.1f}%")
print(f"  Std: {cv_scores_v2.std()*100:.1f}%")

# ============================================================================
# STEP 7: Load V1 Results and Compare
# ============================================================================
print("\nSTEP 7: Comparing V1 vs V2 Results")
print("="*80)

# V1 results (from proper_validation.py)
v1_results = {
    'test_accuracy': 0.733,
    'sensitivity': 0.636,
    'specificity': 0.789,
    'cv_mean': 0.640,
    'cv_std': 0.049,
    'synthetic_accuracy': 0.9921  # From original training
}

v2_results = {
    'test_accuracy': acc_v2,
    'sensitivity': sens_v2,
    'specificity': spec_v2,
    'cv_mean': cv_scores_v2.mean(),
    'cv_std': cv_scores_v2.std(),
    'synthetic_accuracy': 0.9600  # From train_v2_model.py
}

print("\n" + "="*80)
print("COMPARISON: V1 (Uniform) vs V2 (Realistic Stochastics)")
print("="*80)

print("\n1. SYNTHETIC TEST SET PERFORMANCE:")
print(f"   V1 accuracy: {v1_results['synthetic_accuracy']*100:.2f}%")
print(f"   V2 accuracy: {v2_results['synthetic_accuracy']*100:.2f}%")
diff_synthetic = (v2_results['synthetic_accuracy'] - v1_results['synthetic_accuracy']) * 100
print(f"   Difference: {diff_synthetic:+.2f} pp")

print("\n2. REAL MIMIC-IV TEST SET PERFORMANCE:")
print(f"   V1 accuracy: {v1_results['test_accuracy']*100:.1f}%")
print(f"   V2 accuracy: {v2_results['test_accuracy']*100:.1f}%")
diff_mimic = (v2_results['test_accuracy'] - v1_results['test_accuracy']) * 100
print(f"   Difference: {diff_mimic:+.1f} pp  {'✅ IMPROVED' if diff_mimic > 0 else '❌ WORSE'}")

print("\n3. SENSITIVITY (RECALL):")
print(f"   V1: {v1_results['sensitivity']*100:.1f}%")
print(f"   V2: {v2_results['sensitivity']*100:.1f}%")
diff_sens = (v2_results['sensitivity'] - v1_results['sensitivity']) * 100
print(f"   Difference: {diff_sens:+.1f} pp  {'✅' if diff_sens > 0 else '❌'}")

print("\n4. SPECIFICITY:")
print(f"   V1: {v1_results['specificity']*100:.1f}%")
print(f"   V2: {v2_results['specificity']*100:.1f}%")
diff_spec = (v2_results['specificity'] - v1_results['specificity']) * 100
print(f"   Difference: {diff_spec:+.1f} pp  {'✅' if diff_spec > 0 else '❌'}")

print("\n5. CROSS-VALIDATION:")
print(f"   V1: {v1_results['cv_mean']*100:.1f}% ± {v1_results['cv_std']*100:.1f}%")
print(f"   V2: {v2_results['cv_mean']*100:.1f}% ± {v2_results['cv_std']*100:.1f}%")
diff_cv = (v2_results['cv_mean'] - v1_results['cv_mean']) * 100
print(f"   Difference: {diff_cv:+.1f} pp  {'✅' if diff_cv > 0 else '❌'}")

print("\n6. SIM-TO-REAL GAP:")
v1_gap = (v1_results['synthetic_accuracy'] - v1_results['test_accuracy']) * 100
v2_gap = (v2_results['synthetic_accuracy'] - v2_results['test_accuracy']) * 100
print(f"   V1 gap: {v1_gap:.1f} pp  (synthetic - real)")
print(f"   V2 gap: {v2_gap:.1f} pp  (synthetic - real)")
gap_reduction = v1_gap - v2_gap
print(f"   Gap reduction: {gap_reduction:.1f} pp  {'✅ REDUCED' if gap_reduction > 0 else '❌ INCREASED'}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if diff_mimic > 2 and gap_reduction > 5:
    print("🎉 SUCCESS: Realistic stochastics SIGNIFICANTLY IMPROVED real-world performance!")
    print("   ✓ Better synthetic data → better real-world generalization")
    print("   ✓ Reduced sim-to-real gap by {:.1f} percentage points".format(gap_reduction))
elif diff_mimic > 0:
    print("✅ MODERATE SUCCESS: Realistic stochastics slightly improved performance")
    print(f"   ✓ Real-world accuracy improved by {diff_mimic:.1f} pp")
    if gap_reduction > 0:
        print(f"   ✓ Sim-to-real gap reduced by {gap_reduction:.1f} pp")
else:
    print("⚠️  Realistic stochastics did NOT improve real-world performance")
    print("   Possible reasons:")
    print("   - MIMIC-IV distribution still very different from improved synthetic")
    print("   - Small sample size (n=100) creates high variance")
    print("   - May need even more realistic synthetic data generation")

# Save results
results = {
    'v1': v1_results,
    'v2': v2_results,
    'comparison': {
        'synthetic_diff': diff_synthetic,
        'mimic_diff': diff_mimic,
        'sensitivity_diff': diff_sens,
        'specificity_diff': diff_spec,
        'cv_diff': diff_cv,
        'gap_reduction': gap_reduction
    }
}

with open('results/v1_vs_v2_mimic_comparison.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n✓ Saved comparison results to results/v1_vs_v2_mimic_comparison.pkl")
