"""
Train and Test V3 Model (MIMIC-Matched Synthetic Data)

This script:
1. Trains on 7,000 synthetic patients matched to MIMIC-IV distribution
2. Tests on held-out real MIMIC-IV patients
3. Compares V1 vs V2 vs V3 performance

HYPOTHESIS:
Training on synthetic data that matches the target distribution should
give MUCH better real-world performance than V1/V2.
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
print("TRAINING V3 MODEL (MIMIC-MATCHED SYNTHETIC DATA)")
print("="*80)

# ============================================================================
# STEP 1: Load V3 Synthetic Training Data
# ============================================================================
print("\nSTEP 1: Loading V3 MIMIC-Matched Synthetic Training Data")
print("-" * 80)

v3_data = np.load('data/training_data_v3_mimic_matched.npz', allow_pickle=True)
X_train = v3_data['X']
y_train = v3_data['y']
feature_names = v3_data['feature_names']

print(f"✓ Loaded {len(y_train)} synthetic training samples")
print(f"  Cancer: {y_train.sum()} ({100*y_train.sum()/len(y_train):.1f}%)")
print(f"  Features: {list(feature_names)}")

# ============================================================================
# STEP 2: Train V3 Model
# ============================================================================
print("\nSTEP 2: Training Random Forest Model")
print("-" * 80)

model_v3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_v3.fit(X_train, y_train)

print(f"✓ Trained model")
print(f"\nFeature Importance:")
for feat, imp in zip(feature_names, model_v3.feature_importances_):
    print(f"  {feat:15s}: {imp*100:5.1f}%")

# Save model
with open('models/model_v3_mimic_matched.pkl', 'wb') as f:
    pickle.dump({
        'model': model_v3,
        'feature_names': feature_names,
        'feature_importance': model_v3.feature_importances_,
        'version': 'v3_mimic_matched',
        'training_samples': len(y_train)
    }, f)

print(f"\n✓ Saved model to models/model_v3_mimic_matched.pkl")

# ============================================================================
# STEP 3: Test on V3 Synthetic Test Set
# ============================================================================
print("\nSTEP 3: Testing on V3 Synthetic Test Set")
print("-" * 80)

v3_test = np.load('data/test_data_v3_mimic_matched.npz', allow_pickle=True)
X_test_v3 = v3_test['X']
y_test_v3 = v3_test['y']

y_pred_synthetic = model_v3.predict(X_test_v3)
acc_synthetic = accuracy_score(y_test_v3, y_pred_synthetic)

print(f"Synthetic test accuracy: {acc_synthetic*100:.2f}%")

cm_synthetic = confusion_matrix(y_test_v3, y_pred_synthetic)
tn, fp, fn, tp = cm_synthetic.ravel()
sens_synthetic = tp / (tp + fn) if (tp + fn) > 0 else 0
spec_synthetic = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Sensitivity: {sens_synthetic*100:.1f}%")
print(f"Specificity: {spec_synthetic*100:.1f}%")

# ============================================================================
# STEP 4: Load Real MIMIC-IV Data and Test
# ============================================================================
print("\nSTEP 4: Testing on REAL MIMIC-IV Patients")
print("="*80)

BASE_PATH = Path("external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2")

def read_gz_csv(filename):
    with gzip.open(filename, 'rt') as f:
        return pd.read_csv(f)

# Load MIMIC data (same as before)
patients = read_gz_csv(BASE_PATH / "hosp/patients.csv.gz")
labevents = read_gz_csv(BASE_PATH / "hosp/labevents.csv.gz")
diagnoses = read_gz_csv(BASE_PATH / "hosp/diagnoses_icd.csv.gz")
d_icd_diagnoses = read_gz_csv(BASE_PATH / "hosp/d_icd_diagnoses.csv.gz")

# Identify cancer patients
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
df['Age'] = df['anchor_age']
df_complete = df.dropna(subset=['Glucose', 'Lactate', 'LDH', 'Age'])
df_complete['Glucose_mM'] = df_complete['Glucose'] / 18.0

# Create feature matrix
feature_cols = ['Glucose_mM', 'Age', 'Lactate', 'LDH']
X_mimic = df_complete[feature_cols].values
y_mimic = df_complete['cancer'].values

print(f"\nLoaded {len(y_mimic)} real MIMIC-IV patients with complete data")
print(f"  Cancer: {y_mimic.sum()}")
print(f"  Control: {(1-y_mimic).sum()}")

# Split MIMIC data
X_mimic_train, X_mimic_test, y_mimic_train, y_mimic_test = train_test_split(
    X_mimic, y_mimic, test_size=0.3, random_state=42, stratify=y_mimic
)

print(f"\nTest set: {len(y_mimic_test)} patients ({y_mimic_test.sum()} cancer)")

# Predict on real MIMIC patients
y_pred_mimic = model_v3.predict(X_mimic_test)
acc_mimic = accuracy_score(y_mimic_test, y_pred_mimic)

print(f"\nV3 Model Performance on Real MIMIC-IV:")
print(f"  Accuracy: {acc_mimic*100:.1f}%")

cm_mimic = confusion_matrix(y_mimic_test, y_pred_mimic)
tn, fp, fn, tp = cm_mimic.ravel()
sens_mimic = tp / (tp + fn) if (tp + fn) > 0 else 0
spec_mimic = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"  Sensitivity: {sens_mimic*100:.1f}%")
print(f"  Specificity: {spec_mimic*100:.1f}%")

print(f"\nConfusion Matrix:")
print(cm_mimic)

print(f"\nClassification Report:")
print(classification_report(y_mimic_test, y_pred_mimic, target_names=['Control', 'Cancer']))

# Cross-validation on full MIMIC dataset
cv_scores = cross_val_score(model_v3, X_mimic, y_mimic, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation:")
print(f"  Mean: {cv_scores.mean()*100:.1f}%")
print(f"  Std: {cv_scores.std()*100:.1f}%")

# ============================================================================
# STEP 5: Compare V1 vs V2 vs V3
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: V1 vs V2 vs V3")
print("="*80)

# Historical results
v1_results = {
    'synthetic_acc': 0.9921,
    'mimic_acc': 0.733,
    'sensitivity': 0.636,
    'specificity': 0.789,
    'cv_mean': 0.640,
    'cv_std': 0.049
}

v2_results = {
    'synthetic_acc': 0.9600,
    'mimic_acc': 0.529,
    'sensitivity': 0.125,
    'specificity': 0.889,
    'cv_mean': 0.527,
    'cv_std': 0.134
}

v3_results = {
    'synthetic_acc': acc_synthetic,
    'mimic_acc': acc_mimic,
    'sensitivity': sens_mimic,
    'specificity': spec_mimic,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

print("\n1. SYNTHETIC TEST SET ACCURACY:")
print(f"   V1 (Uniform):      {v1_results['synthetic_acc']*100:5.1f}%")
print(f"   V2 (Realistic):    {v2_results['synthetic_acc']*100:5.1f}%")
print(f"   V3 (MIMIC-matched):{v3_results['synthetic_acc']*100:5.1f}%")

print("\n2. REAL MIMIC-IV ACCURACY:")
print(f"   V1 (Uniform):      {v1_results['mimic_acc']*100:5.1f}%")
print(f"   V2 (Realistic):    {v2_results['mimic_acc']*100:5.1f}%  ❌ WORSE")
print(f"   V3 (MIMIC-matched):{v3_results['mimic_acc']*100:5.1f}%  ", end='')

v3_improvement = (v3_results['mimic_acc'] - v1_results['mimic_acc']) * 100
if v3_improvement > 5:
    print(f"✅ MUCH BETTER (+{v3_improvement:.1f} pp)")
elif v3_improvement > 0:
    print(f"✅ IMPROVED (+{v3_improvement:.1f} pp)")
elif v3_improvement > -5:
    print(f"≈ SIMILAR ({v3_improvement:+.1f} pp)")
else:
    print(f"❌ WORSE ({v3_improvement:+.1f} pp)")

print("\n3. SENSITIVITY:")
print(f"   V1: {v1_results['sensitivity']*100:5.1f}%")
print(f"   V2: {v2_results['sensitivity']*100:5.1f}%  ❌ TOO LOW")
print(f"   V3: {v3_results['sensitivity']*100:5.1f}%  ", end='')
sens_diff = (v3_results['sensitivity'] - v1_results['sensitivity']) * 100
print(f"{'✅' if sens_diff > 0 else '❌'} ({sens_diff:+.1f} pp)")

print("\n4. SPECIFICITY:")
print(f"   V1: {v1_results['specificity']*100:5.1f}%")
print(f"   V2: {v2_results['specificity']*100:5.1f}%")
print(f"   V3: {v3_results['specificity']*100:5.1f}%  ", end='')
spec_diff = (v3_results['specificity'] - v1_results['specificity']) * 100
print(f"{'✅' if spec_diff > 0 else '❌'} ({spec_diff:+.1f} pp)")

print("\n5. CROSS-VALIDATION STABILITY:")
print(f"   V1: {v1_results['cv_mean']*100:5.1f}% ± {v1_results['cv_std']*100:4.1f}%")
print(f"   V2: {v2_results['cv_mean']*100:5.1f}% ± {v2_results['cv_std']*100:4.1f}%  ❌ HIGH VARIANCE")
print(f"   V3: {v3_results['cv_mean']*100:5.1f}% ± {v3_results['cv_std']*100:4.1f}%  ", end='')
if v3_results['cv_std'] < v1_results['cv_std']:
    print(f"✅ MORE STABLE")
else:
    print(f"❌ LESS STABLE")

print("\n6. SIM-TO-REAL GAP:")
v1_gap = (v1_results['synthetic_acc'] - v1_results['mimic_acc']) * 100
v2_gap = (v2_results['synthetic_acc'] - v2_results['mimic_acc']) * 100
v3_gap = (v3_results['synthetic_acc'] - v3_results['mimic_acc']) * 100

print(f"   V1 gap: {v1_gap:5.1f} pp")
print(f"   V2 gap: {v2_gap:5.1f} pp  ❌ INCREASED")
print(f"   V3 gap: {v3_gap:5.1f} pp  ", end='')
if v3_gap < v1_gap:
    print(f"✅ REDUCED by {v1_gap - v3_gap:.1f} pp")
else:
    print(f"❌ INCREASED by {v3_gap - v1_gap:.1f} pp")

# Save results
results = {
    'v1': v1_results,
    'v2': v2_results,
    'v3': v3_results,
    'v3_feature_importance': dict(zip(feature_names, model_v3.feature_importances_))
}

with open('results/v1_v2_v3_comparison.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n✓ Saved comparison results to results/v1_v2_v3_comparison.pkl")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if v3_results['mimic_acc'] > v1_results['mimic_acc'] and v3_gap < v1_gap:
    print("🎉 SUCCESS: MIMIC-matched synthetic data IMPROVED performance!")
    print(f"\n   V3 Benefits:")
    print(f"   ✓ Real-world accuracy: {v1_results['mimic_acc']*100:.1f}% → {v3_results['mimic_acc']*100:.1f}% (+{v3_improvement:.1f} pp)")
    print(f"   ✓ Sim-to-real gap: {v1_gap:.1f} pp → {v3_gap:.1f} pp (-{v1_gap - v3_gap:.1f} pp)")
    print(f"\n   This proves: Training on distribution-matched synthetic data works!")

elif v3_results['mimic_acc'] > v2_results['mimic_acc']:
    print("✅ PARTIAL SUCCESS: V3 better than V2, but not better than V1")
    print(f"\n   Key finding:")
    print(f"   - V3 ({v3_results['mimic_acc']*100:.1f}%) > V2 ({v2_results['mimic_acc']*100:.1f}%) - Distribution matching helps!")
    print(f"   - But V1 ({v1_results['mimic_acc']*100:.1f}%) still best - May be due to small real sample size (n=55)")

else:
    print("⚠️  V3 did not improve over V1")
    print(f"\n   Possible reasons:")
    print(f"   - Small real dataset (n=55) → unstable distribution estimates")
    print(f"   - Need more real patients to accurately capture distribution")
    print(f"   - Full MIMIC-IV (n=73,181) would provide much better distribution")

print(f"\n📊 Recommendation:")
if v3_results['mimic_acc'] > max(v1_results['mimic_acc'], v2_results['mimic_acc']):
    print(f"   Use V3 model for deployment ({v3_results['mimic_acc']*100:.1f}% accuracy)")
else:
    print(f"   Continue using V1 model for now ({v1_results['mimic_acc']*100:.1f}% accuracy)")
    print(f"   Apply for full MIMIC-IV access to get better distribution estimates")
