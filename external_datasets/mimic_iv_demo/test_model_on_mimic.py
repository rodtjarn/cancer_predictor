"""
Test Cancer Prediction Model on MIMIC-IV Demo Dataset

This script:
1. Extracts patient data from MIMIC-IV demo
2. Creates feature matrix for model v0.2.0 (6 biomarkers)
3. Tests the cancer prediction model
4. Generates performance metrics and visualizations
"""

import pandas as pd
import numpy as np
import gzip
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve, confusion_matrix,
                            classification_report)
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path("/Users/per/work/claude/cancer_predictor_package/external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2")
MODEL_PATH = Path("/Users/per/work/claude/cancer_predictor_package/models/metabolic_cancer_predictor_v2.pkl")

def read_gz_csv(filename):
    """Read a gzipped CSV file"""
    with gzip.open(filename, 'rt') as f:
        return pd.read_csv(f)

print("="*80)
print("TESTING CANCER PREDICTION MODEL ON MIMIC-IV DEMO DATASET")
print("="*80)
print()

# ============================================================================
# STEP 1: Load MIMIC-IV Data
# ============================================================================
print("STEP 1: Loading MIMIC-IV Demo Data")
print("-" * 80)

patients = read_gz_csv(BASE_PATH / "hosp/patients.csv.gz")
admissions = read_gz_csv(BASE_PATH / "hosp/admissions.csv.gz")
labevents = read_gz_csv(BASE_PATH / "hosp/labevents.csv.gz")
diagnoses = read_gz_csv(BASE_PATH / "hosp/diagnoses_icd.csv.gz")
d_icd_diagnoses = read_gz_csv(BASE_PATH / "hosp/d_icd_diagnoses.csv.gz")
omr = read_gz_csv(BASE_PATH / "hosp/omr.csv.gz")

print(f"✓ Loaded {len(patients)} patients")
print(f"✓ Loaded {len(labevents)} lab events")
print(f"✓ Loaded {len(diagnoses)} diagnoses")
print(f"✓ Loaded {len(omr)} OMR records")
print()

# ============================================================================
# STEP 2: Identify Cancer Patients
# ============================================================================
print("STEP 2: Identifying Cancer Patients")
print("-" * 80)

# Find cancer ICD codes
cancer_codes = d_icd_diagnoses[
    d_icd_diagnoses['long_title'].str.contains(
        'cancer|carcinoma|neoplasm|malignant|melanoma|lymphoma|leukemia',
        case=False, na=False
    )
]

# Exclude benign neoplasms
benign_codes = d_icd_diagnoses[
    d_icd_diagnoses['long_title'].str.contains('benign', case=False, na=False)
]
cancer_codes = cancer_codes[~cancer_codes['icd_code'].isin(benign_codes['icd_code'])]

# Get patient cancer status
cancer_patient_ids = diagnoses[
    diagnoses['icd_code'].isin(cancer_codes['icd_code'])
]['subject_id'].unique()

# Create labels
patients['cancer'] = patients['subject_id'].isin(cancer_patient_ids).astype(int)

print(f"✓ Cancer patients: {patients['cancer'].sum()}")
print(f"✓ Control patients: {(1 - patients['cancer']).sum()}")
print()

# ============================================================================
# STEP 3: Extract Biomarker Measurements
# ============================================================================
print("STEP 3: Extracting Biomarker Measurements")
print("-" * 80)

# Define biomarker item IDs (from analysis)
biomarker_items = {
    'Lactate': [50813, 52442, 53154],
    'Glucose': [50809, 50931, 52027, 52569],
    'LDH': [50954],
    'CRP': [50889, 51652],
}

# Extract each biomarker
biomarker_data = {}

for biomarker, item_ids in biomarker_items.items():
    # Get all measurements for this biomarker
    mask = labevents['itemid'].isin(item_ids)
    measurements = labevents[mask][['subject_id', 'valuenum']].copy()

    # Remove invalid values
    measurements = measurements[measurements['valuenum'].notna()]
    measurements = measurements[measurements['valuenum'] > 0]

    # Aggregate by patient (median of all measurements)
    patient_values = measurements.groupby('subject_id')['valuenum'].median()

    biomarker_data[biomarker] = patient_values

    print(f"✓ {biomarker}: {len(patient_values)} patients, median = {patient_values.median():.2f}")

# Extract BMI from OMR
bmi_data = omr[omr['result_name'] == 'BMI (kg/m2)'][['subject_id', 'result_value']].copy()
bmi_data['result_value'] = pd.to_numeric(bmi_data['result_value'], errors='coerce')
bmi_data = bmi_data[bmi_data['result_value'].notna()]
bmi_data = bmi_data[bmi_data['result_value'] > 0]
bmi_values = bmi_data.groupby('subject_id')['result_value'].median()
biomarker_data['BMI'] = bmi_values

print(f"✓ BMI: {len(bmi_values)} patients, median = {bmi_values.median():.2f}")
print()

# ============================================================================
# STEP 4: Create Feature Matrix
# ============================================================================
print("STEP 4: Creating Feature Matrix for Model v0.2.0")
print("-" * 80)

# Model v0.2.0 uses: Glucose, Age, BMI, Lactate, LDH, CRP
feature_names = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH', 'CRP']

# Create DataFrame with all patients
feature_df = pd.DataFrame(index=patients['subject_id'])

# Add age
feature_df['Age'] = patients.set_index('subject_id')['anchor_age']

# Add biomarkers
for biomarker in ['Glucose', 'BMI', 'Lactate', 'LDH', 'CRP']:
    if biomarker in biomarker_data:
        feature_df[biomarker] = biomarker_data[biomarker]

# Add cancer label
feature_df['cancer'] = patients.set_index('subject_id')['cancer']

print(f"✓ Feature matrix shape: {feature_df.shape}")
print(f"\nFeature completeness:")
for col in feature_names:
    pct = (feature_df[col].notna().sum() / len(feature_df)) * 100
    print(f"  {col}: {feature_df[col].notna().sum()}/{len(feature_df)} ({pct:.1f}%)")

print(f"\nMissing data summary:")
print(feature_df[feature_names].isnull().sum())
print()

# ============================================================================
# STEP 5: Handle Missing Values
# ============================================================================
print("STEP 5: Handling Missing Values")
print("-" * 80)

# Strategy: Use median imputation for missing values
# This is common practice in clinical ML
imputed_df = feature_df.copy()

for col in feature_names:
    if imputed_df[col].isnull().any():
        median_val = imputed_df[col].median()
        n_missing = imputed_df[col].isnull().sum()
        imputed_df[col] = imputed_df[col].fillna(median_val)
        print(f"✓ Imputed {n_missing} missing {col} values with median {median_val:.2f}")

print()

# Check data distributions
print("Feature Statistics (after imputation):")
print(imputed_df[feature_names].describe())
print()

# ============================================================================
# STEP 6: Load Model and Make Predictions
# ============================================================================
print("STEP 6: Loading Model and Making Predictions")
print("-" * 80)

# Load the trained model
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
model_features = model_data['features']
version = model_data['version']

print(f"✓ Loaded model version: {version}")
print(f"✓ Model features: {model_features}")
print()

# Prepare X and y
X = imputed_df[feature_names].values
y_true = imputed_df['cancer'].values

# Make predictions
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

print(f"✓ Generated predictions for {len(y_pred)} patients")
print()

# ============================================================================
# STEP 7: Calculate Performance Metrics
# ============================================================================
print("STEP 7: Performance Metrics")
print("=" * 80)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
f1 = f1_score(y_true, y_pred, zero_division=0)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Specificity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# ROC AUC
try:
    roc_auc = roc_auc_score(y_true, y_pred_proba)
except:
    roc_auc = 0

print(f"\n{'Metric':<20} {'Value':<10}")
print("-" * 30)
print(f"{'Accuracy':<20} {accuracy*100:>6.2f}%")
print(f"{'Precision':<20} {precision*100:>6.2f}%")
print(f"{'Recall/Sensitivity':<20} {recall*100:>6.2f}%")
print(f"{'Specificity':<20} {specificity*100:>6.2f}%")
print(f"{'F1 Score':<20} {f1:>6.2f}")
print(f"{'ROC AUC':<20} {roc_auc:>6.2f}")
print()

print("Confusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")
print()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Control', 'Cancer']))
print()

# ============================================================================
# STEP 8: Generate Visualizations
# ============================================================================
print("STEP 8: Generating Visualizations")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Confusion Matrix
ax = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Control', 'Cancer'],
            yticklabels=['Control', 'Cancer'])
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# 2. ROC Curve
ax = axes[0, 1]
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Prediction Distribution
ax = axes[1, 0]
cancer_probs = y_pred_proba[y_true == 1]
control_probs = y_pred_proba[y_true == 0]
ax.hist(control_probs, bins=20, alpha=0.6, label='Control', color='blue')
ax.hist(cancer_probs, bins=20, alpha=0.6, label='Cancer', color='red')
ax.axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
ax.set_xlabel('Predicted Probability of Cancer')
ax.set_ylabel('Number of Patients')
ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Performance Metrics Bar Chart
ax = axes[1, 1]
metrics = {
    'Accuracy': accuracy,
    'Sensitivity': recall,
    'Specificity': specificity,
    'Precision': precision,
    'F1 Score': f1,
    'ROC AUC': roc_auc
}
bars = ax.bar(range(len(metrics)), list(metrics.values()),
              color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c'])
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
ax.set_ylabel('Score')
ax.set_ylim([0, 1.1])
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (metric, value) in enumerate(metrics.items()):
    ax.text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('external_datasets/mimic_iv_demo/mimic_test_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: mimic_test_results.png")
print()

# ============================================================================
# STEP 9: Feature Importance Analysis
# ============================================================================
print("STEP 9: Feature Importance Analysis")
print("-" * 80)

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for _, row in feature_importance.iterrows():
    print(f"  {row['Feature']:<15} {row['Importance']:>6.4f} {'█' * int(row['Importance'] * 100)}")
print()

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Dataset: MIMIC-IV Demo (100 patients)")
print(f"  - Cancer patients: {y_true.sum()}")
print(f"  - Control patients: {len(y_true) - y_true.sum()}")
print()
print(f"Model: Cancer Prediction v{version}")
print(f"  - Features: {', '.join(feature_names)}")
print()
print(f"Performance:")
print(f"  - Overall Accuracy: {accuracy*100:.1f}%")
print(f"  - Sensitivity (Cancer Detection): {recall*100:.1f}%")
print(f"  - Specificity (Control Identification): {specificity*100:.1f}%")
print(f"  - ROC AUC: {roc_auc:.3f}")
print()

# Compare to UCI results
print("Comparison to UCI Breast Cancer Test:")
print("  - UCI Accuracy: 55%")
print(f"  - MIMIC-IV Accuracy: {accuracy*100:.1f}%")
if accuracy > 0.55:
    print(f"  ✅ MIMIC-IV performance is {(accuracy-0.55)*100:.1f} percentage points BETTER")
elif accuracy < 0.55:
    print(f"  ⚠️  MIMIC-IV performance is {(0.55-accuracy)*100:.1f} percentage points WORSE")
else:
    print("  ↔️  Performance is equivalent")
print()

# Limitations
print("Limitations:")
print("  - Small sample size (100 patients)")
print(f"  - CRP missing in {(1 - (imputed_df['CRP'].notna().sum() / len(imputed_df)))*100:.0f}% of patients (imputed)")
print("  - Mixed cancer types (not specific to one cancer)")
print("  - Demo dataset (not representative of full MIMIC-IV)")
print()

print("="*80)
print("✅ TESTING COMPLETE!")
print("="*80)

# Save results to CSV
results_df = imputed_df.copy()
results_df['predicted_cancer'] = y_pred
results_df['predicted_probability'] = y_pred_proba
results_df.to_csv('external_datasets/mimic_iv_demo/mimic_predictions.csv')
print("\n✓ Saved predictions to: mimic_predictions.csv")
