"""
Analyze MIMIC-IV Demo by Cancer Type
Test the metabolic theory hypothesis: if cancer is a single metabolic disease,
performance should be similar across all cancer types with consistent biomarker patterns.

This analysis will:
1. Group cancer patients by cancer type
2. Analyze biomarker distributions by cancer type
3. Test model performance on each cancer type
4. Compare feature importance patterns across types
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score

# Base path
BASE_PATH = Path("/Users/per/work/claude/cancer_predictor_package/external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2")

def read_gz_csv(filename):
    """Read a gzipped CSV file"""
    with gzip.open(filename, 'rt') as f:
        return pd.read_csv(f)

print("="*80)
print("MIMIC-IV: CANCER TYPE ANALYSIS")
print("Testing Metabolic Theory: Cancer as Single Metabolic Disease")
print("="*80)
print()

# Load data
print("Loading MIMIC-IV demo data...")
patients = read_gz_csv(BASE_PATH / "hosp/patients.csv.gz")
admissions = read_gz_csv(BASE_PATH / "hosp/admissions.csv.gz")
diagnoses = read_gz_csv(BASE_PATH / "hosp/diagnoses_icd.csv.gz")
d_icd = read_gz_csv(BASE_PATH / "hosp/d_icd_diagnoses.csv.gz")
labevents = read_gz_csv(BASE_PATH / "hosp/labevents.csv.gz")
d_labitems = read_gz_csv(BASE_PATH / "hosp/d_labitems.csv.gz")

print(f"Patients: {len(patients)}")
print(f"Admissions: {len(admissions)}")
print(f"Diagnoses: {len(diagnoses)}")
print(f"Lab events: {len(labevents)}")
print()

# Define cancer ICD-10 codes by category
cancer_categories = {
    'Lung Cancer': ['C34'],  # Malignant neoplasm of bronchus and lung
    'GI Cancer': ['C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'],  # Digestive organs
    'Breast Cancer': ['C50'],  # Malignant neoplasm of breast
    'Prostate Cancer': ['C61'],  # Malignant neoplasm of prostate
    'Hematologic Cancer': ['C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C90', 'C91', 'C92', 'C93', 'C94', 'C95'],  # Lymphomas, leukemias, myelomas
    'Urologic Cancer': ['C64', 'C65', 'C66', 'C67', 'C68'],  # Kidney, bladder, urinary organs
    'Gynecologic Cancer': ['C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58'],  # Female genital organs
    'Head/Neck Cancer': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C30', 'C31', 'C32'],  # Oral cavity, pharynx, larynx
    'Other Cancer': []  # Will be filled with any other C codes
}

# Identify cancer patients and categorize them
print("Identifying and categorizing cancer patients...")
print("-" * 80)

# Merge diagnoses with ICD descriptions
diagnoses_full = diagnoses.merge(d_icd, on=['icd_code', 'icd_version'], how='left')

# Get cancer diagnoses (ICD-10 codes starting with C)
cancer_dx = diagnoses_full[
    (diagnoses_full['icd_version'] == 10) &
    (diagnoses_full['icd_code'].str.startswith('C', na=False))
].copy()

print(f"Total cancer diagnoses: {len(cancer_dx)}")
print(f"Unique cancer patients: {cancer_dx['subject_id'].nunique()}")
print()

# Categorize each cancer diagnosis
def categorize_cancer(icd_code):
    """Categorize cancer by ICD-10 code"""
    for category, prefixes in cancer_categories.items():
        if category == 'Other Cancer':
            continue
        for prefix in prefixes:
            if icd_code.startswith(prefix):
                return category
    return 'Other Cancer'

cancer_dx['cancer_category'] = cancer_dx['icd_code'].apply(categorize_cancer)

# Get primary cancer type per patient (most common category)
patient_cancer_types = cancer_dx.groupby('subject_id').agg({
    'cancer_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Other Cancer',
    'long_title': lambda x: ', '.join(x.unique()[:3])  # Top 3 diagnoses
}).reset_index()

patient_cancer_types.columns = ['subject_id', 'cancer_type', 'diagnoses']

print("Cancer patients by type:")
print(patient_cancer_types['cancer_type'].value_counts())
print()

# Get biomarker data
print("Extracting biomarker data...")
print("-" * 80)

# Define biomarker item IDs (from previous analysis)
biomarker_itemids = {
    'Glucose': [50809, 50931],  # Glucose blood
    'Lactate': [50813],  # Lactate blood
    'LDH': [50954],  # LDH
    'CRP': [50889],  # CRP
}

# Extract biomarkers for each patient
biomarker_data = []

for subject_id in patients['subject_id']:
    # Get patient age
    patient_info = patients[patients['subject_id'] == subject_id].iloc[0]
    anchor_age = patient_info['anchor_age']

    # Get lab values for this patient
    patient_labs = labevents[labevents['subject_id'] == subject_id]

    biomarkers = {'subject_id': subject_id, 'Age': anchor_age}

    for biomarker, itemids in biomarker_itemids.items():
        values = patient_labs[patient_labs['itemid'].isin(itemids)]['valuenum'].dropna()
        if len(values) > 0:
            biomarkers[biomarker] = values.median()  # Use median if multiple measurements
        else:
            biomarkers[biomarker] = np.nan

    biomarker_data.append(biomarkers)

biomarker_df = pd.DataFrame(biomarker_data)

# Merge with cancer types
biomarker_df = biomarker_df.merge(patient_cancer_types[['subject_id', 'cancer_type', 'diagnoses']],
                                   on='subject_id', how='left')
biomarker_df['has_cancer'] = biomarker_df['cancer_type'].notna().astype(int)
biomarker_df['cancer_type'] = biomarker_df['cancer_type'].fillna('Control')

print(f"Total patients with biomarker data: {len(biomarker_df)}")
print(f"Cancer patients: {biomarker_df['has_cancer'].sum()}")
print(f"Control patients: {(biomarker_df['has_cancer'] == 0).sum()}")
print()

print("Biomarker availability by cancer type:")
for cancer_type in biomarker_df['cancer_type'].unique():
    subset = biomarker_df[biomarker_df['cancer_type'] == cancer_type]
    print(f"\n{cancer_type} (n={len(subset)}):")
    for biomarker in ['Glucose', 'Age', 'Lactate', 'LDH', 'CRP']:
        coverage = (subset[biomarker].notna().sum() / len(subset)) * 100
        print(f"  {biomarker}: {coverage:.1f}% coverage")

# Save raw data
biomarker_df.to_csv('biomarker_by_cancer_type.csv', index=False)
print("\nSaved: biomarker_by_cancer_type.csv")

# Analysis: Biomarker patterns by cancer type
print("\n" + "="*80)
print("BIOMARKER PATTERNS BY CANCER TYPE")
print("="*80)

# Get complete cases (all 4 biomarkers)
complete_biomarkers = ['Glucose', 'Age', 'Lactate', 'LDH']
biomarker_complete = biomarker_df.dropna(subset=complete_biomarkers).copy()

print(f"\nPatients with complete biomarker data (Glucose, Age, Lactate, LDH): {len(biomarker_complete)}")
print(f"Cancer patients: {biomarker_complete['has_cancer'].sum()}")
print(f"Control patients: {(biomarker_complete['has_cancer'] == 0).sum()}")
print()

# Biomarker statistics by cancer type
print("Biomarker statistics by cancer type:")
print("-" * 80)

for biomarker in complete_biomarkers:
    print(f"\n{biomarker}:")
    print(f"{'Cancer Type':<20} {'N':>5} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print("-" * 60)

    # Control group
    control = biomarker_complete[biomarker_complete['cancer_type'] == 'Control'][biomarker]
    print(f"{'Control':<20} {len(control):>5} {control.mean():>10.2f} {control.median():>10.2f} {control.std():>10.2f}")

    # Each cancer type
    cancer_types = biomarker_complete[biomarker_complete['cancer_type'] != 'Control']['cancer_type'].unique()
    for cancer_type in sorted(cancer_types):
        subset = biomarker_complete[biomarker_complete['cancer_type'] == cancer_type][biomarker]
        if len(subset) >= 3:  # Only show if at least 3 patients
            print(f"{cancer_type:<20} {len(subset):>5} {subset.mean():>10.2f} {subset.median():>10.2f} {subset.std():>10.2f}")

# Train a new model on the complete data (for overall baseline)
print("\n" + "="*80)
print("MODEL PERFORMANCE BY CANCER TYPE")
print("="*80)

print("\nTraining 4-biomarker model (Glucose, Age, Lactate, LDH) on complete data")
print()

# Prepare data
X = biomarker_complete[complete_biomarkers].values
y = biomarker_complete['has_cancer'].values
cancer_types = biomarker_complete['cancer_type'].values

# Train baseline model on all data
baseline_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
baseline_model.fit(X, y)

# Overall performance (training set - for reference only)
y_pred = baseline_model.predict(X)
y_proba = baseline_model.predict_proba(X)[:, 1]

print(f"Overall Performance (n={len(y)}, training set):")
print(f"  Accuracy: {accuracy_score(y, y_pred):.1%}")
print(f"  ROC AUC: {roc_auc_score(y, y_proba):.3f}")
print(f"  Note: This is training set performance, not validated")
print()

# Performance by cancer type (using Leave-One-Out for small samples)
print("Performance by Cancer Type (Leave-One-Out Cross-Validation):")
print("-" * 80)
print(f"{'Cancer Type':<20} {'N':>5} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12}")
print("-" * 80)

cancer_type_results = []

for cancer_type in sorted(biomarker_complete[biomarker_complete['has_cancer'] == 1]['cancer_type'].unique()):
    # Get this cancer type + all controls
    cancer_mask = biomarker_complete['cancer_type'] == cancer_type
    control_mask = biomarker_complete['cancer_type'] == 'Control'
    subset_mask = cancer_mask | control_mask

    X_subset = biomarker_complete[subset_mask][complete_biomarkers].values
    y_subset = biomarker_complete[subset_mask]['has_cancer'].values

    n_cancer = cancer_mask.sum()
    n_total = subset_mask.sum()

    if n_cancer >= 3:  # Only test if at least 3 cancer patients
        # Leave-One-Out CV
        loo = LeaveOneOut()
        y_pred_loo = []

        for train_idx, test_idx in loo.split(X_subset):
            X_train, X_test = X_subset[train_idx], X_subset[test_idx]
            y_train, y_test = y_subset[train_idx], y_subset[test_idx]

            # Train model
            loo_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            loo_model.fit(X_train, y_train)

            # Predict
            y_pred_loo.append(loo_model.predict(X_test)[0])

        y_pred_loo = np.array(y_pred_loo)

        # Calculate metrics
        accuracy = accuracy_score(y_subset, y_pred_loo)

        # Sensitivity (True Positive Rate)
        cancer_indices = np.where(y_subset == 1)[0]
        sensitivity = (y_pred_loo[cancer_indices] == 1).sum() / len(cancer_indices) if len(cancer_indices) > 0 else 0

        # Specificity (True Negative Rate)
        control_indices = np.where(y_subset == 0)[0]
        specificity = (y_pred_loo[control_indices] == 0).sum() / len(control_indices) if len(control_indices) > 0 else 0

        print(f"{cancer_type:<20} {n_cancer:>5} {accuracy:>9.1%} {sensitivity:>11.1%} {specificity:>11.1%}")

        cancer_type_results.append({
            'cancer_type': cancer_type,
            'n_cancer': n_cancer,
            'n_total': n_total,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        })

# Feature importance by cancer type
print("\n" + "="*80)
print("FEATURE IMPORTANCE BY CANCER TYPE")
print("="*80)
print("Testing if biomarker importance is consistent across cancer types")
print()

importance_by_type = {}

for cancer_type in sorted(biomarker_complete[biomarker_complete['has_cancer'] == 1]['cancer_type'].unique()):
    # Get this cancer type + all controls
    cancer_mask = biomarker_complete['cancer_type'] == cancer_type
    control_mask = biomarker_complete['cancer_type'] == 'Control'
    subset_mask = cancer_mask | control_mask

    X_subset = biomarker_complete[subset_mask][complete_biomarkers].values
    y_subset = biomarker_complete[subset_mask]['has_cancer'].values

    n_cancer = cancer_mask.sum()

    if n_cancer >= 3:
        # Train model on this cancer type
        type_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        type_model.fit(X_subset, y_subset)

        # Get feature importance
        importances = type_model.feature_importances_
        importance_by_type[cancer_type] = dict(zip(complete_biomarkers, importances * 100))

        print(f"{cancer_type} (n={n_cancer}):")
        for feat, imp in sorted(zip(complete_biomarkers, importances * 100), key=lambda x: x[1], reverse=True):
            print(f"  {feat:<12} {imp:>6.1f}%")
        print()

# Create visualization
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cancer Type Analysis: Testing Single Metabolic Disease Hypothesis', fontsize=16, fontweight='bold')

# 1. Biomarker distributions by cancer type
ax = axes[0, 0]
biomarker_long = []
for biomarker in ['Glucose', 'Lactate', 'LDH']:
    for cancer_type in biomarker_complete['cancer_type'].unique():
        subset = biomarker_complete[biomarker_complete['cancer_type'] == cancer_type]
        if len(subset) >= 3:
            for value in subset[biomarker]:
                biomarker_long.append({
                    'Biomarker': biomarker,
                    'Cancer Type': cancer_type,
                    'Value': value
                })

biomarker_long_df = pd.DataFrame(biomarker_long)

# Normalize values for comparison
for biomarker in ['Glucose', 'Lactate', 'LDH']:
    mask = biomarker_long_df['Biomarker'] == biomarker
    biomarker_long_df.loc[mask, 'Normalized Value'] = (
        biomarker_long_df.loc[mask, 'Value'] - biomarker_long_df.loc[mask, 'Value'].mean()
    ) / biomarker_long_df.loc[mask, 'Value'].std()

sns.boxplot(data=biomarker_long_df, x='Cancer Type', y='Normalized Value', hue='Biomarker', ax=ax)
ax.set_title('Metabolic Biomarker Patterns Across Cancer Types', fontweight='bold')
ax.set_xlabel('Cancer Type')
ax.set_ylabel('Normalized Value (Z-score)')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Biomarker', loc='upper right')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# 2. Performance by cancer type
if len(cancer_type_results) > 0:
    ax = axes[0, 1]
    results_df = pd.DataFrame(cancer_type_results)
    x = np.arange(len(results_df))
    width = 0.25

    ax.bar(x - width, results_df['accuracy'], width, label='Accuracy', color='steelblue')
    ax.bar(x, results_df['sensitivity'], width, label='Sensitivity', color='coral')
    ax.bar(x + width, results_df['specificity'], width, label='Specificity', color='lightgreen')

    ax.set_xlabel('Cancer Type')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance by Cancer Type (LOO CV)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['cancer_type'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.3, label='70% threshold')
    ax.grid(axis='y', alpha=0.3)

# 3. Feature importance by cancer type
if len(importance_by_type) > 0:
    ax = axes[1, 0]
    importance_df = pd.DataFrame(importance_by_type).T
    importance_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Feature Importance Consistency Across Cancer Types', fontweight='bold')
    ax.set_xlabel('Cancer Type')
    ax.set_ylabel('Importance (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Biomarker', loc='upper right')
    ax.grid(axis='y', alpha=0.3)

# 4. Sample size and performance correlation
if len(cancer_type_results) > 0:
    ax = axes[1, 1]
    results_df = pd.DataFrame(cancer_type_results)

    scatter = ax.scatter(results_df['n_cancer'], results_df['accuracy'],
                        s=100, alpha=0.6, c=range(len(results_df)), cmap='viridis')

    for idx, row in results_df.iterrows():
        ax.annotate(row['cancer_type'],
                   (row['n_cancer'], row['accuracy']),
                   fontsize=8, ha='right')

    ax.set_xlabel('Number of Cancer Patients')
    ax.set_ylabel('Accuracy (LOO CV)')
    ax.set_title('Sample Size vs Performance', fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cancer_type_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: cancer_type_analysis.png")

# Save detailed results
results_summary = {
    'cancer_type_stats': biomarker_complete.groupby('cancer_type')[complete_biomarkers].describe().to_dict(),
    'cancer_type_performance': cancer_type_results,
    'feature_importance_by_type': importance_by_type,
    'overall_n': len(biomarker_complete),
    'cancer_n': biomarker_complete['has_cancer'].sum(),
    'control_n': (biomarker_complete['has_cancer'] == 0).sum()
}

with open('cancer_type_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
print("Saved: cancer_type_results.pkl")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
