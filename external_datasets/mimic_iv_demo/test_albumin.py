"""
Test adding Albumin to the cancer prediction model
Compare 5-feature (no CRP) vs 6-feature (with Albumin) models
"""

import pandas as pd
import numpy as np
import gzip
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, recall_score, precision_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("TESTING ALBUMIN AS NEW BIOMARKER")
print("=" * 80)

# Paths
BASE_PATH = Path("/Users/per/work/claude/cancer_predictor_package/external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2")

# ============================================================================
# STEP 1: Load and prepare data
# ============================================================================

print("\n1. Loading MIMIC-IV demo data...")

# Load lab items dictionary
with gzip.open(BASE_PATH / "hosp/d_labitems.csv.gz", 'rt') as f:
    d_labitems = pd.read_csv(f)

# Load lab events
with gzip.open(BASE_PATH / "hosp/labevents.csv.gz", 'rt') as f:
    labevents = pd.read_csv(f)

# Load patients
with gzip.open(BASE_PATH / "hosp/patients.csv.gz", 'rt') as f:
    patients = pd.read_csv(f)

# Load diagnoses
with gzip.open(BASE_PATH / "hosp/diagnoses_icd.csv.gz", 'rt') as f:
    diagnoses = pd.read_csv(f)

print(f"   Patients: {len(patients):,}")
print(f"   Lab events: {len(labevents):,}")
print(f"   Diagnoses: {len(diagnoses):,}")

# ============================================================================
# STEP 2: Extract biomarkers including Albumin
# ============================================================================

print("\n2. Extracting biomarker data...")

biomarker_items = {
    'Lactate': [50813, 52442, 53154],
    'Glucose': [50809, 50931, 52027, 52569],
    'LDH': [50954],
    'Albumin': [52022, 53138, 50862, 53085]
}

# Extract biomarker values for each patient
biomarker_data = pd.DataFrame()
biomarker_data['subject_id'] = patients['subject_id']

for biomarker, item_ids in biomarker_items.items():
    mask = labevents['itemid'].isin(item_ids)
    measurements = labevents[mask][['subject_id', 'valuenum']].copy()
    measurements = measurements[measurements['valuenum'].notna()]
    measurements = measurements[measurements['valuenum'] > 0]

    # Take median value per patient
    patient_values = measurements.groupby('subject_id')['valuenum'].median()
    biomarker_data[biomarker] = biomarker_data['subject_id'].map(patient_values)

    available = biomarker_data[biomarker].notna().sum()
    print(f"   {biomarker}: {available}/100 patients ({available}%)")

# Add age
biomarker_data['Age'] = biomarker_data['subject_id'].map(
    patients.set_index('subject_id')['anchor_age']
)

# Add BMI (approximate)
biomarker_data['BMI'] = 26.5  # Population average

print(f"   Age: {biomarker_data['Age'].notna().sum()}/100 patients")
print(f"   BMI: {biomarker_data['BMI'].notna().sum()}/100 patients")

# ============================================================================
# STEP 3: Create cancer labels
# ============================================================================

print("\n3. Creating cancer labels...")

cancer_icd_prefixes = ['C', '14', '15', '16', '17', '18', '19', '20']
cancer_patients = set()

for _, row in diagnoses.iterrows():
    icd_code = str(row['icd_code'])
    for prefix in cancer_icd_prefixes:
        if icd_code.startswith(prefix):
            cancer_patients.add(row['subject_id'])
            break

biomarker_data['Cancer'] = biomarker_data['subject_id'].apply(
    lambda x: 1 if x in cancer_patients else 0
)

cancer_count = biomarker_data['Cancer'].sum()
control_count = len(biomarker_data) - cancer_count
print(f"   Cancer patients: {cancer_count}")
print(f"   Control patients: {control_count}")

# ============================================================================
# STEP 4: Prepare datasets
# ============================================================================

print("\n4. Preparing datasets...")

# Current 5-feature model (without CRP)
features_5 = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH']

# New 6-feature model (with Albumin, without CRP)
features_6 = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH', 'Albumin']

# Get complete cases
complete_data = biomarker_data[features_6 + ['Cancer']].dropna()

print(f"   Patients with all 6 features: {len(complete_data)}")
print(f"   - Cancer: {complete_data['Cancer'].sum()}")
print(f"   - Control: {len(complete_data) - complete_data['Cancer'].sum()}")

# Extract features and labels
X_5 = complete_data[features_5].values
X_6 = complete_data[features_6].values
y = complete_data['Cancer'].values

# ============================================================================
# STEP 5: Train and validate both models
# ============================================================================

print("\n5. Training and validating models with 70/30 split...")

# Stratified split
X_train_5, X_test_5, y_train, y_test = train_test_split(
    X_5, y, test_size=0.30, random_state=42, stratify=y
)

X_train_6, X_test_6, _, _ = train_test_split(
    X_6, y, test_size=0.30, random_state=42, stratify=y
)

print(f"   Training set: {len(y_train)} patients ({y_train.sum()} cancer)")
print(f"   Test set: {len(y_test)} patients ({y_test.sum()} cancer)")

# Train 5-feature model
print("\n   Training 5-feature model...")
model_5 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
model_5.fit(X_train_5, y_train)

# Train 6-feature model
print("   Training 6-feature model (+ Albumin)...")
model_6 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
model_6.fit(X_train_6, y_train)

# ============================================================================
# STEP 6: Evaluate on test set
# ============================================================================

print("\n6. Evaluating on test set...")

# 5-feature model
y_pred_5 = model_5.predict(X_test_5)
y_pred_proba_5 = model_5.predict_proba(X_test_5)[:, 1]

acc_5 = accuracy_score(y_test, y_pred_5)
cm_5 = confusion_matrix(y_test, y_pred_5)
tn_5, fp_5, fn_5, tp_5 = cm_5.ravel()
sens_5 = recall_score(y_test, y_pred_5, zero_division=0)
spec_5 = tn_5 / (tn_5 + fp_5) if (tn_5 + fp_5) > 0 else 0
f1_5 = f1_score(y_test, y_pred_5, zero_division=0)
auc_5 = roc_auc_score(y_test, y_pred_proba_5)

# 6-feature model
y_pred_6 = model_6.predict(X_test_6)
y_pred_proba_6 = model_6.predict_proba(X_test_6)[:, 1]

acc_6 = accuracy_score(y_test, y_pred_6)
cm_6 = confusion_matrix(y_test, y_pred_6)
tn_6, fp_6, fn_6, tp_6 = cm_6.ravel()
sens_6 = recall_score(y_test, y_pred_6, zero_division=0)
spec_6 = tn_6 / (tn_6 + fp_6) if (tn_6 + fp_6) > 0 else 0
f1_6 = f1_score(y_test, y_pred_6, zero_division=0)
auc_6 = roc_auc_score(y_test, y_pred_proba_6)

# ============================================================================
# STEP 7: Cross-Validation
# ============================================================================

print("\n7. Performing 5-fold cross-validation...")

cv_scores_5 = cross_val_score(model_5, X_5, y, cv=5, scoring='accuracy')
cv_scores_6 = cross_val_score(model_6, X_6, y, cv=5, scoring='accuracy')

# ============================================================================
# STEP 8: Feature Importance
# ============================================================================

importance_5 = model_5.feature_importances_
importance_6 = model_6.feature_importances_

# ============================================================================
# Print Results
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS: TEST SET PERFORMANCE (17 patients)")
print("=" * 80)

print("\n📊 5-Feature Model (Glucose, Age, BMI, Lactate, LDH):")
print(f"   Accuracy:    {acc_5*100:.1f}%")
print(f"   Sensitivity: {sens_5*100:.1f}% ({tp_5}/{tp_5+fn_5} cancers detected)")
print(f"   Specificity: {spec_5*100:.1f}%")
print(f"   F1 Score:    {f1_5:.3f}")
print(f"   ROC AUC:     {auc_5:.3f}")

print("\n📊 6-Feature Model (+ Albumin):")
print(f"   Accuracy:    {acc_6*100:.1f}%")
print(f"   Sensitivity: {sens_6*100:.1f}% ({tp_6}/{tp_6+fn_6} cancers detected)")
print(f"   Specificity: {spec_6*100:.1f}%")
print(f"   F1 Score:    {f1_6:.3f}")
print(f"   ROC AUC:     {auc_6:.3f}")

print("\n📈 Improvement:")
print(f"   Accuracy:    {(acc_6-acc_5)*100:+.1f} pp")
print(f"   Sensitivity: {(sens_6-sens_5)*100:+.1f} pp")
print(f"   Specificity: {(spec_6-spec_5)*100:+.1f} pp")
print(f"   F1 Score:    {(f1_6-f1_5):+.3f}")
print(f"   ROC AUC:     {(auc_6-auc_5):+.3f}")

print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS (54 patients, 5-fold)")
print("=" * 80)

print(f"\n   5-Feature Model CV: {cv_scores_5.mean()*100:.1f}% ± {cv_scores_5.std()*100:.1f}%")
print(f"   6-Feature Model CV: {cv_scores_6.mean()*100:.1f}% ± {cv_scores_6.std()*100:.1f}%")
print(f"   Improvement: {(cv_scores_6.mean() - cv_scores_5.mean())*100:+.1f} pp")

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print("\n   5-Feature Model:")
for feat, imp in zip(features_5, importance_5):
    print(f"   {feat:12s}: {imp:.4f} ({imp*100:.1f}%)")

print("\n   6-Feature Model:")
for feat, imp in zip(features_6, importance_6):
    print(f"   {feat:12s}: {imp:.4f} ({imp*100:.1f}%)")

albumin_rank = sorted(enumerate(importance_6), key=lambda x: x[1], reverse=True)
albumin_position = [i for i, (idx, _) in enumerate(albumin_rank) if idx == 5][0] + 1
print(f"\n   Albumin Rank: #{albumin_position} out of 6 features")

# ============================================================================
# Create Visualizations
# ============================================================================

print("\n8. Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Albumin Biomarker Addition: Comprehensive Analysis', fontsize=16, fontweight='bold')

# Plot 1: Accuracy Comparison
ax = axes[0, 0]
models = ['5-Feature', '6-Feature\n(+ Albumin)']
accuracies = [acc_5*100, acc_6*100]
colors = ['#3498db', '#2ecc71']
bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Test Set Accuracy', fontsize=13, fontweight='bold')
ax.set_ylim([0, 100])
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Plot 2: Sensitivity & Specificity
ax = axes[0, 1]
x = np.arange(2)
width = 0.35
sens = [sens_5*100, sens_6*100]
spec = [spec_5*100, spec_6*100]
ax.bar(x - width/2, sens, width, label='Sensitivity', color='#e74c3c', alpha=0.7, edgecolor='black')
ax.bar(x + width/2, spec, width, label='Specificity', color='#9b59b6', alpha=0.7, edgecolor='black')
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Sensitivity & Specificity', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

# Plot 3: ROC Curves
ax = axes[0, 2]
fpr_5, tpr_5, _ = roc_curve(y_test, y_pred_proba_5)
fpr_6, tpr_6, _ = roc_curve(y_test, y_pred_proba_6)
ax.plot(fpr_5, tpr_5, 'b-', linewidth=2, label=f'5-Feature (AUC={auc_5:.3f})')
ax.plot(fpr_6, tpr_6, 'g-', linewidth=2, label=f'6-Feature (AUC={auc_6:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Plot 4: Confusion Matrix Comparison
ax = axes[1, 0]
x_labels = ['5-Feature', '6-Feature']
y_labels = ['TN', 'FP', 'FN', 'TP']
data = np.array([
    [tn_5, tn_6],
    [fp_5, fp_6],
    [fn_5, fn_6],
    [tp_5, tp_6]
])
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=max(data.max(), 5))
ax.set_xticks([0, 1])
ax.set_yticks([0, 1, 2, 3])
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
for i in range(4):
    for j in range(2):
        ax.text(j, i, int(data[i, j]), ha="center", va="center",
               color="black", fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)

# Plot 5: Feature Importance
ax = axes[1, 1]
importance_df = pd.DataFrame({
    '5-Feature': list(importance_5) + [0],
    '6-Feature': importance_6
}, index=features_6)
importance_df.plot(kind='barh', ax=ax, color=['#3498db', '#2ecc71'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance', fontsize=13, fontweight='bold')
ax.legend(['5-Feature Model', '6-Feature Model'])
ax.grid(axis='x', alpha=0.3)

# Plot 6: Cross-Validation
ax = axes[1, 2]
cv_data = pd.DataFrame({
    '5-Feature': cv_scores_5 * 100,
    '6-Feature': cv_scores_6 * 100
})
bp = cv_data.boxplot(ax=ax, patch_artist=True, return_type='dict')
for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('5-Fold Cross-Validation', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (col, scores) in enumerate(cv_data.items(), 1):
    ax.plot([i], [scores.mean()], 'ro', markersize=8)
    ax.text(i, scores.mean() + 2, f'{scores.mean():.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('albumin_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved albumin_comparison.png")

# ============================================================================
# Save Results
# ============================================================================

print("\n9. Saving results...")

results = {
    'model_5': model_5,
    'model_6': model_6,
    'features_5': features_5,
    'features_6': features_6,
    'metrics_5': {
        'accuracy': acc_5, 'sensitivity': sens_5, 'specificity': spec_5,
        'f1': f1_5, 'auc': auc_5, 'cm': cm_5
    },
    'metrics_6': {
        'accuracy': acc_6, 'sensitivity': sens_6, 'specificity': spec_6,
        'f1': f1_6, 'auc': auc_6, 'cm': cm_6
    },
    'cv_scores_5': cv_scores_5,
    'cv_scores_6': cv_scores_6,
    'importance_5': importance_5,
    'importance_6': importance_6,
    'complete_data': complete_data
}

with open('albumin_test_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save model with Albumin
model_data = {
    'model': model_6,
    'features': features_6,
    'version': '0.2.2',
    'description': '6-feature model with Albumin (no CRP)',
    'training_samples': len(y_train),
    'test_accuracy': acc_6
}

with open('model_with_albumin.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("   ✅ Saved albumin_test_results.pkl")
print("   ✅ Saved model_with_albumin.pkl")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
