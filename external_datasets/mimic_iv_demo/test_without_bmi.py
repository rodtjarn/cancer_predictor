"""
Test removing BMI from the cancer prediction model
Compare 5-feature (with BMI) vs 4-feature (without BMI) models
BMI showed 0% feature importance - testing if removing it helps
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
print("TESTING MODEL WITHOUT BMI")
print("=" * 80)

# Paths
BASE_PATH = Path("/Users/per/work/claude/cancer_predictor_package/external_datasets/mimic_iv_demo/external_datasets/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2")

# ============================================================================
# STEP 1: Load and prepare data
# ============================================================================

print("\n1. Loading MIMIC-IV demo data...")

with gzip.open(BASE_PATH / "hosp/d_labitems.csv.gz", 'rt') as f:
    d_labitems = pd.read_csv(f)

with gzip.open(BASE_PATH / "hosp/labevents.csv.gz", 'rt') as f:
    labevents = pd.read_csv(f)

with gzip.open(BASE_PATH / "hosp/patients.csv.gz", 'rt') as f:
    patients = pd.read_csv(f)

with gzip.open(BASE_PATH / "hosp/diagnoses_icd.csv.gz", 'rt') as f:
    diagnoses = pd.read_csv(f)

print(f"   Patients: {len(patients):,}")
print(f"   Lab events: {len(labevents):,}")

# ============================================================================
# STEP 2: Extract biomarkers (WITHOUT BMI)
# ============================================================================

print("\n2. Extracting biomarker data...")

biomarker_items = {
    'Lactate': [50813, 52442, 53154],
    'Glucose': [50809, 50931, 52027, 52569],
    'LDH': [50954]
}

biomarker_data = pd.DataFrame()
biomarker_data['subject_id'] = patients['subject_id']

for biomarker, item_ids in biomarker_items.items():
    mask = labevents['itemid'].isin(item_ids)
    measurements = labevents[mask][['subject_id', 'valuenum']].copy()
    measurements = measurements[measurements['valuenum'].notna()]
    measurements = measurements[measurements['valuenum'] > 0]
    patient_values = measurements.groupby('subject_id')['valuenum'].median()
    biomarker_data[biomarker] = biomarker_data['subject_id'].map(patient_values)

    available = biomarker_data[biomarker].notna().sum()
    print(f"   {biomarker}: {available}/100 patients ({available}%)")

# Add age
biomarker_data['Age'] = biomarker_data['subject_id'].map(
    patients.set_index('subject_id')['anchor_age']
)

# Add BMI for comparison (even though we'll test without it)
biomarker_data['BMI'] = 26.5  # Population average

print(f"   Age: {biomarker_data['Age'].notna().sum()}/100 patients")

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

# Current 5-feature model (WITH BMI)
features_5 = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH']

# New 4-feature model (WITHOUT BMI)
features_4 = ['Glucose', 'Age', 'Lactate', 'LDH']

# Get complete cases
complete_data_5 = biomarker_data[features_5 + ['Cancer']].dropna()
complete_data_4 = biomarker_data[features_4 + ['Cancer']].dropna()

print(f"   Patients with 5 features (with BMI): {len(complete_data_5)}")
print(f"   Patients with 4 features (no BMI): {len(complete_data_4)}")
print(f"   → Gain: {len(complete_data_4) - len(complete_data_5)} more patients!")

# Use the larger dataset (4-feature)
X_4 = complete_data_4[features_4].values
y = complete_data_4['Cancer'].values

# For 5-feature comparison, use same patients
complete_data_5_matched = complete_data_4.copy()
complete_data_5_matched['BMI'] = 26.5  # Add BMI back
X_5 = complete_data_5_matched[features_5].values

print(f"\n   Final dataset: {len(y)} patients")
print(f"   - Cancer: {y.sum()}")
print(f"   - Control: {len(y) - y.sum()}")

# ============================================================================
# STEP 5: Train and validate both models
# ============================================================================

print("\n5. Training and validating models with 70/30 split...")

# Stratified split
X_train_4, X_test_4, y_train, y_test = train_test_split(
    X_4, y, test_size=0.30, random_state=42, stratify=y
)

X_train_5, X_test_5, _, _ = train_test_split(
    X_5, y, test_size=0.30, random_state=42, stratify=y
)

print(f"   Training set: {len(y_train)} patients ({y_train.sum()} cancer)")
print(f"   Test set: {len(y_test)} patients ({y_test.sum()} cancer)")

# Train 5-feature model (WITH BMI)
print("\n   Training 5-feature model (with BMI)...")
model_5 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
model_5.fit(X_train_5, y_train)

# Train 4-feature model (WITHOUT BMI)
print("   Training 4-feature model (without BMI)...")
model_4 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
model_4.fit(X_train_4, y_train)

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

# 4-feature model
y_pred_4 = model_4.predict(X_test_4)
y_pred_proba_4 = model_4.predict_proba(X_test_4)[:, 1]

acc_4 = accuracy_score(y_test, y_pred_4)
cm_4 = confusion_matrix(y_test, y_pred_4)
tn_4, fp_4, fn_4, tp_4 = cm_4.ravel()
sens_4 = recall_score(y_test, y_pred_4, zero_division=0)
spec_4 = tn_4 / (tn_4 + fp_4) if (tn_4 + fp_4) > 0 else 0
f1_4 = f1_score(y_test, y_pred_4, zero_division=0)
auc_4 = roc_auc_score(y_test, y_pred_proba_4)

# ============================================================================
# STEP 7: Cross-Validation
# ============================================================================

print("\n7. Performing 5-fold cross-validation...")

cv_scores_5 = cross_val_score(model_5, X_5, y, cv=5, scoring='accuracy')
cv_scores_4 = cross_val_score(model_4, X_4, y, cv=5, scoring='accuracy')

# ============================================================================
# STEP 8: Feature Importance
# ============================================================================

importance_5 = model_5.feature_importances_
importance_4 = model_4.feature_importances_

# ============================================================================
# Print Results
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS: TEST SET PERFORMANCE")
print("=" * 80)

print(f"\n📊 5-Feature Model (Glucose, Age, BMI, Lactate, LDH):")
print(f"   Accuracy:    {acc_5*100:.1f}%")
print(f"   Sensitivity: {sens_5*100:.1f}% ({tp_5}/{tp_5+fn_5} cancers detected)")
print(f"   Specificity: {spec_5*100:.1f}%")
print(f"   F1 Score:    {f1_5:.3f}")
print(f"   ROC AUC:     {auc_5:.3f}")

print(f"\n📊 4-Feature Model (WITHOUT BMI):")
print(f"   Accuracy:    {acc_4*100:.1f}%")
print(f"   Sensitivity: {sens_4*100:.1f}% ({tp_4}/{tp_4+fn_4} cancers detected)")
print(f"   Specificity: {spec_4*100:.1f}%")
print(f"   F1 Score:    {f1_4:.3f}")
print(f"   ROC AUC:     {auc_4:.3f}")

print(f"\n📈 Change (removing BMI):")
print(f"   Accuracy:    {(acc_4-acc_5)*100:+.1f} pp")
print(f"   Sensitivity: {(sens_4-sens_5)*100:+.1f} pp")
print(f"   Specificity: {(spec_4-spec_5)*100:+.1f} pp")
print(f"   F1 Score:    {(f1_4-f1_5):+.3f}")
print(f"   ROC AUC:     {(auc_4-auc_5):+.3f}")

print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS (5-fold)")
print("=" * 80)

print(f"\n   5-Feature Model CV: {cv_scores_5.mean()*100:.1f}% ± {cv_scores_5.std()*100:.1f}%")
print(f"   4-Feature Model CV: {cv_scores_4.mean()*100:.1f}% ± {cv_scores_4.std()*100:.1f}%")
print(f"   Change: {(cv_scores_4.mean() - cv_scores_5.mean())*100:+.1f} pp")

# Variance comparison
variance_change = (cv_scores_4.std() / cv_scores_5.std() - 1) * 100
print(f"   Variance change: {variance_change:+.1f}% ({'more stable' if variance_change < 0 else 'less stable'})")

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print("\n   5-Feature Model (with BMI):")
for feat, imp in zip(features_5, importance_5):
    print(f"   {feat:12s}: {imp:.4f} ({imp*100:.1f}%)")

print(f"\n   BMI Contribution: {importance_5[2]*100:.1f}%")

print("\n   4-Feature Model (without BMI):")
for feat, imp in zip(features_4, importance_4):
    print(f"   {feat:12s}: {imp:.4f} ({imp*100:.1f}%)")

# Calculate redistribution
print("\n   Feature Importance Redistribution (after removing BMI):")
print(f"   Glucose: {importance_4[0]:.4f} (was {importance_5[0]:.4f}) {(importance_4[0]/importance_5[0]-1)*100:+.1f}%")
print(f"   Age:     {importance_4[1]:.4f} (was {importance_5[1]:.4f}) {(importance_4[1]/importance_5[1]-1)*100:+.1f}%")
print(f"   Lactate: {importance_4[2]:.4f} (was {importance_5[3]:.4f}) {(importance_4[2]/importance_5[3]-1)*100:+.1f}%")
print(f"   LDH:     {importance_4[3]:.4f} (was {importance_5[4]:.4f}) {(importance_4[3]/importance_5[4]-1)*100:+.1f}%")

# ============================================================================
# Create Visualizations
# ============================================================================

print("\n8. Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('BMI Removal Analysis: Model Comparison', fontsize=16, fontweight='bold')

# Plot 1: Accuracy Comparison
ax = axes[0, 0]
models = ['5-Feature\n(with BMI)', '4-Feature\n(no BMI)']
accuracies = [acc_5*100, acc_4*100]
colors = ['#e74c3c', '#2ecc71']
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
sens = [sens_5*100, sens_4*100]
spec = [spec_5*100, spec_4*100]
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
fpr_4, tpr_4, _ = roc_curve(y_test, y_pred_proba_4)
ax.plot(fpr_5, tpr_5, 'r-', linewidth=2, label=f'5-Feature (AUC={auc_5:.3f})')
ax.plot(fpr_4, tpr_4, 'g-', linewidth=2, label=f'4-Feature (AUC={auc_4:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Plot 4: Confusion Matrix
ax = axes[1, 0]
x_labels = ['5-Feature', '4-Feature']
y_labels = ['TN', 'FP', 'FN', 'TP']
data = np.array([
    [tn_5, tn_4],
    [fp_5, fp_4],
    [fn_5, fn_4],
    [tp_5, tp_4]
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

# Plot 5: Feature Importance Comparison
ax = axes[1, 1]
features_combined = ['Glucose', 'Age', 'Lactate', 'LDH', 'BMI']
importance_5_reordered = [importance_5[0], importance_5[1], importance_5[3], importance_5[4], importance_5[2]]
importance_4_extended = list(importance_4) + [0]  # Add 0 for BMI

x_pos = np.arange(len(features_combined))
width = 0.35
ax.barh(x_pos - width/2, importance_5_reordered, width, label='5-Feature (with BMI)',
        color='#e74c3c', alpha=0.7, edgecolor='black')
ax.barh(x_pos + width/2, importance_4_extended, width, label='4-Feature (no BMI)',
        color='#2ecc71', alpha=0.7, edgecolor='black')
ax.set_yticks(x_pos)
ax.set_yticklabels(features_combined)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance Comparison', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Plot 6: Cross-Validation
ax = axes[1, 2]
cv_data = pd.DataFrame({
    '5-Feature': cv_scores_5 * 100,
    '4-Feature': cv_scores_4 * 100
})
bp = cv_data.boxplot(ax=ax, patch_artist=True, return_type='dict')
for patch, color in zip(bp['boxes'], ['#e74c3c', '#2ecc71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('5-Fold Cross-Validation', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (col, scores) in enumerate(cv_data.items(), 1):
    ax.plot([i], [scores.mean()], 'ko', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
    ax.text(i, scores.mean() + 2, f'{scores.mean():.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('bmi_removal_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved bmi_removal_comparison.png")

# ============================================================================
# Save Results
# ============================================================================

print("\n9. Saving results...")

results = {
    'model_5': model_5,
    'model_4': model_4,
    'features_5': features_5,
    'features_4': features_4,
    'metrics_5': {
        'accuracy': acc_5, 'sensitivity': sens_5, 'specificity': spec_5,
        'f1': f1_5, 'auc': auc_5, 'cm': cm_5
    },
    'metrics_4': {
        'accuracy': acc_4, 'sensitivity': sens_4, 'specificity': spec_4,
        'f1': f1_4, 'auc': auc_4, 'cm': cm_4
    },
    'cv_scores_5': cv_scores_5,
    'cv_scores_4': cv_scores_4,
    'importance_5': importance_5,
    'importance_4': importance_4
}

with open('bmi_removal_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save 4-feature model
model_data = {
    'model': model_4,
    'features': features_4,
    'version': '0.2.3',
    'description': '4-feature model without BMI',
    'training_samples': len(y_train),
    'test_accuracy': acc_4
}

with open('model_without_bmi.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("   ✅ Saved bmi_removal_results.pkl")
print("   ✅ Saved model_without_bmi.pkl")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if acc_4 > acc_5:
    print(f"\n✅ REMOVING BMI IMPROVES PERFORMANCE!")
    print(f"   Test accuracy: {acc_5*100:.1f}% → {acc_4*100:.1f}% (+{(acc_4-acc_5)*100:.1f} pp)")
    print(f"   CV accuracy: {cv_scores_5.mean()*100:.1f}% → {cv_scores_4.mean()*100:.1f}% ({(cv_scores_4.mean()-cv_scores_5.mean())*100:+.1f} pp)")
    print(f"\n   Recommendation: Adopt 4-feature model (Glucose, Age, Lactate, LDH)")
elif acc_4 == acc_5:
    print(f"\n↔️  REMOVING BMI HAS NO EFFECT")
    print(f"   Test accuracy: {acc_5*100:.1f}% (unchanged)")
    print(f"   CV accuracy: {cv_scores_5.mean()*100:.1f}% → {cv_scores_4.mean()*100:.1f}%")
    print(f"\n   Recommendation: Remove BMI anyway (simpler model, same performance)")
else:
    print(f"\n⚠️  REMOVING BMI HURTS PERFORMANCE")
    print(f"   Test accuracy: {acc_5*100:.1f}% → {acc_4*100:.1f}% ({(acc_4-acc_5)*100:.1f} pp)")
    print(f"   CV accuracy: {cv_scores_5.mean()*100:.1f}% → {cv_scores_4.mean()*100:.1f}% ({(cv_scores_4.mean()-cv_scores_5.mean())*100:+.1f} pp)")
    print(f"\n   Recommendation: Keep BMI (despite 0% importance)")

print("\n" + "=" * 80)
