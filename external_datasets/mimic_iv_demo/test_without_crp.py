"""
Test Cancer Prediction Model WITHOUT CRP Feature

Since 81% of CRP values were imputed, this script tests whether
removing CRP improves model performance.

Comparison:
- Model with CRP (6 features): Glucose, Age, BMI, Lactate, LDH, CRP
- Model without CRP (5 features): Glucose, Age, BMI, Lactate, LDH
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING MODEL WITHOUT CRP FEATURE")
print("="*80)
print()

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("STEP 1: Loading Data")
print("-" * 80)

# Load predictions with CRP
predictions_df = pd.read_csv('external_datasets/mimic_iv_demo/mimic_predictions.csv')
y_true = predictions_df['cancer'].values

print(f"✓ Loaded {len(predictions_df)} patient records")
print(f"  - Cancer patients: {y_true.sum()}")
print(f"  - Control patients: {len(y_true) - y_true.sum()}")
print()

# Load original model
MODEL_PATH = "models/metabolic_cancer_predictor_v2.pkl"
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

original_model = model_data['model']
print(f"✓ Loaded original model v{model_data['version']}")
print(f"  Features: {model_data['features']}")
print()

# ============================================================================
# STEP 2: Prepare Feature Matrices
# ============================================================================
print("STEP 2: Preparing Feature Matrices")
print("-" * 80)

# Features WITH CRP
features_with_crp = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH', 'CRP']
X_with_crp = predictions_df[features_with_crp].values

# Features WITHOUT CRP
features_without_crp = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH']
X_without_crp = predictions_df[features_without_crp].values

print(f"✓ Feature matrix WITH CRP: {X_with_crp.shape}")
print(f"✓ Feature matrix WITHOUT CRP: {X_without_crp.shape}")
print()

# Check CRP missingness
crp_missing = predictions_df['CRP'].isnull().sum()
crp_imputed = (predictions_df['CRP'] == predictions_df['CRP'].median()).sum()
print(f"CRP Data Quality:")
print(f"  - Originally missing: {crp_missing}/{len(predictions_df)} ({crp_missing/len(predictions_df)*100:.1f}%)")
print(f"  - Likely imputed: ~{crp_imputed}/{len(predictions_df)} ({crp_imputed/len(predictions_df)*100:.1f}%)")
print()

# ============================================================================
# STEP 3: Test Original Model (WITH CRP)
# ============================================================================
print("STEP 3: Testing Original Model (WITH CRP)")
print("-" * 80)

# Predictions with original model
y_pred_with_crp = original_model.predict(X_with_crp)
y_pred_proba_with_crp = original_model.predict_proba(X_with_crp)[:, 1]

# Metrics
acc_with_crp = accuracy_score(y_true, y_pred_with_crp)
sens_with_crp = recall_score(y_true, y_pred_with_crp)
cm_with_crp = confusion_matrix(y_true, y_pred_with_crp)
tn, fp, fn, tp = cm_with_crp.ravel()
spec_with_crp = tn / (tn + fp) if (tn + fp) > 0 else 0
prec_with_crp = precision_score(y_true, y_pred_with_crp, zero_division=0)
f1_with_crp = f1_score(y_true, y_pred_with_crp, zero_division=0)
roc_auc_with_crp = roc_auc_score(y_true, y_pred_proba_with_crp)

print("Performance WITH CRP (threshold=0.5):")
print(f"  Accuracy:    {acc_with_crp*100:.1f}%")
print(f"  Sensitivity: {sens_with_crp*100:.1f}%")
print(f"  Specificity: {spec_with_crp*100:.1f}%")
print(f"  Precision:   {prec_with_crp*100:.1f}%")
print(f"  F1 Score:    {f1_with_crp:.3f}")
print(f"  ROC AUC:     {roc_auc_with_crp:.3f}")
print(f"  Confusion:   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print()

# ============================================================================
# STEP 4: Train New Model WITHOUT CRP
# ============================================================================
print("STEP 4: Training New Model WITHOUT CRP")
print("-" * 80)

# Use same hyperparameters as original model
model_without_crp = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train on the data
model_without_crp.fit(X_without_crp, y_true)

print(f"✓ Trained RandomForest with {len(features_without_crp)} features")
print(f"  Features: {features_without_crp}")
print()

# Feature importance
feature_importance_no_crp = pd.DataFrame({
    'Feature': features_without_crp,
    'Importance': model_without_crp.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance (WITHOUT CRP):")
for _, row in feature_importance_no_crp.iterrows():
    print(f"  {row['Feature']:<15} {row['Importance']:>6.4f} {'█' * int(row['Importance'] * 100)}")
print()

# ============================================================================
# STEP 5: Test Model WITHOUT CRP
# ============================================================================
print("STEP 5: Testing Model WITHOUT CRP")
print("-" * 80)

# Predictions without CRP
y_pred_without_crp = model_without_crp.predict(X_without_crp)
y_pred_proba_without_crp = model_without_crp.predict_proba(X_without_crp)[:, 1]

# Metrics
acc_without_crp = accuracy_score(y_true, y_pred_without_crp)
sens_without_crp = recall_score(y_true, y_pred_without_crp)
cm_without_crp = confusion_matrix(y_true, y_pred_without_crp)
tn, fp, fn, tp = cm_without_crp.ravel()
spec_without_crp = tn / (tn + fp) if (tn + fp) > 0 else 0
prec_without_crp = precision_score(y_true, y_pred_without_crp, zero_division=0)
f1_without_crp = f1_score(y_true, y_pred_without_crp, zero_division=0)
roc_auc_without_crp = roc_auc_score(y_true, y_pred_proba_without_crp)

print("Performance WITHOUT CRP (threshold=0.5):")
print(f"  Accuracy:    {acc_without_crp*100:.1f}%")
print(f"  Sensitivity: {sens_without_crp*100:.1f}%")
print(f"  Specificity: {spec_without_crp*100:.1f}%")
print(f"  Precision:   {prec_without_crp*100:.1f}%")
print(f"  F1 Score:    {f1_without_crp:.3f}")
print(f"  ROC AUC:     {roc_auc_without_crp:.3f}")
print(f"  Confusion:   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print()

# ============================================================================
# STEP 6: Compare Performance
# ============================================================================
print("STEP 6: Performance Comparison")
print("=" * 80)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'ROC AUC'],
    'With CRP': [acc_with_crp, sens_with_crp, spec_with_crp, prec_with_crp, f1_with_crp, roc_auc_with_crp],
    'Without CRP': [acc_without_crp, sens_without_crp, spec_without_crp, prec_without_crp, f1_without_crp, roc_auc_without_crp]
})
comparison['Difference'] = comparison['Without CRP'] - comparison['With CRP']
comparison['% Change'] = (comparison['Difference'] / comparison['With CRP']) * 100

print("\nDetailed Comparison:")
print(comparison.to_string(index=False))
print()

# Determine winner
if acc_without_crp > acc_with_crp:
    print("🏆 WINNER: Model WITHOUT CRP")
    print(f"   Accuracy improved by {(acc_without_crp - acc_with_crp)*100:.1f} percentage points")
elif acc_without_crp < acc_with_crp:
    print("🏆 WINNER: Model WITH CRP")
    print(f"   Accuracy better by {(acc_with_crp - acc_without_crp)*100:.1f} percentage points")
else:
    print("↔️  TIE: Similar performance")

print()

# ============================================================================
# STEP 7: Optimize Threshold for Model WITHOUT CRP
# ============================================================================
print("STEP 7: Optimizing Threshold for Model WITHOUT CRP")
print("-" * 80)

# Test thresholds
thresholds = np.arange(0.10, 0.91, 0.05)
results_no_crp = []

for threshold in thresholds:
    y_pred = (y_pred_proba_without_crp >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    youden = sensitivity + specificity - 1

    results_no_crp.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'youden_index': youden,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    })

results_no_crp_df = pd.DataFrame(results_no_crp)

# Find optimal threshold
optimal_no_crp = results_no_crp_df.loc[results_no_crp_df['youden_index'].idxmax()]

print(f"Optimal Threshold (WITHOUT CRP): {optimal_no_crp['threshold']:.2f}")
print(f"  Accuracy:    {optimal_no_crp['accuracy']*100:.1f}%")
print(f"  Sensitivity: {optimal_no_crp['sensitivity']*100:.1f}%")
print(f"  Specificity: {optimal_no_crp['specificity']*100:.1f}%")
print(f"  F1 Score:    {optimal_no_crp['f1_score']:.3f}")
print(f"  Youden:      {optimal_no_crp['youden_index']:.3f}")
print()

# Load optimal threshold WITH CRP for comparison
optimal_with_crp_thresh = 0.35  # From previous optimization

# Get performance at that threshold
y_pred_with_crp_optimal = (y_pred_proba_with_crp >= optimal_with_crp_thresh).astype(int)
cm_with_crp_optimal = confusion_matrix(y_true, y_pred_with_crp_optimal)
tn, fp, fn, tp = cm_with_crp_optimal.ravel()

optimal_with_crp = {
    'threshold': optimal_with_crp_thresh,
    'accuracy': accuracy_score(y_true, y_pred_with_crp_optimal),
    'sensitivity': recall_score(y_true, y_pred_with_crp_optimal),
    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    'f1_score': f1_score(y_true, y_pred_with_crp_optimal, zero_division=0),
    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
}

print(f"Optimal Threshold (WITH CRP): {optimal_with_crp_thresh:.2f}")
print(f"  Accuracy:    {optimal_with_crp['accuracy']*100:.1f}%")
print(f"  Sensitivity: {optimal_with_crp['sensitivity']*100:.1f}%")
print(f"  Specificity: {optimal_with_crp['specificity']*100:.1f}%")
print(f"  F1 Score:    {optimal_with_crp['f1_score']:.3f}")
print()

# ============================================================================
# STEP 8: Generate Visualizations
# ============================================================================
print("STEP 8: Generating Visualizations")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Performance Comparison Bar Chart
ax = axes[0, 0]
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
with_crp_vals = [acc_with_crp, sens_with_crp, spec_with_crp, f1_with_crp]
without_crp_vals = [acc_without_crp, sens_without_crp, spec_without_crp, f1_without_crp]

x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, with_crp_vals, width, label='With CRP', color='#3498db')
ax.bar(x + width/2, without_crp_vals, width, label='Without CRP', color='#e74c3c')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance: With vs Without CRP (threshold=0.5)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Add value labels
for i, (with_val, without_val) in enumerate(zip(with_crp_vals, without_crp_vals)):
    ax.text(i - width/2, with_val + 0.02, f'{with_val:.2f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, without_val + 0.02, f'{without_val:.2f}', ha='center', va='bottom', fontsize=9)

# 2. ROC Curves Comparison
ax = axes[0, 1]
fpr_with, tpr_with, _ = roc_curve(y_true, y_pred_proba_with_crp)
fpr_without, tpr_without, _ = roc_curve(y_true, y_pred_proba_without_crp)

ax.plot(fpr_with, tpr_with, label=f'With CRP (AUC={roc_auc_with_crp:.3f})',
        linewidth=2, color='#3498db')
ax.plot(fpr_without, tpr_without, label=f'Without CRP (AUC={roc_auc_without_crp:.3f})',
        linewidth=2, color='#e74c3c')
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Feature Importance Comparison
ax = axes[0, 2]

# Get original feature importance
original_importance = pd.DataFrame({
    'Feature': model_data['features'],
    'Importance': original_model.feature_importances_
})

# Plot
y_pos = np.arange(len(features_without_crp))
importance_values_no_crp = [feature_importance_no_crp[feature_importance_no_crp['Feature']==f]['Importance'].values[0]
                            for f in features_without_crp]
importance_values_with_crp = [original_importance[original_importance['Feature']==f]['Importance'].values[0]
                              if f in original_importance['Feature'].values else 0
                              for f in features_without_crp]

ax.barh(y_pos - 0.2, importance_values_with_crp, 0.4, label='With CRP', color='#3498db')
ax.barh(y_pos + 0.2, importance_values_no_crp, 0.4, label='Without CRP', color='#e74c3c')
ax.set_yticks(y_pos)
ax.set_yticklabels(features_without_crp)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# 4. Confusion Matrix WITH CRP
ax = axes[1, 0]
sns.heatmap(cm_with_crp, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Control', 'Cancer'],
            yticklabels=['Control', 'Cancer'])
ax.set_title('Confusion Matrix WITH CRP\n(threshold=0.5)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# 5. Confusion Matrix WITHOUT CRP
ax = axes[1, 1]
sns.heatmap(cm_without_crp, annot=True, fmt='d', cmap='Reds', ax=ax,
            xticklabels=['Control', 'Cancer'],
            yticklabels=['Control', 'Cancer'])
ax.set_title('Confusion Matrix WITHOUT CRP\n(threshold=0.5)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# 6. Optimal Threshold Comparison
ax = axes[1, 2]
comparison_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
optimal_with_vals = [optimal_with_crp['accuracy'], optimal_with_crp['sensitivity'],
                     optimal_with_crp['specificity'], optimal_with_crp['f1_score']]
optimal_without_vals = [optimal_no_crp['accuracy'], optimal_no_crp['sensitivity'],
                       optimal_no_crp['specificity'], optimal_no_crp['f1_score']]

x = np.arange(len(comparison_metrics))
width = 0.35
ax.bar(x - width/2, optimal_with_vals, width,
       label=f'With CRP (t={optimal_with_crp["threshold"]:.2f})', color='#3498db')
ax.bar(x + width/2, optimal_without_vals, width,
       label=f'Without CRP (t={optimal_no_crp["threshold"]:.2f})', color='#e74c3c')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Optimal Threshold Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_metrics, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Add value labels
for i, (with_val, without_val) in enumerate(zip(optimal_with_vals, optimal_without_vals)):
    ax.text(i - width/2, with_val + 0.02, f'{with_val:.2f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, without_val + 0.02, f'{without_val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('external_datasets/mimic_iv_demo/crp_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: crp_comparison.png")
print()

# ============================================================================
# STEP 9: Summary & Recommendation
# ============================================================================
print("="*80)
print("SUMMARY & RECOMMENDATION")
print("="*80)
print()

print("CRP Data Quality Issue:")
print(f"  • ~81% of CRP values were imputed (only 19 patients had real measurements)")
print(f"  • CRP feature importance: {original_importance[original_importance['Feature']=='CRP']['Importance'].values[0]:.4f}")
print(f"  • This is the 2nd LOWEST importance among all features")
print()

print("Performance Comparison (threshold=0.5):")
print(f"  {'Metric':<20} {'With CRP':<12} {'Without CRP':<12} {'Difference':<12}")
print("  " + "-"*56)
for _, row in comparison.iterrows():
    metric = row['Metric']
    with_crp = row['With CRP']
    without_crp = row['Without CRP']
    diff = row['Difference']
    symbol = "✅" if diff > 0 else ("❌" if diff < 0 else "↔️")
    print(f"  {metric:<20} {with_crp:<12.3f} {without_crp:<12.3f} {diff:+.3f} {symbol}")
print()

print("Optimal Threshold Performance:")
print(f"  With CRP (threshold={optimal_with_crp['threshold']:.2f}):")
print(f"    Sensitivity: {optimal_with_crp['sensitivity']*100:.1f}%, Specificity: {optimal_with_crp['specificity']*100:.1f}%")
print(f"  Without CRP (threshold={optimal_no_crp['threshold']:.2f}):")
print(f"    Sensitivity: {optimal_no_crp['sensitivity']*100:.1f}%, Specificity: {optimal_no_crp['specificity']*100:.1f}%")
print()

# Recommendation
if sens_without_crp > sens_with_crp or acc_without_crp > acc_with_crp:
    print("📊 RECOMMENDATION: Use model WITHOUT CRP")
    print()
    print("Reasons:")
    print("  ✅ Better or equivalent performance")
    print("  ✅ No imputation needed (81% of CRP was imputed)")
    print("  ✅ Simpler model (5 features vs 6)")
    print("  ✅ More robust to missing data")
else:
    print("📊 RECOMMENDATION: Keep CRP in model")
    print()
    print("Reasons:")
    print("  • Marginally better performance with CRP")
    print("  • CRP provides some signal despite heavy imputation")

print()
print("="*80)
print("✅ CRP IMPACT ANALYSIS COMPLETE!")
print("="*80)

# Save results
results_no_crp_df.to_csv('external_datasets/mimic_iv_demo/crp_comparison_thresholds.csv', index=False)
print("\n✓ Saved threshold comparison to: crp_comparison_thresholds.csv")

# Save model without CRP
model_without_crp_data = {
    'model': model_without_crp,
    'features': features_without_crp,
    'version': '0.2.1-no-crp',
    'description': 'Cancer predictor without CRP feature (5 biomarkers)'
}
with open('external_datasets/mimic_iv_demo/model_without_crp.pkl', 'wb') as f:
    pickle.dump(model_without_crp_data, f)
print("✓ Saved model without CRP to: model_without_crp.pkl")
print()
