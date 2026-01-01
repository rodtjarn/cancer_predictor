"""
Optimize Decision Threshold for Cancer Prediction Model

This script:
1. Tests multiple decision thresholds (0.1 to 0.9)
2. Calculates metrics for each threshold
3. Finds optimal threshold using multiple criteria
4. Visualizes trade-offs
5. Recommends best threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OPTIMIZING DECISION THRESHOLD FOR CANCER PREDICTION")
print("="*80)
print()

# ============================================================================
# STEP 1: Load Previous Predictions
# ============================================================================
print("STEP 1: Loading Previous Predictions")
print("-" * 80)

predictions_df = pd.read_csv('external_datasets/mimic_iv_demo/mimic_predictions.csv')
y_true = predictions_df['cancer'].values
y_pred_proba = predictions_df['predicted_probability'].values

print(f"✓ Loaded {len(predictions_df)} predictions")
print(f"  - Cancer patients: {y_true.sum()}")
print(f"  - Control patients: {len(y_true) - y_true.sum()}")
print()

# ============================================================================
# STEP 2: Test Multiple Thresholds
# ============================================================================
print("STEP 2: Testing Multiple Decision Thresholds")
print("-" * 80)

# Test thresholds from 0.1 to 0.9 (include 0.5 explicitly)
thresholds = np.arange(0.10, 0.91, 0.05)
# Ensure 0.5 is included
if 0.5 not in thresholds:
    thresholds = np.append(thresholds, 0.5)
    thresholds = np.sort(thresholds)
results = []

for threshold in thresholds:
    # Make predictions with this threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Youden's index (Sensitivity + Specificity - 1)
    youden = recall + specificity - 1

    # Balanced accuracy
    balanced_acc = (recall + specificity) / 2

    # Store results
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'sensitivity': recall,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'youden_index': youden,
        'balanced_accuracy': balanced_acc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    })

results_df = pd.DataFrame(results)

print(f"✓ Tested {len(thresholds)} different thresholds")
print()

# Display sample results
print("Sample Threshold Results:")
print(results_df[['threshold', 'accuracy', 'sensitivity', 'specificity', 'f1_score']].head(10).to_string(index=False))
print()

# ============================================================================
# STEP 3: Find Optimal Thresholds
# ============================================================================
print("STEP 3: Finding Optimal Thresholds")
print("=" * 80)

# Find optimal thresholds using different criteria
optimal_youden = results_df.loc[results_df['youden_index'].idxmax()]
optimal_f1 = results_df.loc[results_df['f1_score'].idxmax()]
optimal_balanced = results_df.loc[results_df['balanced_accuracy'].idxmax()]

# Find threshold where sensitivity >= 80%
high_sens = results_df[results_df['sensitivity'] >= 0.80]
if len(high_sens) > 0:
    optimal_high_sens = high_sens.loc[high_sens['specificity'].idxmax()]
else:
    # If we can't get 80%, find highest sensitivity
    optimal_high_sens = results_df.loc[results_df['sensitivity'].idxmax()]

# Current threshold (0.5) - use approximate matching
current = results_df.iloc[(results_df['threshold'] - 0.5).abs().argsort()[:1]].iloc[0]

print("\n1. CURRENT THRESHOLD (0.5):")
print(f"   Accuracy:    {current['accuracy']*100:.1f}%")
print(f"   Sensitivity: {current['sensitivity']*100:.1f}%")
print(f"   Specificity: {current['specificity']*100:.1f}%")
print(f"   F1 Score:    {current['f1_score']:.3f}")

print("\n2. OPTIMAL BY YOUDEN'S INDEX (Sensitivity + Specificity - 1):")
print(f"   Threshold:   {optimal_youden['threshold']:.2f}")
print(f"   Accuracy:    {optimal_youden['accuracy']*100:.1f}%")
print(f"   Sensitivity: {optimal_youden['sensitivity']*100:.1f}%")
print(f"   Specificity: {optimal_youden['specificity']*100:.1f}%")
print(f"   F1 Score:    {optimal_youden['f1_score']:.3f}")
print(f"   Youden:      {optimal_youden['youden_index']:.3f}")

print("\n3. OPTIMAL BY F1 SCORE:")
print(f"   Threshold:   {optimal_f1['threshold']:.2f}")
print(f"   Accuracy:    {optimal_f1['accuracy']*100:.1f}%")
print(f"   Sensitivity: {optimal_f1['sensitivity']*100:.1f}%")
print(f"   Specificity: {optimal_f1['specificity']*100:.1f}%")
print(f"   F1 Score:    {optimal_f1['f1_score']:.3f}")

print("\n4. OPTIMAL BALANCED ACCURACY:")
print(f"   Threshold:   {optimal_balanced['threshold']:.2f}")
print(f"   Accuracy:    {optimal_balanced['accuracy']*100:.1f}%")
print(f"   Sensitivity: {optimal_balanced['sensitivity']*100:.1f}%")
print(f"   Specificity: {optimal_balanced['specificity']*100:.1f}%")
print(f"   Balanced:    {optimal_balanced['balanced_accuracy']*100:.1f}%")

print("\n5. HIGH SENSITIVITY (≥80% if possible):")
print(f"   Threshold:   {optimal_high_sens['threshold']:.2f}")
print(f"   Accuracy:    {optimal_high_sens['accuracy']*100:.1f}%")
print(f"   Sensitivity: {optimal_high_sens['sensitivity']*100:.1f}%")
print(f"   Specificity: {optimal_high_sens['specificity']*100:.1f}%")
print(f"   F1 Score:    {optimal_high_sens['f1_score']:.3f}")

print()
print("="*80)

# ============================================================================
# STEP 4: Detailed Comparison
# ============================================================================
print("\nSTEP 4: Detailed Comparison of Key Thresholds")
print("-" * 80)

comparison_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
comparison_results = results_df[results_df['threshold'].isin(comparison_thresholds)]

print("\nDetailed Metrics Comparison:")
print(comparison_results[['threshold', 'accuracy', 'sensitivity', 'specificity',
                          'precision', 'f1_score', 'tp', 'fp', 'fn', 'tn']].to_string(index=False))
print()

# ============================================================================
# STEP 5: Generate Visualizations
# ============================================================================
print("STEP 5: Generating Visualizations")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Sensitivity vs Specificity Trade-off
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(results_df['threshold'], results_df['sensitivity'],
         label='Sensitivity (Recall)', linewidth=2, marker='o', markersize=4)
ax1.plot(results_df['threshold'], results_df['specificity'],
         label='Specificity', linewidth=2, marker='s', markersize=4)
ax1.plot(results_df['threshold'], results_df['accuracy'],
         label='Accuracy', linewidth=2, marker='^', markersize=4, linestyle='--')
ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Current (0.5)')
ax1.axvline(optimal_youden['threshold'], color='green', linestyle='--', alpha=0.5,
            label=f"Optimal Youden ({optimal_youden['threshold']:.2f})")
ax1.set_xlabel('Decision Threshold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Sensitivity vs Specificity Trade-off', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# 2. F1 Score and Youden's Index
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(results_df['threshold'], results_df['f1_score'],
         label='F1 Score', linewidth=2, color='purple', marker='o', markersize=4)
ax2.plot(results_df['threshold'], results_df['youden_index'],
         label="Youden's Index", linewidth=2, color='orange', marker='s', markersize=4)
ax2.axvline(optimal_f1['threshold'], color='purple', linestyle='--', alpha=0.5)
ax2.axvline(optimal_youden['threshold'], color='orange', linestyle='--', alpha=0.5)
ax2.set_xlabel('Decision Threshold', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Optimization Metrics', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix Components
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(results_df['threshold'], results_df['tp'],
         label='True Positives', linewidth=2, marker='o', color='green')
ax3.plot(results_df['threshold'], results_df['tn'],
         label='True Negatives', linewidth=2, marker='s', color='blue')
ax3.plot(results_df['threshold'], results_df['fp'],
         label='False Positives', linewidth=2, marker='^', color='red')
ax3.plot(results_df['threshold'], results_df['fn'],
         label='False Negatives', linewidth=2, marker='v', color='orange')
ax3.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Current (0.5)')
ax3.axvline(optimal_youden['threshold'], color='gray', linestyle='--', alpha=0.5,
            label=f"Optimal ({optimal_youden['threshold']:.2f})")
ax3.set_xlabel('Decision Threshold', fontsize=12)
ax3.set_ylabel('Number of Patients', fontsize=12)
ax3.set_title('Confusion Matrix Components vs Threshold', fontsize=14, fontweight='bold')
ax3.legend(loc='best', ncol=3)
ax3.grid(True, alpha=0.3)

# 4. Comparison Bar Chart - Current vs Optimal
ax4 = fig.add_subplot(gs[2, 0])
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
current_values = [current['accuracy'], current['sensitivity'],
                 current['specificity'], current['f1_score']]
optimal_values = [optimal_youden['accuracy'], optimal_youden['sensitivity'],
                 optimal_youden['specificity'], optimal_youden['f1_score']]

x = np.arange(len(metrics))
width = 0.35
ax4.bar(x - width/2, current_values, width, label='Current (0.5)', color='lightcoral')
ax4.bar(x + width/2, optimal_values, width, label=f"Optimal ({optimal_youden['threshold']:.2f})",
        color='lightgreen')
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Current vs Optimal Threshold', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 1.1])

# Add value labels
for i, (curr, opt) in enumerate(zip(current_values, optimal_values)):
    ax4.text(i - width/2, curr + 0.02, f'{curr:.2f}', ha='center', va='bottom', fontsize=9)
    ax4.text(i + width/2, opt + 0.02, f'{opt:.2f}', ha='center', va='bottom', fontsize=9)

# 5. Optimal Confusion Matrix
ax5 = fig.add_subplot(gs[2, 1])
optimal_cm = np.array([[optimal_youden['tn'], optimal_youden['fp']],
                       [optimal_youden['fn'], optimal_youden['tp']]])
sns.heatmap(optimal_cm, annot=True, fmt='g', cmap='Greens', ax=ax5,
            xticklabels=['Control', 'Cancer'],
            yticklabels=['Control', 'Cancer'],
            cbar_kws={'label': 'Count'})
ax5.set_title(f'Confusion Matrix (Threshold={optimal_youden["threshold"]:.2f})',
              fontsize=14, fontweight='bold')
ax5.set_ylabel('True Label')
ax5.set_xlabel('Predicted Label')

# 6. Performance Improvement
ax6 = fig.add_subplot(gs[2, 2])
improvements = {
    'Accuracy': (optimal_youden['accuracy'] - current['accuracy']) * 100,
    'Sensitivity': (optimal_youden['sensitivity'] - current['sensitivity']) * 100,
    'Specificity': (optimal_youden['specificity'] - current['specificity']) * 100,
    'F1 Score': (optimal_youden['f1_score'] - current['f1_score']) * 100
}
colors = ['green' if v > 0 else 'red' for v in improvements.values()]
bars = ax6.barh(list(improvements.keys()), list(improvements.values()), color=colors)
ax6.axvline(0, color='black', linewidth=0.8)
ax6.set_xlabel('Change (percentage points)', fontsize=12)
ax6.set_title('Performance Improvement', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (metric, value) in enumerate(improvements.items()):
    label = f'{value:+.1f}pp'
    x_pos = value + (2 if value > 0 else -2)
    ax6.text(x_pos, i, label, va='center', ha='left' if value > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig('external_datasets/mimic_iv_demo/threshold_optimization.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: threshold_optimization.png")
print()

# ============================================================================
# STEP 6: Recommendation
# ============================================================================
print("="*80)
print("RECOMMENDATION")
print("="*80)
print()

print(f"📊 RECOMMENDED THRESHOLD: {optimal_youden['threshold']:.2f}")
print(f"   (Based on Youden's Index - maximizes Sensitivity + Specificity)")
print()
print("Performance with Recommended Threshold:")
print(f"  • Accuracy:    {optimal_youden['accuracy']*100:.1f}% (was {current['accuracy']*100:.1f}%)")
print(f"  • Sensitivity: {optimal_youden['sensitivity']*100:.1f}% (was {current['sensitivity']*100:.1f}%) ⬆")
print(f"  • Specificity: {optimal_youden['specificity']*100:.1f}% (was {current['specificity']*100:.1f}%)")
print(f"  • F1 Score:    {optimal_youden['f1_score']:.3f} (was {current['f1_score']:.3f}) ⬆")
print()
print("Confusion Matrix:")
print(f"  • True Positives:  {int(optimal_youden['tp'])} (was {int(current['tp'])})")
print(f"  • False Negatives: {int(optimal_youden['fn'])} (was {int(current['fn'])})")
print(f"  • True Negatives:  {int(optimal_youden['tn'])} (was {int(current['tn'])})")
print(f"  • False Positives: {int(optimal_youden['fp'])} (was {int(current['fp'])})")
print()

sens_improvement = (optimal_youden['sensitivity'] - current['sensitivity']) * 100
spec_loss = (current['specificity'] - optimal_youden['specificity']) * 100

print("Key Trade-offs:")
print(f"  ✅ Sensitivity improved by {sens_improvement:.1f} percentage points")
print(f"  ⚠️  Specificity decreased by {spec_loss:.1f} percentage points")
print(f"  ✅ Now detecting {int(optimal_youden['tp'])} out of {int(y_true.sum())} cancers ({optimal_youden['sensitivity']*100:.1f}%)")
print(f"  ⚠️  {int(optimal_youden['fp'])} controls incorrectly flagged as cancer")
print()

# Clinical interpretation
cancer_detected = int(optimal_youden['tp'])
cancer_missed = int(optimal_youden['fn'])
false_alarms = int(optimal_youden['fp'])

print("Clinical Interpretation:")
print(f"  • Out of {int(y_true.sum())} cancer patients:")
print(f"    - {cancer_detected} would be correctly detected ({optimal_youden['sensitivity']*100:.1f}%)")
print(f"    - {cancer_missed} would be missed ({(cancer_missed/y_true.sum())*100:.1f}%)")
print()
print(f"  • Out of {int((1-y_true).sum())} control patients:")
print(f"    - {int(optimal_youden['tn'])} would be correctly identified ({optimal_youden['specificity']*100:.1f}%)")
print(f"    - {false_alarms} would receive false alarms ({(false_alarms/(1-y_true).sum())*100:.1f}%)")
print()

# Alternative recommendations
print("Alternative Thresholds for Different Use Cases:")
print()
print(f"1. For Cancer SCREENING (prioritize sensitivity):")
print(f"   Threshold: {optimal_high_sens['threshold']:.2f}")
print(f"   - Sensitivity: {optimal_high_sens['sensitivity']*100:.1f}% (detect {int(optimal_high_sens['tp'])}/{int(y_true.sum())} cancers)")
print(f"   - Specificity: {optimal_high_sens['specificity']*100:.1f}% ({int(optimal_high_sens['fp'])} false alarms)")
print()
print(f"2. For DIAGNOSIS (balanced approach):")
print(f"   Threshold: {optimal_youden['threshold']:.2f} ⭐ RECOMMENDED")
print(f"   - Sensitivity: {optimal_youden['sensitivity']*100:.1f}%")
print(f"   - Specificity: {optimal_youden['specificity']*100:.1f}%")
print()
print(f"3. For CONFIRMATION (prioritize specificity):")
print(f"   Threshold: 0.50 (current)")
print(f"   - Sensitivity: {current['sensitivity']*100:.1f}%")
print(f"   - Specificity: {current['specificity']*100:.1f}%")
print()

print("="*80)
print("✅ THRESHOLD OPTIMIZATION COMPLETE!")
print("="*80)

# Save optimized threshold results
results_df.to_csv('external_datasets/mimic_iv_demo/threshold_optimization_results.csv', index=False)
print("\n✓ Saved threshold analysis to: threshold_optimization_results.csv")

# Save updated predictions with optimal threshold
predictions_df['predicted_cancer_optimal'] = (y_pred_proba >= optimal_youden['threshold']).astype(int)
predictions_df['optimal_threshold'] = optimal_youden['threshold']
predictions_df.to_csv('external_datasets/mimic_iv_demo/mimic_predictions_optimized.csv', index=False)
print("✓ Saved optimized predictions to: mimic_predictions_optimized.csv")
print()
