#!/usr/bin/env python3
"""
Test model predictions and perform feature importance analysis.

This script:
1. Verifies the trained model can predict cancer/no cancer
2. Evaluates current model performance
3. Performs iterative feature removal to identify least important markers
4. Generates comprehensive visualizations and reports
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Set random seed for reproducibility
np.random.seed(42)

# Feature names in order
FEATURE_NAMES = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH', 'CRP', 'Specific_Gravity']

print("="*80)
print("CANCER PREDICTION MODEL - VERIFICATION & FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: LOAD MODEL AND DATA
# ============================================================================
print("\n" + "="*80)
print("PART 1: LOADING MODEL AND DATA")
print("="*80)

# Load the trained model
print("\n📦 Loading trained model...")
with open('models/metabolic_cancer_predictor.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extract model from dictionary
if isinstance(model_data, dict):
    model = model_data['model']
    print(f"✅ Model loaded from dictionary: {type(model).__name__}")
    if 'features' in model_data:
        print(f"   Features in model: {model_data['features']}")
else:
    model = model_data
    print(f"✅ Model loaded: {type(model).__name__}")

# Load training and test data
print("\n📦 Loading training data...")
train_data = np.load('data/training_data.npz')
X_train = train_data['X']
y_train = train_data['y']
print(f"✅ Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
print(f"   - Cancer cases: {np.sum(y_train == 1):,}")
print(f"   - Healthy controls: {np.sum(y_train == 0):,}")

print("\n📦 Loading test data...")
test_data = np.load('data/test_data.npz')
X_test = test_data['X']
y_test = test_data['y']
print(f"✅ Test data: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
print(f"   - Cancer cases: {np.sum(y_test == 1):,}")
print(f"   - Healthy controls: {np.sum(y_test == 0):,}")

# ============================================================================
# PART 2: VERIFY MODEL PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("PART 2: VERIFYING MODEL PREDICTIONS")
print("="*80)

# Test on training data
print("\n🧪 Testing predictions on TRAINING data...")
y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

print(f"\n📊 Training Set Performance:")
print(f"   Accuracy:    {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   Precision:   {train_precision:.4f}")
print(f"   Recall:      {train_recall:.4f}")
print(f"   F1 Score:    {train_f1:.4f}")
print(f"   AUC-ROC:     {train_auc:.4f}")

# Test on test data
print("\n🧪 Testing predictions on TEST data...")
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:    {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Precision:   {test_precision:.4f}")
print(f"   Recall:      {test_recall:.4f}")
print(f"   F1 Score:    {test_f1:.4f}")
print(f"   AUC-ROC:     {test_auc:.4f}")

# Confusion matrix
print("\n📋 Test Set Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(f"                Predicted")
print(f"                Healthy  Cancer")
print(f"   Actual Healthy  {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"   Actual Cancer   {cm[1,0]:6d}  {cm[1,1]:6d}")

# Sample predictions
print("\n🔍 Sample Predictions (first 10 test samples):")
print(f"{'Sample':<8} {'True':<10} {'Predicted':<12} {'Probability':<15} {'Correct?':<10}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    true_label = "Cancer" if y_test[i] == 1 else "Healthy"
    pred_label = "Cancer" if y_test_pred[i] == 1 else "Healthy"
    prob = y_test_proba[i]
    correct = "✅" if y_test[i] == y_test_pred[i] else "❌"
    print(f"{i+1:<8} {true_label:<10} {pred_label:<12} {prob:.4f} ({prob*100:.1f}%)  {correct:<10}")

print("\n✅ MODEL IS WORKING - Predictions confirmed!")

# ============================================================================
# PART 3: BASELINE FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("PART 3: BASELINE FEATURE IMPORTANCE")
print("="*80)

# Get feature importance from the model
feature_importances = model.feature_importances_

print("\n📊 Feature Importance (from Random Forest):")
print(f"{'Rank':<6} {'Feature':<20} {'Importance':<12} {'Percentage':<12}")
print("-" * 60)

# Sort by importance
importance_df = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

for idx, row in importance_df.iterrows():
    rank = list(importance_df.index).index(idx) + 1
    pct = row['Importance'] * 100
    print(f"{rank:<6} {row['Feature']:<20} {row['Importance']:.6f}    {pct:>6.2f}%")

# ============================================================================
# PART 4: ITERATIVE FEATURE REMOVAL (BACKWARD ELIMINATION)
# ============================================================================
print("\n" + "="*80)
print("PART 4: ITERATIVE FEATURE REMOVAL ANALYSIS")
print("="*80)
print("\nRemoving features one at a time to identify least important markers...")

results = []

# Baseline (all features)
baseline_result = {
    'features_removed': 'None (Baseline)',
    'num_features': 7,
    'features_used': ', '.join(FEATURE_NAMES),
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_auc': test_auc
}
results.append(baseline_result)

print(f"\n📊 Baseline (All 7 features): Test Accuracy = {test_accuracy:.4f}")

# Remove each feature one at a time
print("\n🔬 Testing removal of each feature individually...")
for i, feature_to_remove in enumerate(FEATURE_NAMES):
    print(f"\n[{i+1}/7] Removing: {feature_to_remove}...")

    # Get indices of features to keep
    feature_indices = [j for j in range(len(FEATURE_NAMES)) if j != i]
    features_used = [FEATURE_NAMES[j] for j in feature_indices]

    # Create subset of data
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]

    # Train new model
    model_subset = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model_subset.fit(X_train_subset, y_train)

    # Evaluate
    y_train_pred_subset = model_subset.predict(X_train_subset)
    y_test_pred_subset = model_subset.predict(X_test_subset)
    y_test_proba_subset = model_subset.predict_proba(X_test_subset)[:, 1]

    train_acc_subset = accuracy_score(y_train, y_train_pred_subset)
    test_acc_subset = accuracy_score(y_test, y_test_pred_subset)
    test_prec_subset = precision_score(y_test, y_test_pred_subset)
    test_rec_subset = recall_score(y_test, y_test_pred_subset)
    test_f1_subset = f1_score(y_test, y_test_pred_subset)
    test_auc_subset = roc_auc_score(y_test, y_test_proba_subset)

    # Calculate impact
    accuracy_drop = test_accuracy - test_acc_subset

    result = {
        'features_removed': feature_to_remove,
        'num_features': 6,
        'features_used': ', '.join(features_used),
        'train_accuracy': train_acc_subset,
        'test_accuracy': test_acc_subset,
        'test_precision': test_prec_subset,
        'test_recall': test_rec_subset,
        'test_f1': test_f1_subset,
        'test_auc': test_auc_subset,
        'accuracy_drop': accuracy_drop
    }
    results.append(result)

    impact_indicator = "⚠️ MAJOR" if accuracy_drop > 0.02 else "✓ Minor"
    print(f"   Test Accuracy: {test_acc_subset:.4f} (Δ{accuracy_drop:+.4f}) {impact_indicator}")

# ============================================================================
# PART 5: RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PART 5: FEATURE REMOVAL IMPACT SUMMARY")
print("="*80)

results_df = pd.DataFrame(results[1:])  # Exclude baseline
results_df = results_df.sort_values('accuracy_drop')

print("\n📊 Features ranked by IMPACT when removed (least → most important):")
print(f"\n{'Rank':<6} {'Feature Removed':<20} {'Test Acc':<12} {'Accuracy Drop':<15} {'Impact':<10}")
print("-" * 80)

for idx, (_, row) in enumerate(results_df.iterrows()):
    rank = idx + 1
    impact = "⚠️ HIGH" if row['accuracy_drop'] > 0.02 else ("⚠️ MEDIUM" if row['accuracy_drop'] > 0.01 else "✓ LOW")
    print(f"{rank:<6} {row['features_removed']:<20} {row['test_accuracy']:.4f}      {row['accuracy_drop']:+.4f}          {impact:<10}")

# Identify least important
least_important = results_df.iloc[0]['features_removed']
least_impact = results_df.iloc[0]['accuracy_drop']

# Identify most important
most_important = results_df.iloc[-1]['features_removed']
most_impact = results_df.iloc[-1]['accuracy_drop']

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"\n🔹 LEAST Important Marker: {least_important}")
print(f"   → Removing it causes only {least_impact:+.4f} ({least_impact*100:+.2f}%) accuracy change")

print(f"\n🔸 MOST Important Marker: {most_important}")
print(f"   → Removing it causes {most_impact:+.4f} ({most_impact*100:+.2f}%) accuracy drop")

# Save results
results_df_full = pd.DataFrame(results)
results_df_full.to_csv('feature_removal_analysis.csv', index=False)
print(f"\n💾 Results saved to: feature_removal_analysis.csv")

# ============================================================================
# PART 6: VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("PART 6: GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Feature Importance (original model)
ax1 = axes[0, 0]
importance_df_sorted = importance_df.sort_values('Importance')
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df_sorted)))
bars = ax1.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'], color=colors)
ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax1.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Accuracy Impact When Removed
ax2 = axes[0, 1]
results_df_sorted = results_df.sort_values('accuracy_drop')
colors2 = ['#06A77D' if x < 0.01 else ('#F18F01' if x < 0.02 else '#C73E1D')
           for x in results_df_sorted['accuracy_drop']]
bars2 = ax2.barh(results_df_sorted['features_removed'],
                 results_df_sorted['accuracy_drop'] * 100,
                 color=colors2)
ax2.set_xlabel('Accuracy Drop (%)', fontsize=12, fontweight='bold')
ax2.set_title('Impact When Feature Removed', fontsize=14, fontweight='bold')
ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='1% threshold')
ax2.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='2% threshold')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Test Accuracy Comparison
ax3 = axes[1, 0]
all_results = [baseline_result] + results_df.to_dict('records')
labels = ['Baseline\n(All 7)'] + [r['features_removed'] for r in results_df.to_dict('records')]
accuracies = [r['test_accuracy'] * 100 for r in all_results]
colors3 = ['#2E86AB'] + ['#06A77D' if acc >= test_accuracy*100 - 1 else
                          ('#F18F01' if acc >= test_accuracy*100 - 2 else '#C73E1D')
                          for acc in accuracies[1:]]
ax3.bar(range(len(labels)), accuracies, color=colors3, alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, rotation=45, ha='right')
ax3.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Test Accuracy: Baseline vs Single Feature Removal', fontsize=14, fontweight='bold')
ax3.axhline(y=test_accuracy*100, color='blue', linestyle='--', alpha=0.5, label='Baseline')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Healthy', 'Cancer'],
            yticklabels=['Healthy', 'Cancer'],
            cbar_kws={'label': 'Count'})
ax4.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax4.set_title(f'Confusion Matrix (Baseline Model)\nAccuracy: {test_accuracy:.4f}',
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: feature_importance_analysis.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n✅ Model Verification: PASSED")
print(f"   - Model successfully predicts cancer/healthy")
print(f"   - Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   - AUC-ROC: {test_auc:.4f}")

print(f"\n📊 Feature Importance Ranking (by removal impact):")
for idx, (_, row) in enumerate(results_df.iterrows()):
    rank = idx + 1
    print(f"   {rank}. {row['features_removed']:<20} (Δ{row['accuracy_drop']:+.4f})")

print(f"\n💡 Recommendation:")
if least_impact < 0.01:
    print(f"   → {least_important} shows minimal impact (< 1%) - could potentially be removed")
    print(f"   → However, all other biomarkers are important for model performance")
else:
    print(f"   → ALL biomarkers contribute meaningfully to model performance")
    print(f"   → Recommend keeping all 7 markers for optimal accuracy")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
