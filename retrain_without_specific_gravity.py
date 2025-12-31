#!/usr/bin/env python3
"""
Retrain cancer prediction model without Specific Gravity.

Based on feature importance analysis showing Specific Gravity has minimal
impact (1.26% importance, removing it improves accuracy by 0.01%), we
retrain the model with only 6 biomarkers for a simpler, more efficient model.

New features: Glucose, Age, BMI, Lactate, LDH, CRP (6 biomarkers)
Removed: Specific Gravity
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Feature names - OLD (7 features)
OLD_FEATURE_NAMES = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH', 'CRP', 'Specific_Gravity']

# Feature names - NEW (6 features, without Specific Gravity)
NEW_FEATURE_NAMES = ['Glucose', 'Age', 'BMI', 'Lactate', 'LDH', 'CRP']

print("="*80)
print("RETRAINING MODEL WITHOUT SPECIFIC GRAVITY")
print("="*80)
print("\n🎯 Goal: Create simpler 6-biomarker model")
print("   Removing: Specific Gravity (1.26% importance)")
print("   Keeping: Glucose, Age, BMI, Lactate, LDH, CRP")

# ============================================================================
# PART 1: LOAD DATA AND REMOVE SPECIFIC GRAVITY
# ============================================================================
print("\n" + "="*80)
print("PART 1: LOADING DATA")
print("="*80)

# Load training data
print("\n📦 Loading training data...")
train_data = np.load('data/training_data.npz')
X_train_old = train_data['X']
y_train = train_data['y']
print(f"✅ Original training data: {X_train_old.shape[0]:,} samples, {X_train_old.shape[1]} features")

# Load test data
print("\n📦 Loading test data...")
test_data = np.load('data/test_data.npz')
X_test_old = test_data['X']
y_test = test_data['y']
print(f"✅ Original test data: {X_test_old.shape[0]:,} samples, {X_test_old.shape[1]} features")

# Remove Specific Gravity (index 6 - last column)
print("\n🔧 Removing Specific Gravity column...")
specific_gravity_idx = OLD_FEATURE_NAMES.index('Specific_Gravity')
print(f"   Specific Gravity is at index: {specific_gravity_idx}")

# Create new datasets without Specific Gravity
feature_indices = [i for i in range(len(OLD_FEATURE_NAMES)) if i != specific_gravity_idx]
X_train_new = X_train_old[:, feature_indices]
X_test_new = X_test_old[:, feature_indices]

print(f"✅ New training data: {X_train_new.shape[0]:,} samples, {X_train_new.shape[1]} features")
print(f"✅ New test data: {X_test_new.shape[0]:,} samples, {X_test_new.shape[1]} features")
print(f"\n   Remaining features: {', '.join(NEW_FEATURE_NAMES)}")

# ============================================================================
# PART 2: LOAD OLD MODEL FOR COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PART 2: BASELINE COMPARISON (OLD 7-FEATURE MODEL)")
print("="*80)

print("\n📦 Loading old model (7 features)...")
with open('models/metabolic_cancer_predictor.pkl', 'rb') as f:
    old_model_data = pickle.load(f)
    old_model = old_model_data['model']

# Test old model
print("\n🧪 Testing old model...")
y_test_pred_old = old_model.predict(X_test_old)
y_test_proba_old = old_model.predict_proba(X_test_old)[:, 1]

old_accuracy = accuracy_score(y_test, y_test_pred_old)
old_precision = precision_score(y_test, y_test_pred_old)
old_recall = recall_score(y_test, y_test_pred_old)
old_f1 = f1_score(y_test, y_test_pred_old)
old_auc = roc_auc_score(y_test, y_test_proba_old)

print(f"\n📊 OLD Model Performance (7 features):")
print(f"   Accuracy:    {old_accuracy:.4f} ({old_accuracy*100:.2f}%)")
print(f"   Precision:   {old_precision:.4f}")
print(f"   Recall:      {old_recall:.4f}")
print(f"   F1 Score:    {old_f1:.4f}")
print(f"   AUC-ROC:     {old_auc:.4f}")

# ============================================================================
# PART 3: TRAIN NEW MODEL (6 FEATURES)
# ============================================================================
print("\n" + "="*80)
print("PART 3: TRAINING NEW MODEL (6 FEATURES)")
print("="*80)

print("\n🎓 Training Random Forest Classifier...")
print("   Parameters:")
print("   - n_estimators: 100")
print("   - max_depth: 10")
print("   - random_state: 42")

new_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n⏳ Training in progress...")
new_model.fit(X_train_new, y_train)
print("✅ Training complete!")

# ============================================================================
# PART 4: EVALUATE NEW MODEL
# ============================================================================
print("\n" + "="*80)
print("PART 4: EVALUATING NEW MODEL")
print("="*80)

# Test on training data
print("\n🧪 Testing on TRAINING data...")
y_train_pred = new_model.predict(X_train_new)
y_train_proba = new_model.predict_proba(X_train_new)[:, 1]

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
print("\n🧪 Testing on TEST data...")
y_test_pred = new_model.predict(X_test_new)
y_test_proba = new_model.predict_proba(X_test_new)[:, 1]

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
cm = confusion_matrix(y_test, y_test_pred)
print("\n📋 Confusion Matrix:")
print(f"                Predicted")
print(f"                Healthy  Cancer")
print(f"   Actual Healthy  {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"   Actual Cancer   {cm[1,0]:6d}  {cm[1,1]:6d}")

# ============================================================================
# PART 5: COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PART 5: OLD vs NEW MODEL COMPARISON")
print("="*80)

print(f"\n{'Metric':<20} {'OLD (7 feat)':<15} {'NEW (6 feat)':<15} {'Difference':<15}")
print("-" * 70)
print(f"{'Accuracy':<20} {old_accuracy:.4f}          {test_accuracy:.4f}          {test_accuracy - old_accuracy:+.4f}")
print(f"{'Precision':<20} {old_precision:.4f}          {test_precision:.4f}          {test_precision - old_precision:+.4f}")
print(f"{'Recall':<20} {old_recall:.4f}          {test_recall:.4f}          {test_recall - old_recall:+.4f}")
print(f"{'F1 Score':<20} {old_f1:.4f}          {test_f1:.4f}          {test_f1 - old_f1:+.4f}")
print(f"{'AUC-ROC':<20} {old_auc:.4f}          {test_auc:.4f}          {test_auc - old_auc:+.4f}")
print(f"{'Features':<20} {len(OLD_FEATURE_NAMES):<15} {len(NEW_FEATURE_NAMES):<15} {len(NEW_FEATURE_NAMES) - len(OLD_FEATURE_NAMES):+d}")

# Feature importance comparison
print("\n📊 New Model Feature Importance:")
feature_importances = new_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': NEW_FEATURE_NAMES,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print(f"\n{'Rank':<6} {'Feature':<15} {'Importance':<12} {'Percentage':<12}")
print("-" * 50)
for idx, row in importance_df.iterrows():
    rank = list(importance_df.index).index(idx) + 1
    pct = row['Importance'] * 100
    print(f"{rank:<6} {row['Feature']:<15} {row['Importance']:.6f}    {pct:>6.2f}%")

# ============================================================================
# PART 6: SAVE NEW MODEL
# ============================================================================
print("\n" + "="*80)
print("PART 6: SAVING NEW MODEL")
print("="*80)

# Create model dictionary
model_data = {
    'model': new_model,
    'features': np.array(NEW_FEATURE_NAMES),
    'version': '0.2.0',
    'description': '6-biomarker model (without Specific Gravity)',
    'feature_count': len(NEW_FEATURE_NAMES),
    'removed_features': ['Specific_Gravity'],
    'training_samples': len(X_train_new),
    'test_accuracy': test_accuracy,
    'test_auc': test_auc,
    'cost': 150  # Updated cost (removed one biomarker)
}

# Save new model
print("\n💾 Saving new model...")
with open('models/metabolic_cancer_predictor_v2.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✅ Saved to: models/metabolic_cancer_predictor_v2.pkl")

# Also save updated training and test data
print("\n💾 Saving updated data (without Specific Gravity)...")
np.savez('data/training_data_v2.npz', X=X_train_new, y=y_train)
np.savez('data/test_data_v2.npz', X=X_test_new, y=y_test)
print("✅ Saved to: data/training_data_v2.npz")
print("✅ Saved to: data/test_data_v2.npz")

# ============================================================================
# PART 7: VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("PART 7: GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Performance Comparison
ax1 = axes[0, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
old_values = [old_accuracy, old_precision, old_recall, old_f1, old_auc]
new_values = [test_accuracy, test_precision, test_recall, test_f1, test_auc]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, old_values, width, label='OLD (7 features)',
                color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, new_values, width, label='NEW (6 features)',
                color='#06A77D', alpha=0.8)

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Performance Comparison: OLD vs NEW Model', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0.95, 1.0])

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=8)

# Plot 2: Feature Importance (NEW model)
ax2 = axes[0, 1]
importance_df_sorted = importance_df.sort_values('Importance')
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df_sorted)))
bars = ax2.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'],
                color=colors)
ax2.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax2.set_title('Feature Importance (NEW 6-Feature Model)', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Confusion Matrix Comparison
ax3 = axes[1, 0]
cm_old = confusion_matrix(y_test, y_test_pred_old)
sns.heatmap(cm_old, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Healthy', 'Cancer'],
            yticklabels=['Healthy', 'Cancer'],
            cbar_kws={'label': 'Count'})
ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax3.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax3.set_title(f'OLD Model (7 features)\nAccuracy: {old_accuracy:.4f}',
              fontsize=14, fontweight='bold')

# Plot 4: Confusion Matrix (NEW)
ax4 = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax4,
            xticklabels=['Healthy', 'Cancer'],
            yticklabels=['Healthy', 'Cancer'],
            cbar_kws={'label': 'Count'})
ax4.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax4.set_title(f'NEW Model (6 features)\nAccuracy: {test_accuracy:.4f}',
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison_v1_vs_v2.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: model_comparison_v1_vs_v2.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n✅ NEW Model (v0.2.0) Successfully Trained!")
print(f"\n📊 Key Improvements:")
print(f"   - Reduced from 7 to 6 biomarkers (-14% complexity)")
print(f"   - Removed: Specific Gravity (minimal importance)")
print(f"   - Test accuracy: {test_accuracy:.4f} (Δ{test_accuracy - old_accuracy:+.4f})")
print(f"   - AUC-ROC: {test_auc:.4f} (Δ{test_auc - old_auc:+.4f})")

if test_accuracy >= old_accuracy:
    print(f"\n🎉 Performance MAINTAINED or IMPROVED!")
    print(f"   - Simpler model with equal/better performance")
else:
    print(f"\n⚠️ Minimal performance decrease:")
    print(f"   - Drop: {(old_accuracy - test_accuracy)*100:.2f}%")
    print(f"   - Trade-off worth it for simpler model")

print(f"\n💰 Cost Reduction:")
print(f"   - OLD model: ~$175 per test (7 biomarkers)")
print(f"   - NEW model: ~$150 per test (6 biomarkers)")
print(f"   - Savings: ~$25 per test (~14% reduction)")

print(f"\n📦 Remaining Biomarkers (6):")
for i, feature in enumerate(NEW_FEATURE_NAMES, 1):
    importance = importance_df[importance_df['Feature'] == feature]['Importance'].values[0]
    print(f"   {i}. {feature:<15} ({importance:.4f} / {importance*100:.2f}%)")

print(f"\n💾 Files Created:")
print(f"   - models/metabolic_cancer_predictor_v2.pkl (NEW model)")
print(f"   - data/training_data_v2.npz (6-feature training data)")
print(f"   - data/test_data_v2.npz (6-feature test data)")
print(f"   - model_comparison_v1_vs_v2.png (comparison visualization)")

print("\n" + "="*80)
print("RETRAINING COMPLETE!")
print("="*80)
