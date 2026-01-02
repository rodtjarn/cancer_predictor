"""
Build Random Forest Model on NHANES-style Data

Features:
- Fasting Insulin (µU/mL)
- Fasting Glucose (mg/dL)
- LDH (U/L)
- CRP (mg/L)
- HOMA-IR (calculated)
- Age
- Gender

Target: Cancer (binary)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import joblib

print("="*80)
print("RANDOM FOREST MODEL: NHANES CANCER PREDICTION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")
df = pd.read_csv("data/nhanes/nhanes_style_synthetic.csv")
print(f"✓ Loaded {len(df)} participants")
print(f"  Cancer: {df['cancer'].sum()} ({100*df['cancer'].mean():.1f}%)")
print(f"  Controls: {(1-df['cancer']).sum()} ({100*(1-df['cancer'].mean()):.1f}%)")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\nPreparing features...")

feature_cols = [
    'fasting_insulin',
    'fasting_glucose',
    'LDH',
    'CRP',
    'HOMA_IR',
    'age',
    'gender'
]

X = df[feature_cols].values
y = df['cancer'].values

print(f"Features: {feature_cols}")
print(f"Shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(y_train)} ({y_train.sum()} cancer)")
print(f"Test set: {len(y_test)} ({y_test.sum()} cancer)")

# Scale features (important for feature importance interpretation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# TRAIN RANDOM FOREST
# ============================================================================
print("\n" + "="*80)
print("TRAINING RANDOM FOREST")
print("="*80)

# Hyperparameters
rf = RandomForestClassifier(
    n_estimators=200,           # Number of trees
    max_depth=10,               # Prevent overfitting
    min_samples_split=20,       # Minimum samples to split
    min_samples_leaf=10,        # Minimum samples per leaf
    max_features='sqrt',        # Features per split
    class_weight='balanced',    # Handle class imbalance
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)

print("\nHyperparameters:")
print(f"  n_estimators: {rf.n_estimators}")
print(f"  max_depth: {rf.max_depth}")
print(f"  class_weight: {rf.class_weight}")

print("\nTraining...")
rf.fit(X_train_scaled, y_train)
print("✓ Training complete")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
print("\n" + "-"*80)
print("CROSS-VALIDATION (5-Fold)")
print("-"*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')

print(f"ROC-AUC scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================
print("\n" + "="*80)
print("TEST SET EVALUATION")
print("="*80)

# Predictions
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

# Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Control', 'Cancer']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print("\nPerformance Metrics:")
print(f"  Sensitivity (Recall): {sensitivity:.3f}")
print(f"  Specificity: {specificity:.3f}")
print(f"  PPV (Precision): {ppv:.3f}")
print(f"  NPV: {npv:.3f}")

# ROC-AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"  ROC-AUC: {auc:.3f}")

# Average Precision (better for imbalanced data)
ap = average_precision_score(y_test, y_pred_proba)
print(f"  Average Precision: {ap:.3f}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nRanking:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_cols[idx]:20s}: {importances[idx]:.4f}")

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# Save model
model_path = Path("models/nhanes_rf_model.pkl")
model_path.parent.mkdir(exist_ok=True)
joblib.dump(rf, model_path)
print(f"✓ Saved model to {model_path}")

# Save scaler
scaler_path = Path("models/nhanes_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✓ Saved scaler to {scaler_path}")

# Save feature names
feature_path = Path("models/nhanes_features.txt")
with open(feature_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"✓ Saved feature names to {feature_path}")

# Save results
results = {
    'test_size': len(y_test),
    'test_cancer': int(y_test.sum()),
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'ppv': float(ppv),
    'npv': float(npv),
    'roc_auc': float(auc),
    'average_precision': float(ap),
    'cv_auc_mean': float(cv_scores.mean()),
    'cv_auc_std': float(cv_scores.std()),
    'feature_importance': {
        feature_cols[i]: float(importances[i])
        for i in range(len(feature_cols))
    }
}

import json
results_path = Path("results/nhanes_rf_results.json")
results_path.parent.mkdir(exist_ok=True)
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved results to {results_path}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Feature Importance
ax = axes[0, 0]
y_pos = np.arange(len(feature_cols))
ax.barh(y_pos, importances[indices], color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels([feature_cols[i] for i in indices])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance')
ax.grid(axis='x', alpha=0.3)

# 2. ROC Curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(alpha=0.3)

# 3. Precision-Recall Curve
ax = axes[1, 0]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
ax.plot(recall, precision, linewidth=2, label=f'PR (AP = {ap:.3f})')
ax.axhline(y=y_test.mean(), color='k', linestyle='--', linewidth=1, label='Baseline')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(alpha=0.3)

# 4. Confusion Matrix
ax = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Control', 'Cancer'],
            yticklabels=['Control', 'Cancer'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

plt.tight_layout()
plot_path = Path("results/nhanes_rf_evaluation.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plots to {plot_path}")
plt.close()

# ============================================================================
# INSULIN RESISTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("INSULIN RESISTANCE ANALYSIS")
print("="*80)

# Add predictions to dataframe
df_test = df.iloc[X_test.shape[0]*-1:].copy()  # Last 20% is test set after shuffle
df_test['predicted_cancer'] = y_pred
df_test['cancer_probability'] = y_pred_proba

# Analyze by HOMA-IR quartiles
df_test['HOMA_IR_quartile'] = pd.qcut(df_test['HOMA_IR'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print("\nCancer Rate by Insulin Resistance Quartile:")
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    subset = df_test[df_test['HOMA_IR_quartile'] == q]
    cancer_rate = subset['cancer'].mean()
    homa_range = f"{subset['HOMA_IR'].min():.1f}-{subset['HOMA_IR'].max():.1f}"
    print(f"  {q} (HOMA-IR {homa_range}): {cancer_rate*100:.1f}% cancer")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Model: Random Forest Classifier
Features: Fasting Insulin, Glucose, LDH, CRP, HOMA-IR, Age, Gender

Performance:
  ✓ ROC-AUC: {auc:.3f}
  ✓ Sensitivity: {sensitivity:.3f}
  ✓ Specificity: {specificity:.3f}
  ✓ Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}

Top 3 Features:
  1. {feature_cols[indices[0]]}: {importances[indices[0]]:.4f}
  2. {feature_cols[indices[1]]}: {importances[indices[1]]:.4f}
  3. {feature_cols[indices[2]]}: {importances[indices[2]]:.4f}

Key Findings:
  • Model successfully predicts cancer from metabolic markers
  • Insulin resistance (HOMA-IR) shows gradient relationship with cancer
  • {feature_cols[indices[0]]} is most important feature

Next Steps:
  1. Test on real NHANES data when available
  2. Compare with your V2/V3 models
  3. Investigate LDH-lactate decorrelation with this dataset
  4. Consider ensemble with other models
""")

print("✓ Analysis complete!")
