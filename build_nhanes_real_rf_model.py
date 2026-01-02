"""
Build Random Forest Model on REAL NHANES 2017-2018 Data

Features:
- Fasting Insulin (LBXIN - µU/mL)
- Fasting Glucose (LBXGLU - mg/dL)
- LDH (LBXSLDSI - U/L)
- CRP (LBXHSCRP - mg/L)
- HOMA-IR (calculated)
- Age
- Gender

Target: Cancer history (MCQ220)
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
print("RANDOM FOREST MODEL: REAL NHANES 2017-2018 DATA")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading NHANES data files...")

# Demographics
demo = pd.read_sas('/Users/per/work/claude/cancer_predictor_package/data/nhanes/DEMO_J.XPT')
print(f"  Demographics: {len(demo)} participants")

# Glucose & Insulin
glu = pd.read_sas('/Users/per/work/claude/cancer_predictor_package/data/nhanes/GLU_J.XPT')
ins = pd.read_sas('/Users/per/work/claude/cancer_predictor_package/data/nhanes/INS_J.XPT')
print(f"  Glucose: {len(glu)}, Insulin: {len(ins)}")

# Biochemistry (LDH)
biopro = pd.read_sas('/Users/per/work/claude/cancer_predictor_package/data/nhanes/BIOPRO_J.XPT')
print(f"  Biochemistry: {len(biopro)}")

# CRP
crp = pd.read_sas('/Users/per/work/claude/cancer_predictor_package/data/nhanes/HSCRP_J.XPT')
print(f"  CRP: {len(crp)}")

# Medical Conditions (cancer)
mcq = pd.read_sas('/Users/per/work/claude/cancer_predictor_package/data/nhanes/MCQ_J.XPT')
print(f"  Medical Conditions: {len(mcq)}")

# Fasting status
fastqx = pd.read_sas('/Users/per/work/claude/cancer_predictor_package/data/nhanes/FASTQX_J.XPT')
print(f"  Fasting questionnaire: {len(fastqx)}")

# ============================================================================
# MERGE DATASETS
# ============================================================================
print("\nMerging datasets...")

# Start with demographics
df = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()
df = df.rename(columns={
    'SEQN': 'id',
    'RIDAGEYR': 'age',
    'RIAGENDR': 'gender'
})

# Merge glucose
glu_vars = ['SEQN', 'LBXGLU'] if 'LBXGLU' in glu.columns else ['SEQN']
if len(glu_vars) > 1:
    df = df.merge(glu[glu_vars].rename(columns={'SEQN': 'id', 'LBXGLU': 'glucose'}),
                  on='id', how='left')

# Merge insulin
ins_vars = ['SEQN', 'LBXIN'] if 'LBXIN' in ins.columns else ['SEQN']
if len(ins_vars) > 1:
    df = df.merge(ins[ins_vars].rename(columns={'SEQN': 'id', 'LBXIN': 'insulin'}),
                  on='id', how='left')

# Merge LDH (variable name varies by cycle)
ldh_var = None
for var in ['LBXSLDSI', 'LBXLD', 'LBDSLDSI']:
    if var in biopro.columns:
        ldh_var = var
        break

if ldh_var:
    df = df.merge(biopro[['SEQN', ldh_var]].rename(columns={'SEQN': 'id', ldh_var: 'ldh'}),
                  on='id', how='left')
    print(f"  Using LDH variable: {ldh_var}")
else:
    print("  WARNING: LDH variable not found!")
    df['ldh'] = np.nan

# Merge CRP
crp_var = None
for var in ['LBXHSCRP', 'LBXCRP', 'LBDHSCRP']:
    if var in crp.columns:
        crp_var = var
        break

if crp_var:
    df = df.merge(crp[['SEQN', crp_var]].rename(columns={'SEQN': 'id', crp_var: 'crp'}),
                  on='id', how='left')
    print(f"  Using CRP variable: {crp_var}")
else:
    print("  WARNING: CRP variable not found!")
    df['crp'] = np.nan

# Merge cancer history (MCQ220 = "Ever told had cancer?")
mcq_var = 'MCQ220'
if mcq_var in mcq.columns:
    df = df.merge(mcq[['SEQN', mcq_var]].rename(columns={'SEQN': 'id', mcq_var: 'cancer_raw'}),
                  on='id', how='left')
else:
    print(f"  WARNING: {mcq_var} not found!")
    df['cancer_raw'] = np.nan

# Merge fasting status
fast_var = 'PHAFSTHR'  # Fasting hours
if fast_var in fastqx.columns:
    df = df.merge(fastqx[['SEQN', fast_var]].rename(columns={'SEQN': 'id', fast_var: 'fasting_hours'}),
                  on='id', how='left')

print(f"\nMerged dataset: {len(df)} participants")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

# Filter for fasting participants (>=8 hours)
if 'fasting_hours' in df.columns:
    df_fasting = df[df['fasting_hours'] >= 8].copy()
    print(f"\nFasting participants (>=8 hrs): {len(df_fasting)}")
else:
    df_fasting = df.copy()
    print(f"\nWarning: No fasting info, using all participants: {len(df_fasting)}")

# Convert cancer variable (1=Yes, 2=No in NHANES coding)
if 'cancer_raw' in df_fasting.columns:
    df_fasting['cancer'] = (df_fasting['cancer_raw'] == 1).astype(int)
else:
    print("ERROR: No cancer variable!")
    df_fasting['cancer'] = 0

# Remove missing values
initial_count = len(df_fasting)
df_clean = df_fasting.dropna(subset=['glucose', 'insulin', 'ldh', 'crp', 'cancer', 'age']).copy()
print(f"After removing missing values: {len(df_clean)} ({100*len(df_clean)/initial_count:.1f}% retained)")

# Calculate HOMA-IR
df_clean['HOMA_IR'] = (df_clean['insulin'] * df_clean['glucose']) / 405

# Remove outliers (extreme values that are likely data errors)
df_clean = df_clean[
    (df_clean['glucose'] > 40) & (df_clean['glucose'] < 400) &
    (df_clean['insulin'] > 0.5) & (df_clean['insulin'] < 200) &
    (df_clean['ldh'] > 50) & (df_clean['ldh'] < 1000) &
    (df_clean['crp'] >= 0) & (df_clean['crp'] < 100) &
    (df_clean['age'] >= 18)
].copy()

print(f"After outlier removal: {len(df_clean)}")

# Cancer statistics
n_cancer = df_clean['cancer'].sum()
n_control = len(df_clean) - n_cancer
cancer_pct = 100 * n_cancer / len(df_clean)

print(f"\nFinal dataset:")
print(f"  Total: {len(df_clean)}")
print(f"  Cancer: {n_cancer} ({cancer_pct:.1f}%)")
print(f"  Controls: {n_control} ({100-cancer_pct:.1f}%)")

if n_cancer < 10:
    print("\n⚠️ WARNING: Very few cancer cases - model may not be reliable!")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

for label, group_val in [("Controls", 0), ("Cancer", 1)]:
    subset = df_clean[df_clean['cancer'] == group_val]
    print(f"\n{label} (n={len(subset)}):")
    print(f"  Fasting Insulin: {subset['insulin'].mean():.1f} ± {subset['insulin'].std():.1f} µU/mL")
    print(f"  Fasting Glucose: {subset['glucose'].mean():.1f} ± {subset['glucose'].std():.1f} mg/dL")
    print(f"  LDH: {subset['ldh'].mean():.1f} ± {subset['ldh'].std():.1f} U/L")
    print(f"  CRP: {subset['crp'].median():.2f} (median) mg/L")
    print(f"  HOMA-IR: {subset['HOMA_IR'].mean():.2f} ± {subset['HOMA_IR'].std():.2f}")
    print(f"  Age: {subset['age'].mean():.1f} ± {subset['age'].std():.1f} years")
    print(f"  Gender (% Male): {100*(subset['gender']==1).mean():.1f}%")

# Statistical tests
from scipy.stats import mannwhitneyu

cancer_data = df_clean[df_clean['cancer'] == 1]
control_data = df_clean[df_clean['cancer'] == 0]

print("\n" + "-"*80)
print("STATISTICAL TESTS (Mann-Whitney U)")
print("-"*80)

for var in ['insulin', 'glucose', 'ldh', 'crp', 'HOMA_IR']:
    if var in df_clean.columns:
        u_stat, p_val = mannwhitneyu(cancer_data[var], control_data[var], alternative='two-sided')
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{var:15s}: p = {p_val:.3e} {sig}")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\n" + "="*80)
print("PREPARING FEATURES FOR MODEL")
print("="*80)

feature_cols = ['insulin', 'glucose', 'ldh', 'crp', 'HOMA_IR', 'age', 'gender']
X = df_clean[feature_cols].values
y = df_clean['cancer'].values

print(f"\nFeatures: {feature_cols}")
print(f"X shape: {X.shape}")
print(f"y distribution: {np.bincount(y.astype(int))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(y_train)} ({y_train.sum()} cancer)")
print(f"Test set: {len(y_test)} ({y_test.sum()} cancer)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# TRAIN RANDOM FOREST
# ============================================================================
print("\n" + "="*80)
print("TRAINING RANDOM FOREST")
print("="*80)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\nTraining...")
rf.fit(X_train_scaled, y_train)
print("✓ Training complete")

# Cross-validation
print("\nCross-validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f"ROC-AUC scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "="*80)
print("TEST SET EVALUATION")
print("="*80)

y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Control', 'Cancer'], zero_division=0))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print("\nPerformance Metrics:")
print(f"  Sensitivity: {sensitivity:.3f}")
print(f"  Specificity: {specificity:.3f}")
print(f"  PPV: {ppv:.3f}")
print(f"  NPV: {npv:.3f}")

if len(np.unique(y_test)) > 1:
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  Average Precision: {ap:.3f}")
else:
    print("  ROC-AUC: N/A (only one class in test set)")
    auc = 0
    ap = 0

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
    print(f"  {i+1}. {feature_cols[idx]:15s}: {importances[idx]:.4f}")

# ============================================================================
# INSULIN RESISTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("INSULIN RESISTANCE ANALYSIS")
print("="*80)

df_clean['HOMA_IR_quartile'] = pd.qcut(df_clean['HOMA_IR'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

print("\nCancer Rate by HOMA-IR Quartile:")
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    subset = df_clean[df_clean['HOMA_IR_quartile'] == q]
    if len(subset) > 0:
        cancer_rate = subset['cancer'].mean()
        homa_range = f"{subset['HOMA_IR'].min():.1f}-{subset['HOMA_IR'].max():.1f}"
        print(f"  {q} (HOMA-IR {homa_range}): {cancer_rate*100:.1f}% cancer (n={len(subset)})")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save processed data
df_clean.to_csv('data/nhanes/nhanes_2017_2018_processed.csv', index=False)
print("✓ Saved processed data")

# Save model
joblib.dump(rf, 'models/nhanes_real_rf_model.pkl')
joblib.dump(scaler, 'models/nhanes_real_scaler.pkl')
print("✓ Saved model and scaler")

# Save results
results = {
    'n_total': len(df_clean),
    'n_cancer': int(n_cancer),
    'n_control': int(n_control),
    'cancer_pct': float(cancer_pct),
    'test_sensitivity': float(sensitivity),
    'test_specificity': float(specificity),
    'test_ppv': float(ppv),
    'test_npv': float(npv),
    'test_roc_auc': float(auc) if auc > 0 else None,
    'cv_auc_mean': float(cv_scores.mean()),
    'cv_auc_std': float(cv_scores.std()),
    'feature_importance': {
        feature_cols[i]: float(importances[i])
        for i in range(len(feature_cols))
    }
}

import json
with open('results/nhanes_real_rf_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Saved results JSON")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nCreating visualizations...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Feature Importance
ax1 = fig.add_subplot(gs[0, 0])
y_pos = np.arange(len(feature_cols))
ax1.barh(y_pos, importances[indices], color='steelblue')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([feature_cols[i] for i in indices])
ax1.set_xlabel('Importance')
ax1.set_title('Feature Importance')
ax1.grid(axis='x', alpha=0.3)

# ROC Curve
if len(np.unique(y_test)) > 1:
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)

# Confusion Matrix
ax3 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Control', 'Cancer'],
            yticklabels=['Control', 'Cancer'])
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
ax3.set_title('Confusion Matrix')

# HOMA-IR by Cancer Status
ax4 = fig.add_subplot(gs[1, 1])
cancer_homa = df_clean[df_clean['cancer'] == 1]['HOMA_IR']
control_homa = df_clean[df_clean['cancer'] == 0]['HOMA_IR']
ax4.boxplot([control_homa, cancer_homa], labels=['Control', 'Cancer'])
ax4.set_ylabel('HOMA-IR')
ax4.set_title('Insulin Resistance by Cancer Status')
ax4.grid(axis='y', alpha=0.3)

# Biomarker comparison
ax5 = fig.add_subplot(gs[2, :])
biomarkers = ['insulin', 'glucose', 'ldh', 'crp']
cancer_means = [df_clean[df_clean['cancer']==1][b].mean() for b in biomarkers]
control_means = [df_clean[df_clean['cancer']==0][b].mean() for b in biomarkers]

x = np.arange(len(biomarkers))
width = 0.35
ax5.bar(x - width/2, control_means, width, label='Control', color='steelblue')
ax5.bar(x + width/2, cancer_means, width, label='Cancer', color='coral')
ax5.set_xticks(x)
ax5.set_xticklabels([b.upper() for b in biomarkers])
ax5.set_ylabel('Mean Value')
ax5.set_title('Biomarker Levels: Cancer vs Control')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

plt.savefig('results/nhanes_real_rf_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualizations")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"""
Dataset: REAL NHANES 2017-2018
Participants: {len(df_clean)}
Cancer cases: {n_cancer} ({cancer_pct:.1f}%)

Model Performance:
  ROC-AUC (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}
  Sensitivity: {sensitivity:.3f}
  Specificity: {specificity:.3f}

Top 3 Features:
  1. {feature_cols[indices[0]]}: {importances[indices[0]]:.4f}
  2. {feature_cols[indices[1]]}: {importances[indices[1]]:.4f}
  3. {feature_cols[indices[2]]}: {importances[indices[2]]:.4f}
""")
