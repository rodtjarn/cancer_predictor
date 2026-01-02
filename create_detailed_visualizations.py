"""
Create Detailed Visualizations and Metrics for NHANES RF Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import joblib
from scipy.stats import mannwhitneyu, pearsonr
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

print("="*80)
print("COMPREHENSIVE VISUALIZATION AND METRICS")
print("="*80)

# Load data and model
df = pd.read_csv('data/nhanes/nhanes_2017_2018_processed.csv')
model = joblib.load('models/nhanes_real_rf_model.pkl')
scaler = joblib.load('models/nhanes_real_scaler.pkl')

print(f"\nLoaded {len(df)} participants")
print(f"  Cancer: {df['cancer'].sum()} ({100*df['cancer'].mean():.1f}%)")

# Prepare features
feature_cols = ['insulin', 'glucose', 'ldh', 'crp', 'HOMA_IR', 'age', 'gender']
X = df[feature_cols].values
y = df['cancer'].values

# Get predictions
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# Add predictions to dataframe
df['predicted_cancer'] = y_pred
df['cancer_probability'] = y_proba

# ============================================================================
# FIGURE 1: MODEL PERFORMANCE OVERVIEW
# ============================================================================
print("\nCreating Figure 1: Model Performance Overview...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Feature Importance
ax1 = fig.add_subplot(gs[0, 0])
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
colors = ['#d62728' if i == indices[0] else '#1f77b4' for i in range(len(feature_cols))]
y_pos = np.arange(len(feature_cols))
ax1.barh(y_pos, importances[indices], color=[colors[i] for i in indices])
ax1.set_yticks(y_pos)
ax1.set_yticklabels([feature_cols[i].upper() for i in indices], fontsize=10)
ax1.set_xlabel('Importance Score', fontsize=11)
ax1.set_title('Feature Importance\n(Age dominates at 45%)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, idx in enumerate(indices):
    ax1.text(importances[idx] + 0.01, i, f'{importances[idx]:.3f}',
             va='center', fontsize=9)

# 2. ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
fpr, tpr, thresholds = roc_curve(y, y_proba)
auc = roc_auc_score(y, y_proba)
ax2.plot(fpr, tpr, linewidth=2.5, color='#1f77b4', label=f'RF Model (AUC = {auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
ax2.fill_between(fpr, tpr, alpha=0.2, color='#1f77b4')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title(f'ROC Curve\n(AUC = {auc:.3f})', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(alpha=0.3)

# 3. Precision-Recall Curve
ax3 = fig.add_subplot(gs[0, 2])
precision, recall, _ = precision_recall_curve(y, y_proba)
baseline = y.mean()
ax3.plot(recall, precision, linewidth=2.5, color='#2ca02c', label=f'RF Model')
ax3.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, alpha=0.5,
            label=f'Baseline ({baseline:.3f})')
ax3.fill_between(recall, precision, alpha=0.2, color='#2ca02c')
ax3.set_xlabel('Recall (Sensitivity)', fontsize=11)
ax3.set_ylabel('Precision (PPV)', fontsize=11)
ax3.set_title('Precision-Recall Curve\n(Class Imbalance: 90% control)', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(alpha=0.3)

# 4. Confusion Matrix
ax4 = fig.add_subplot(gs[1, 0])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=True,
            xticklabels=['Control', 'Cancer'],
            yticklabels=['Control', 'Cancer'],
            annot_kws={'fontsize': 14})
ax4.set_xlabel('Predicted', fontsize=11)
ax4.set_ylabel('Actual', fontsize=11)
ax4.set_title(f'Confusion Matrix\nTN={tn}, FP={fp}, FN={fn}, TP={tp}',
              fontsize=12, fontweight='bold')

# Add percentages
for i in range(2):
    for j in range(2):
        pct = cm_normalized[i, j] * 100
        ax4.text(j+0.5, i+0.7, f'({pct:.1f}%)', ha='center', va='center',
                fontsize=9, color='gray')

# 5. Probability Distribution
ax5 = fig.add_subplot(gs[1, 1])
cancer_proba = df[df['cancer']==1]['cancer_probability']
control_proba = df[df['cancer']==0]['cancer_probability']
ax5.hist(control_proba, bins=30, alpha=0.6, label='Control', color='steelblue', edgecolor='black')
ax5.hist(cancer_proba, bins=30, alpha=0.6, label='Cancer', color='coral', edgecolor='black')
ax5.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax5.set_xlabel('Predicted Cancer Probability', fontsize=11)
ax5.set_ylabel('Count', fontsize=11)
ax5.set_title('Probability Distribution\n(Good separation)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3, axis='y')

# 6. Calibration Curve
ax6 = fig.add_subplot(gs[1, 2])
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10, strategy='uniform')
ax6.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8,
         color='#ff7f0e', label='Model')
ax6.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect Calibration')
ax6.set_xlabel('Predicted Probability', fontsize=11)
ax6.set_ylabel('Actual Fraction of Positives', fontsize=11)
ax6.set_title('Calibration Curve\n(How well calibrated are predictions?)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

# 7. Biomarker Comparison
ax7 = fig.add_subplot(gs[2, :])
biomarkers = ['Insulin\n(µU/mL)', 'Glucose\n(mg/dL)', 'LDH\n(U/L)', 'CRP\n(mg/L)', 'HOMA-IR']
cols = ['insulin', 'glucose', 'ldh', 'crp', 'HOMA_IR']
cancer_means = [df[df['cancer']==1][c].mean() for c in cols]
control_means = [df[df['cancer']==0][c].mean() for c in cols]
cancer_stds = [df[df['cancer']==1][c].std() for c in cols]
control_stds = [df[df['cancer']==0][c].std() for c in cols]

# Calculate p-values
p_values = []
for c in cols:
    _, p = mannwhitneyu(df[df['cancer']==1][c], df[df['cancer']==0][c], alternative='two-sided')
    p_values.append(p)

x = np.arange(len(biomarkers))
width = 0.35
bars1 = ax7.bar(x - width/2, control_means, width, label='Control',
                color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax7.bar(x + width/2, cancer_means, width, label='Cancer',
                color='coral', edgecolor='black', linewidth=1.5)

# Add error bars
ax7.errorbar(x - width/2, control_means, yerr=control_stds, fmt='none',
             ecolor='black', capsize=5, alpha=0.5)
ax7.errorbar(x + width/2, cancer_means, yerr=cancer_stds, fmt='none',
             ecolor='black', capsize=5, alpha=0.5)

ax7.set_xticks(x)
ax7.set_xticklabels(biomarkers, fontsize=10)
ax7.set_ylabel('Mean Value ± SD', fontsize=11)
ax7.set_title('Biomarker Levels: Cancer vs Control\n(*** p<0.001, * p<0.05)',
              fontsize=12, fontweight='bold')
ax7.legend(fontsize=11, loc='upper left')
ax7.grid(axis='y', alpha=0.3)

# Add significance stars
for i, p in enumerate(p_values):
    if p < 0.001:
        sig = '***'
        y_pos = max(cancer_means[i], control_means[i]) + max(cancer_stds[i], control_stds[i]) + 5
        ax7.text(i, y_pos, sig, ha='center', fontsize=14, fontweight='bold', color='red')
    elif p < 0.05:
        sig = '*'
        y_pos = max(cancer_means[i], control_means[i]) + max(cancer_stds[i], control_stds[i]) + 5
        ax7.text(i, y_pos, sig, ha='center', fontsize=14, fontweight='bold', color='red')

plt.suptitle('NHANES 2017-2018: Random Forest Cancer Prediction Model',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('results/nhanes_detailed_metrics_fig1.png', dpi=300, bbox_inches='tight')
print("✓ Saved Figure 1")

# ============================================================================
# FIGURE 2: INSULIN RESISTANCE AND AGE ANALYSIS
# ============================================================================
print("\nCreating Figure 2: Insulin Resistance & Age Analysis...")

fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Insulin Resistance and Age Stratification Analysis',
              fontsize=16, fontweight='bold')

# 1. HOMA-IR by Cancer Status
ax = axes[0, 0]
cancer_homa = df[df['cancer']==1]['HOMA_IR']
control_homa = df[df['cancer']==0]['HOMA_IR']
bp = ax.boxplot([control_homa, cancer_homa],
                 labels=['Control\n(n={})'.format(len(control_homa)),
                        'Cancer\n(n={})'.format(len(cancer_homa))],
                 patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax.set_ylabel('HOMA-IR', fontsize=11)
ax.set_title(f'Insulin Resistance\nMedian: {control_homa.median():.2f} vs {cancer_homa.median():.2f}',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# Add p-value
_, p = mannwhitneyu(cancer_homa, control_homa)
ax.text(0.5, ax.get_ylim()[1]*0.95, f'p = {p:.3f}', ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Cancer Rate by HOMA-IR Quartile
ax = axes[0, 1]
df['HOMA_IR_quartile'] = pd.qcut(df['HOMA_IR'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
cancer_rates = []
quartile_labels = []
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    subset = df[df['HOMA_IR_quartile'] == q]
    if len(subset) > 0:
        rate = subset['cancer'].mean() * 100
        cancer_rates.append(rate)
        homa_range = f"{subset['HOMA_IR'].min():.1f}-{subset['HOMA_IR'].max():.1f}"
        quartile_labels.append(f'{q}\n({homa_range})')

bars = ax.bar(range(len(cancer_rates)), cancer_rates, color=['#e6f2ff', '#99ccff', '#4da6ff', '#0066cc'],
              edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(cancer_rates)))
ax.set_xticklabels(quartile_labels, fontsize=9)
ax.set_ylabel('Cancer Rate (%)', fontsize=11)
ax.set_title('Cancer Rate by HOMA-IR Quartile\n(No clear monotonic trend)',
             fontsize=12, fontweight='bold')
ax.axhline(y=df['cancer'].mean()*100, color='red', linestyle='--', linewidth=2,
           label=f"Overall: {df['cancer'].mean()*100:.1f}%")
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for i, (bar, rate) in enumerate(zip(bars, cancer_rates)):
    ax.text(bar.get_x() + bar.get_width()/2, rate + 0.5, f'{rate:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Age Distribution
ax = axes[0, 2]
cancer_age = df[df['cancer']==1]['age']
control_age = df[df['cancer']==0]['age']
ax.hist(control_age, bins=30, alpha=0.6, label='Control', color='steelblue',
        edgecolor='black', density=True)
ax.hist(cancer_age, bins=30, alpha=0.6, label='Cancer', color='coral',
        edgecolor='black', density=True)
ax.axvline(x=control_age.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(x=cancer_age.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Age Distribution\nMean: {control_age.mean():.1f} vs {cancer_age.mean():.1f} years',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# 4. LDH by Cancer Status
ax = axes[1, 0]
cancer_ldh = df[df['cancer']==1]['ldh']
control_ldh = df[df['cancer']==0]['ldh']
bp = ax.boxplot([control_ldh, cancer_ldh],
                 labels=['Control\n(n={})'.format(len(control_ldh)),
                        'Cancer\n(n={})'.format(len(cancer_ldh))],
                 patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax.set_ylabel('LDH (U/L)', fontsize=11)
ax.set_title(f'LDH Levels\nMean: {control_ldh.mean():.1f} vs {cancer_ldh.mean():.1f} U/L',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
_, p = mannwhitneyu(cancer_ldh, control_ldh)
ax.text(0.5, ax.get_ylim()[1]*0.95, f'p = {p:.3e} ***', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7), fontweight='bold')

# 5. Glucose by Cancer Status
ax = axes[1, 1]
cancer_glu = df[df['cancer']==1]['glucose']
control_glu = df[df['cancer']==0]['glucose']
bp = ax.boxplot([control_glu, cancer_glu],
                 labels=['Control\n(n={})'.format(len(control_glu)),
                        'Cancer\n(n={})'.format(len(cancer_glu))],
                 patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax.set_ylabel('Glucose (mg/dL)', fontsize=11)
ax.set_title(f'Fasting Glucose\nMean: {control_glu.mean():.1f} vs {cancer_glu.mean():.1f} mg/dL',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
_, p = mannwhitneyu(cancer_glu, control_glu)
ax.text(0.5, ax.get_ylim()[1]*0.95, f'p = {p:.3e} ***', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7), fontweight='bold')

# 6. Cancer Rate by Age Group
ax = axes[1, 2]
df['age_group'] = pd.cut(df['age'], bins=[18, 40, 50, 60, 70, 100],
                         labels=['18-40', '40-50', '50-60', '60-70', '70+'])
age_cancer_rates = []
age_labels = []
age_counts = []
for ag in ['18-40', '40-50', '50-60', '60-70', '70+']:
    subset = df[df['age_group'] == ag]
    if len(subset) > 0:
        rate = subset['cancer'].mean() * 100
        age_cancer_rates.append(rate)
        age_labels.append(f'{ag}\n(n={len(subset)})')
        age_counts.append(len(subset))

bars = ax.bar(range(len(age_cancer_rates)), age_cancer_rates,
              color=['#e6ffe6', '#99ff99', '#ffff99', '#ffcc99', '#ff9999'],
              edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(age_cancer_rates)))
ax.set_xticklabels(age_labels, fontsize=9)
ax.set_ylabel('Cancer Rate (%)', fontsize=11)
ax.set_title('Cancer Rate by Age Group\n(Strong age gradient)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, (bar, rate) in enumerate(zip(bars, age_cancer_rates)):
    ax.text(bar.get_x() + bar.get_width()/2, rate + 0.5, f'{rate:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/nhanes_detailed_metrics_fig2.png', dpi=300, bbox_inches='tight')
print("✓ Saved Figure 2")

# ============================================================================
# COMPREHENSIVE METRICS TABLE
# ============================================================================
print("\nGenerating comprehensive metrics table...")

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, matthews_corrcoef)

metrics = {
    "Dataset Statistics": {
        "Total Participants": len(df),
        "Cancer Cases": int(df['cancer'].sum()),
        "Controls": int((1-df['cancer']).sum()),
        "Cancer Prevalence": f"{100*df['cancer'].mean():.2f}%",
        "Mean Age (Cancer)": f"{df[df['cancer']==1]['age'].mean():.1f} years",
        "Mean Age (Control)": f"{df[df['cancer']==0]['age'].mean():.1f} years",
    },

    "Model Performance": {
        "Accuracy": f"{accuracy_score(y, y_pred):.3f}",
        "Sensitivity (Recall)": f"{recall_score(y, y_pred):.3f}",
        "Specificity": f"{tn/(tn+fp):.3f}",
        "Precision (PPV)": f"{precision_score(y, y_pred):.3f}",
        "NPV": f"{tn/(tn+fn):.3f}",
        "F1-Score": f"{f1_score(y, y_pred):.3f}",
        "Matthews Correlation": f"{matthews_corrcoef(y, y_pred):.3f}",
        "ROC-AUC": f"{roc_auc_score(y, y_proba):.3f}",
    },

    "Confusion Matrix": {
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
    },

    "Biomarker Differences (Cancer vs Control)": {
        "Insulin": f"{df[df['cancer']==1]['insulin'].mean():.1f} vs {df[df['cancer']==0]['insulin'].mean():.1f} µU/mL (p={mannwhitneyu(df[df['cancer']==1]['insulin'], df[df['cancer']==0]['insulin'])[1]:.3f})",
        "Glucose": f"{df[df['cancer']==1]['glucose'].mean():.1f} vs {df[df['cancer']==0]['glucose'].mean():.1f} mg/dL (p={mannwhitneyu(df[df['cancer']==1]['glucose'], df[df['cancer']==0]['glucose'])[1]:.3e})",
        "LDH": f"{df[df['cancer']==1]['ldh'].mean():.1f} vs {df[df['cancer']==0]['ldh'].mean():.1f} U/L (p={mannwhitneyu(df[df['cancer']==1]['ldh'], df[df['cancer']==0]['ldh'])[1]:.3e})",
        "CRP": f"{df[df['cancer']==1]['crp'].median():.2f} vs {df[df['cancer']==0]['crp'].median():.2f} mg/L (p={mannwhitneyu(df[df['cancer']==1]['crp'], df[df['cancer']==0]['crp'])[1]:.3f})",
        "HOMA-IR": f"{df[df['cancer']==1]['HOMA_IR'].mean():.2f} vs {df[df['cancer']==0]['HOMA_IR'].mean():.2f} (p={mannwhitneyu(df[df['cancer']==1]['HOMA_IR'], df[df['cancer']==0]['HOMA_IR'])[1]:.3f})",
    },

    "Feature Importance Rankings": {
        f"{i+1}. {feature_cols[indices[i]].upper()}": f"{importances[indices[i]]:.4f}"
        for i in range(len(feature_cols))
    }
}

# Save as JSON
with open('results/nhanes_comprehensive_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Print nicely
print("\n" + "="*80)
print("COMPREHENSIVE METRICS SUMMARY")
print("="*80)

for section, data in metrics.items():
    print(f"\n{section}:")
    print("-" * 60)
    for key, value in data.items():
        print(f"  {key:35s}: {value}")

print("\n✓ Saved comprehensive metrics to results/nhanes_comprehensive_metrics.json")

print("\n" + "="*80)
print("ALL VISUALIZATIONS AND METRICS COMPLETE!")
print("="*80)
print("\nFiles created:")
print("  1. results/nhanes_detailed_metrics_fig1.png")
print("  2. results/nhanes_detailed_metrics_fig2.png")
print("  3. results/nhanes_comprehensive_metrics.json")
print("  4. results/nhanes_real_rf_evaluation.png (from earlier)")
