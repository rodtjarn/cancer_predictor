"""
Test Glucose-LDH Correlation in Cancer vs Controls

Following the LDH-lactate decorrelation finding (r=0.94 → 0.009),
test if glucose-LDH shows similar decorrelation in cancer patients.

Hypothesis: Cancer disrupts normal metabolic coupling between glucose and LDH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

print("="*80)
print("GLUCOSE-LDH CORRELATION ANALYSIS: CANCER VS CONTROLS")
print("="*80)

# ============================================================================
# LOAD NHANES DATA
# ============================================================================
print("\nLoading NHANES 2017-2018 data...")
df = pd.read_csv('data/nhanes/nhanes_2017_2018_processed.csv')
print(f"✓ Loaded {len(df)} participants")

# Separate groups
cancer = df[df['cancer'] == 1].copy()
control = df[df['cancer'] == 0].copy()

print(f"  Cancer: {len(cancer)}")
print(f"  Control: {len(control)}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CORRELATION ANALYSIS: GLUCOSE vs LDH")
print("="*80)

# Overall correlation
r_all, p_all = pearsonr(df['glucose'], df['ldh'])
rho_all, _ = spearmanr(df['glucose'], df['ldh'])

print(f"\nOverall (all participants, n={len(df)}):")
print(f"  Pearson r  = {r_all:+.3f}, p = {p_all:.3e}")
print(f"  Spearman ρ = {rho_all:+.3f}")

# Controls
r_control, p_control = pearsonr(control['glucose'], control['ldh'])
rho_control, _ = spearmanr(control['glucose'], control['ldh'])

print(f"\nHealthy Controls (n={len(control)}):")
print(f"  Pearson r  = {r_control:+.3f}, p = {p_control:.3e}")
print(f"  Spearman ρ = {rho_control:+.3f}")

# Cancer patients
r_cancer, p_cancer = pearsonr(cancer['glucose'], cancer['ldh'])
rho_cancer, _ = spearmanr(cancer['glucose'], cancer['ldh'])

print(f"\nCancer Patients (n={len(cancer)}):")
print(f"  Pearson r  = {r_cancer:+.3f}, p = {p_cancer:.3e}")
print(f"  Spearman ρ = {rho_cancer:+.3f}")

# Difference
delta_r = r_cancer - r_control
print(f"\n" + "-"*80)
print(f"CORRELATION DIFFERENCE:")
print(f"  Δr = {delta_r:+.3f}")
print(f"  Cancer correlation is {abs(delta_r):.3f} {'LOWER' if delta_r < 0 else 'HIGHER'} than controls")

if abs(delta_r) > 0.1:
    if delta_r < 0:
        print(f"  ⚠️  DECORRELATION DETECTED in cancer patients!")
    else:
        print(f"  ⚠️  STRONGER correlation in cancer patients!")
else:
    print(f"  ✓ Similar correlation in both groups")

# ============================================================================
# COMPARE WITH LDH-LACTATE FINDING
# ============================================================================
print("\n" + "="*80)
print("COMPARISON WITH LDH-LACTATE DECORRELATION")
print("="*80)

print(f"\nMIMIC-IV (LDH-Lactate):")
print(f"  Healthy:  r = +0.940 (very strong positive)")
print(f"  Cancer:   r = +0.009 (essentially zero)")
print(f"  Δr = -0.931 (MASSIVE DECORRELATION)")

print(f"\nNHANES (Glucose-LDH):")
print(f"  Healthy:  r = {r_control:+.3f}")
print(f"  Cancer:   r = {r_cancer:+.3f}")
print(f"  Δr = {delta_r:+.3f}")

if abs(delta_r) > 0.1:
    if delta_r < 0:
        print(f"\n✓ SIMILAR PATTERN: Decorrelation in cancer!")
    else:
        print(f"\n✗ OPPOSITE PATTERN: Stronger correlation in cancer")
else:
    print(f"\n✗ DIFFERENT PATTERN: No decorrelation observed")

# ============================================================================
# STRATIFIED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STRATIFIED ANALYSIS")
print("="*80)

# By age groups
print("\nBy Age Group:")
print("-" * 60)
age_groups = [(18, 50, "Young"), (50, 65, "Middle"), (65, 100, "Elderly")]

for age_min, age_max, label in age_groups:
    subset = df[(df['age'] >= age_min) & (df['age'] < age_max)]
    subset_control = subset[subset['cancer'] == 0]
    subset_cancer = subset[subset['cancer'] == 1]

    if len(subset_control) > 10 and len(subset_cancer) > 10:
        r_ctrl, _ = pearsonr(subset_control['glucose'], subset_control['ldh'])
        r_canc, _ = pearsonr(subset_cancer['glucose'], subset_cancer['ldh'])
        print(f"  {label} ({age_min}-{age_max}y):")
        print(f"    Control: r={r_ctrl:+.3f} (n={len(subset_control)})")
        print(f"    Cancer:  r={r_canc:+.3f} (n={len(subset_cancer)})")
        print(f"    Δr = {r_canc - r_ctrl:+.3f}")

# By glucose quartiles
print("\nBy Glucose Quartile (Controls):")
print("-" * 60)
control['glucose_quartile'] = pd.qcut(control['glucose'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    subset = control[control['glucose_quartile'] == q]
    if len(subset) > 10:
        r, p = pearsonr(subset['glucose'], subset['ldh'])
        glu_range = f"{subset['glucose'].min():.0f}-{subset['glucose'].max():.0f}"
        print(f"  {q} (glucose {glu_range} mg/dL): r={r:+.3f}, n={len(subset)}")

print("\nBy Glucose Quartile (Cancer):")
print("-" * 60)
cancer['glucose_quartile'] = pd.qcut(cancer['glucose'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

for q in cancer['glucose_quartile'].unique():
    subset = cancer[cancer['glucose_quartile'] == q]
    if len(subset) > 10:
        r, p = pearsonr(subset['glucose'], subset['ldh'])
        glu_range = f"{subset['glucose'].min():.0f}-{subset['glucose'].max():.0f}"
        print(f"  {q} (glucose {glu_range} mg/dL): r={r:+.3f}, n={len(subset)}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Scatter: Controls
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(control['glucose'], control['ldh'], alpha=0.3, s=20, color='steelblue', label='Data')
# Fit line
z = np.polyfit(control['glucose'], control['ldh'], 1)
p = np.poly1d(z)
x_line = np.linspace(control['glucose'].min(), control['glucose'].max(), 100)
ax1.plot(x_line, p(x_line), "r-", linewidth=2.5, label=f'Linear fit (r={r_control:.3f})')
ax1.set_xlabel('Glucose (mg/dL)', fontsize=11)
ax1.set_ylabel('LDH (U/L)', fontsize=11)
ax1.set_title(f'Controls (n={len(control)})\nPearson r = {r_control:+.3f}, p = {p_control:.3e}',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 2. Scatter: Cancer
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(cancer['glucose'], cancer['ldh'], alpha=0.5, s=30, color='coral', label='Data')
# Fit line
z = np.polyfit(cancer['glucose'], cancer['ldh'], 1)
p = np.poly1d(z)
x_line = np.linspace(cancer['glucose'].min(), cancer['glucose'].max(), 100)
ax2.plot(x_line, p(x_line), "r-", linewidth=2.5, label=f'Linear fit (r={r_cancer:.3f})')
ax2.set_xlabel('Glucose (mg/dL)', fontsize=11)
ax2.set_ylabel('LDH (U/L)', fontsize=11)
ax2.set_title(f'Cancer (n={len(cancer)})\nPearson r = {r_cancer:+.3f}, p = {p_cancer:.3e}',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 3. Comparison bar chart
ax3 = fig.add_subplot(gs[0, 2])
correlations = [r_control, r_cancer]
colors = ['steelblue', 'coral']
bars = ax3.bar(['Control', 'Cancer'], correlations, color=colors, edgecolor='black', linewidth=2)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('Pearson Correlation (r)', fontsize=11)
ax3.set_title(f'Glucose-LDH Correlation Comparison\nΔr = {delta_r:+.3f}',
              fontsize=12, fontweight='bold')
ax3.set_ylim(-0.1, max(correlations) + 0.1)
ax3.grid(axis='y', alpha=0.3)

# Add values on bars
for bar, r in zip(bars, correlations):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f'{r:+.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 4. Hexbin: Controls (density)
ax4 = fig.add_subplot(gs[1, 0])
hb = ax4.hexbin(control['glucose'], control['ldh'], gridsize=30, cmap='Blues', mincnt=1)
ax4.set_xlabel('Glucose (mg/dL)', fontsize=11)
ax4.set_ylabel('LDH (U/L)', fontsize=11)
ax4.set_title(f'Controls - Density Plot\n(Darker = more data points)',
              fontsize=12, fontweight='bold')
plt.colorbar(hb, ax=ax4, label='Count')

# 5. Hexbin: Cancer (density)
ax5 = fig.add_subplot(gs[1, 1])
hb = ax5.hexbin(cancer['glucose'], cancer['ldh'], gridsize=20, cmap='Reds', mincnt=1)
ax5.set_xlabel('Glucose (mg/dL)', fontsize=11)
ax5.set_ylabel('LDH (U/L)', fontsize=11)
ax5.set_title(f'Cancer - Density Plot\n(Darker = more data points)',
              fontsize=12, fontweight='bold')
plt.colorbar(hb, ax=ax5, label='Count')

# 6. Comparison with MIMIC LDH-Lactate
ax6 = fig.add_subplot(gs[1, 2])
datasets = ['MIMIC\nLDH-Lactate\nControl', 'MIMIC\nLDH-Lactate\nCancer',
            'NHANES\nGlucose-LDH\nControl', 'NHANES\nGlucose-LDH\nCancer']
corrs = [0.940, 0.009, r_control, r_cancer]
colors_comp = ['#4CAF50', '#FF5252', 'steelblue', 'coral']
bars = ax6.bar(range(len(datasets)), corrs, color=colors_comp, edgecolor='black', linewidth=2)
ax6.set_xticks(range(len(datasets)))
ax6.set_xticklabels(datasets, fontsize=9)
ax6.set_ylabel('Pearson Correlation (r)', fontsize=11)
ax6.set_title('Comparison: LDH-Lactate vs Glucose-LDH\n(Do both show decorrelation?)',
              fontsize=12, fontweight='bold')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax6.grid(axis='y', alpha=0.3)

for i, (bar, r) in enumerate(zip(bars, corrs)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2, height + 0.03,
             f'{r:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 7. Correlation by age group
ax7 = fig.add_subplot(gs[2, 0])
age_bins = [(18, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
age_labels = ['18-40', '40-50', '50-60', '60-70', '70+']
corr_control_age = []
corr_cancer_age = []

for age_min, age_max in age_bins:
    subset_ctrl = control[(control['age'] >= age_min) & (control['age'] < age_max)]
    subset_canc = cancer[(cancer['age'] >= age_min) & (cancer['age'] < age_max)]

    if len(subset_ctrl) > 10:
        r_ctrl, _ = pearsonr(subset_ctrl['glucose'], subset_ctrl['ldh'])
        corr_control_age.append(r_ctrl)
    else:
        corr_control_age.append(np.nan)

    if len(subset_canc) > 10:
        r_canc, _ = pearsonr(subset_canc['glucose'], subset_canc['ldh'])
        corr_cancer_age.append(r_canc)
    else:
        corr_cancer_age.append(np.nan)

x = np.arange(len(age_labels))
width = 0.35
ax7.bar(x - width/2, corr_control_age, width, label='Control', color='steelblue', edgecolor='black')
ax7.bar(x + width/2, corr_cancer_age, width, label='Cancer', color='coral', edgecolor='black')
ax7.set_xticks(x)
ax7.set_xticklabels(age_labels, fontsize=10)
ax7.set_ylabel('Pearson Correlation (r)', fontsize=11)
ax7.set_xlabel('Age Group', fontsize=11)
ax7.set_title('Glucose-LDH Correlation by Age\n(Does age affect correlation?)',
              fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax7.grid(axis='y', alpha=0.3)

# 8. Residuals: Controls
ax8 = fig.add_subplot(gs[2, 1])
z = np.polyfit(control['glucose'], control['ldh'], 1)
p = np.poly1d(z)
predicted = p(control['glucose'])
residuals = control['ldh'] - predicted
ax8.scatter(control['glucose'], residuals, alpha=0.3, s=20, color='steelblue')
ax8.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax8.set_xlabel('Glucose (mg/dL)', fontsize=11)
ax8.set_ylabel('Residuals (U/L)', fontsize=11)
ax8.set_title('Controls - Residual Plot\n(Random scatter = good fit)',
              fontsize=12, fontweight='bold')
ax8.grid(alpha=0.3)

# 9. Residuals: Cancer
ax9 = fig.add_subplot(gs[2, 2])
z = np.polyfit(cancer['glucose'], cancer['ldh'], 1)
p = np.poly1d(z)
predicted = p(cancer['glucose'])
residuals = cancer['ldh'] - predicted
ax9.scatter(cancer['glucose'], residuals, alpha=0.5, s=30, color='coral')
ax9.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax9.set_xlabel('Glucose (mg/dL)', fontsize=11)
ax9.set_ylabel('Residuals (U/L)', fontsize=11)
ax9.set_title('Cancer - Residual Plot\n(Pattern suggests non-linearity?)',
              fontsize=12, fontweight='bold')
ax9.grid(alpha=0.3)

plt.suptitle('Glucose-LDH Correlation Analysis: Cancer vs Controls',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('results/glucose_ldh_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization")

# ============================================================================
# SAVE RESULTS
# ============================================================================
results = {
    "overall": {
        "n": len(df),
        "pearson_r": float(r_all),
        "p_value": float(p_all),
        "spearman_rho": float(rho_all)
    },
    "controls": {
        "n": len(control),
        "pearson_r": float(r_control),
        "p_value": float(p_control),
        "spearman_rho": float(rho_control)
    },
    "cancer": {
        "n": len(cancer),
        "pearson_r": float(r_cancer),
        "p_value": float(p_cancer),
        "spearman_rho": float(rho_cancer)
    },
    "difference": {
        "delta_r": float(delta_r),
        "interpretation": "decorrelation" if delta_r < -0.1 else "stronger_correlation" if delta_r > 0.1 else "similar"
    },
    "comparison_with_mimic": {
        "mimic_ldh_lactate_control": 0.940,
        "mimic_ldh_lactate_cancer": 0.009,
        "mimic_delta_r": -0.931,
        "nhanes_glucose_ldh_control": float(r_control),
        "nhanes_glucose_ldh_cancer": float(r_cancer),
        "nhanes_delta_r": float(delta_r)
    }
}

import json
with open('results/glucose_ldh_correlation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Saved results JSON")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
GLUCOSE-LDH CORRELATION FINDINGS:

Controls (n={len(control)}):
  • Pearson r = {r_control:+.3f} (p = {p_control:.3e})
  • {'Weak' if abs(r_control) < 0.3 else 'Moderate' if abs(r_control) < 0.7 else 'Strong'} {'positive' if r_control > 0 else 'negative'} correlation

Cancer (n={len(cancer)}):
  • Pearson r = {r_cancer:+.3f} (p = {p_cancer:.3e})
  • {'Weak' if abs(r_cancer) < 0.3 else 'Moderate' if abs(r_cancer) < 0.7 else 'Strong'} {'positive' if r_cancer > 0 else 'negative'} correlation

Difference:
  • Δr = {delta_r:+.3f}
  • Cancer correlation is {abs(delta_r):.3f} {'LOWER' if delta_r < 0 else 'HIGHER'} than controls
""")

if abs(delta_r) > 0.1:
    if delta_r < 0:
        print("⚠️  DECORRELATION DETECTED!")
        print("    Similar to LDH-lactate pattern (r=0.94 → 0.009)")
        print("    Suggests disrupted glucose-LDH coupling in cancer")
    else:
        print("⚠️  STRONGER CORRELATION in cancer!")
        print("    Opposite to LDH-lactate pattern")
        print("    Suggests tighter glucose-LDH coupling in cancer")
else:
    print("✓ No significant decorrelation")
    print("  Different from LDH-lactate pattern")

print(f"""
COMPARISON WITH MIMIC-IV LDH-LACTATE:
  • MIMIC LDH-Lactate:  Control r=+0.940 → Cancer r=+0.009 (Δr=-0.931)
  • NHANES Glucose-LDH: Control r={r_control:+.3f} → Cancer r={r_cancer:+.3f} (Δr={delta_r:+.3f})

INTERPRETATION:
  {'• Both show decorrelation - suggests common metabolic disruption' if delta_r < -0.1
   else '• Different patterns - glucose-LDH coupling may be different from lactate-LDH'}
""")

print("✓ Analysis complete!")
print("\nFiles created:")
print("  • results/glucose_ldh_correlation_analysis.png")
print("  • results/glucose_ldh_correlation_results.json")
