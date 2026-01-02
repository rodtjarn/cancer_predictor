"""
Analyze NHANES Data: Insulin Resistance and LDH in Cancer Patients

HYPOTHESIS:
  Cancer patients have insulin resistance (high HOMA-IR) which drives:
  1. High glucose → high lactate (via aerobic glycolysis)
  2. High LDH (as enzyme marker of glycolysis)
  3. BUT: Lactate and LDH become decorrelated due to mitochondrial dysfunction
       or alternative lactate production pathways

TEST:
  Without lactate data, we can test:
  - Do cancer patients have higher HOMA-IR than controls?
  - Does HOMA-IR correlate with LDH levels?
  - Is this correlation stronger/weaker in cancer vs controls?

HOMA-IR (Homeostatic Model Assessment of Insulin Resistance):
  HOMA-IR = (Fasting Insulin [µU/mL] × Fasting Glucose [mg/dL]) / 405
  Or in SI units: = (Fasting Insulin [pmol/L] × Fasting Glucose [mmol/L]) / 22.5

  Normal: < 2.0
  Insulin resistant: > 2.5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NHANES INSULIN RESISTANCE AND LDH ANALYSIS")
print("="*80)

CYCLES = [
    ("2007-2008", "E"),
    ("2009-2010", "F"),
    ("2011-2012", "G"),
    ("2013-2014", "H"),
]

all_data = []

for cycle, suffix in CYCLES:
    print(f"\nProcessing {cycle}...")

    try:
        # Load demographics
        demo_path = Path(f"data/nhanes/DEMO_{suffix}.XPT")
        if not demo_path.exists():
            print(f"  ✗ Demographics file not found")
            continue
        demo = pd.read_sas(demo_path)

        # Load glucose & insulin
        glu_path = Path(f"data/nhanes/GLU_{suffix}.XPT")
        if not glu_path.exists():
            print(f"  ✗ Glucose/Insulin file not found")
            continue
        glu = pd.read_sas(glu_path)

        # For 2013-2014, load separate insulin file
        if cycle == "2013-2014":
            ins_path = Path(f"data/nhanes/INS_{suffix}.XPT")
            if ins_path.exists():
                ins = pd.read_sas(ins_path)
                glu = glu.merge(ins, on='SEQN', how='outer')
            else:
                print(f"  ✗ Insulin file not found for 2013-2014")
                continue

        # Load biochemistry (LDH)
        bio_path = Path(f"data/nhanes/BIOPRO_{suffix}.XPT")
        if not bio_path.exists():
            print(f"  ✗ Biochemistry file not found")
            continue
        bio = pd.read_sas(bio_path)

        # Load medical conditions (cancer history)
        mcq_path = Path(f"data/nhanes/MCQ_{suffix}.XPT")
        if not mcq_path.exists():
            print(f"  ✗ Medical conditions file not found")
            continue
        mcq = pd.read_sas(mcq_path)

        # Merge datasets
        df = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'WTSAF2YR']].copy()
        df = df.merge(glu[['SEQN', 'LBXGLU', 'LBXIN']], on='SEQN', how='left')
        df = df.merge(bio[['SEQN', 'LBXSLDSI']], on='SEQN', how='left')

        # Cancer variables (MCQ220 = "Ever told you had cancer?")
        if 'MCQ220' in mcq.columns:
            df = df.merge(mcq[['SEQN', 'MCQ220']], on='SEQN', how='left')
        else:
            print(f"  ⚠ MCQ220 (cancer) not found in {cycle}")
            continue

        # Rename columns for clarity
        df = df.rename(columns={
            'RIDAGEYR': 'age',
            'RIAGENDR': 'gender',
            'LBXGLU': 'glucose_mg_dL',  # Fasting glucose (mg/dL)
            'LBXIN': 'insulin_uU_mL',   # Fasting insulin (µU/mL)
            'LBXSLDSI': 'ldh_U_L',      # LDH (U/L)
            'MCQ220': 'cancer_history',  # Ever told had cancer (1=Yes, 2=No)
            'WTSAF2YR': 'fasting_weight'  # Fasting subsample weight
        })

        df['cycle'] = cycle
        all_data.append(df)
        print(f"  ✓ Loaded {len(df)} participants")

    except Exception as e:
        print(f"  ✗ Error processing {cycle}: {e}")

if not all_data:
    print("\n✗ No data loaded. Please run download_nhanes_data.py first.")
    exit(1)

# Combine all cycles
combined = pd.concat(all_data, ignore_index=True)
print(f"\n✓ Combined data: {len(combined)} total participants")

# ============================================================================
# DATA CLEANING
# ============================================================================
print("\n" + "-"*80)
print("DATA CLEANING")
print("-"*80)

# Keep only fasting participants with complete data
df = combined.copy()

# Filter for fasting sample (has fasting weight)
df = df[df['fasting_weight'] > 0]
print(f"Fasting participants: {len(df)}")

# Remove missing values
df = df.dropna(subset=['glucose_mg_dL', 'insulin_uU_mL', 'ldh_U_L', 'cancer_history', 'age'])
print(f"Complete data: {len(df)}")

# Calculate HOMA-IR
df['HOMA_IR'] = (df['insulin_uU_mL'] * df['glucose_mg_dL']) / 405
print(f"✓ Calculated HOMA-IR for {len(df)} participants")

# Define cancer status (1=Yes, 2=No in NHANES coding)
df['has_cancer'] = (df['cancer_history'] == 1).astype(int)

# Basic stats
n_cancer = df['has_cancer'].sum()
n_control = len(df) - n_cancer
print(f"\nCancer patients: {n_cancer} ({100*n_cancer/len(df):.1f}%)")
print(f"Controls: {n_control} ({100*n_control/len(df):.1f}%)")

# Age distribution
print(f"\nAge range: {df['age'].min():.0f}-{df['age'].max():.0f} years")
print(f"Mean age (cancer): {df[df['has_cancer']==1]['age'].mean():.1f} years")
print(f"Mean age (control): {df[df['has_cancer']==0]['age'].mean():.1f} years")

# ============================================================================
# HYPOTHESIS TEST 1: Do cancer patients have higher insulin resistance?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS TEST 1: Insulin Resistance in Cancer vs Controls")
print("="*80)

cancer_homa = df[df['has_cancer'] == 1]['HOMA_IR']
control_homa = df[df['has_cancer'] == 0]['HOMA_IR']

print(f"\nHOMA-IR Statistics:")
print(f"  Cancer (n={len(cancer_homa)}):")
print(f"    Mean: {cancer_homa.mean():.2f}")
print(f"    Median: {cancer_homa.median():.2f}")
print(f"    SD: {cancer_homa.std():.2f}")

print(f"\n  Control (n={len(control_homa)}):")
print(f"    Mean: {control_homa.mean():.2f}")
print(f"    Median: {control_homa.median():.2f}")
print(f"    SD: {control_homa.std():.2f}")

# Mann-Whitney U test (non-parametric, better for skewed data)
u_stat, p_value = mannwhitneyu(cancer_homa, control_homa, alternative='two-sided')
print(f"\nMann-Whitney U test: U={u_stat:.1f}, p={p_value:.3e}")

if p_value < 0.05:
    diff = cancer_homa.median() - control_homa.median()
    print(f"✓ SIGNIFICANT: Cancer patients have {'higher' if diff > 0 else 'lower'} HOMA-IR (median diff: {diff:+.2f})")
else:
    print(f"✗ NOT SIGNIFICANT: No difference in insulin resistance")

# Insulin resistance prevalence (HOMA-IR > 2.5)
cancer_ir_pct = 100 * (cancer_homa > 2.5).sum() / len(cancer_homa)
control_ir_pct = 100 * (control_homa > 2.5).sum() / len(control_homa)
print(f"\nInsulin Resistance Prevalence (HOMA-IR > 2.5):")
print(f"  Cancer: {cancer_ir_pct:.1f}%")
print(f"  Control: {control_ir_pct:.1f}%")

# ============================================================================
# HYPOTHESIS TEST 2: Does HOMA-IR correlate with LDH?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS TEST 2: HOMA-IR and LDH Correlation")
print("="*80)

# Overall correlation
r_all, p_all = pearsonr(df['HOMA_IR'], df['ldh_U_L'])
print(f"\nOverall (all participants): r = {r_all:.3f}, p = {p_all:.3e}")

# Cancer patients
cancer_df = df[df['has_cancer'] == 1]
if len(cancer_df) > 2:
    r_cancer, p_cancer = pearsonr(cancer_df['HOMA_IR'], cancer_df['ldh_U_L'])
    print(f"Cancer patients (n={len(cancer_df)}): r = {r_cancer:.3f}, p = {p_cancer:.3e}")

# Controls
control_df = df[df['has_cancer'] == 0]
if len(control_df) > 2:
    r_control, p_control = pearsonr(control_df['HOMA_IR'], control_df['ldh_U_L'])
    print(f"Controls (n={len(control_df)}): r = {r_control:.3f}, p = {p_control:.3e}")

if len(cancer_df) > 2 and len(control_df) > 2:
    print(f"\nDifference: Cancer correlation is {abs(r_cancer - r_control):.3f} {'higher' if r_cancer > r_control else 'lower'} than controls")

# ============================================================================
# HYPOTHESIS TEST 3: LDH levels in cancer vs controls
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS TEST 3: LDH Levels in Cancer vs Controls")
print("="*80)

cancer_ldh = df[df['has_cancer'] == 1]['ldh_U_L']
control_ldh = df[df['has_cancer'] == 0]['ldh_U_L']

print(f"\nLDH Statistics:")
print(f"  Cancer (n={len(cancer_ldh)}):")
print(f"    Mean: {cancer_ldh.mean():.1f} U/L")
print(f"    Median: {cancer_ldh.median():.1f} U/L")

print(f"\n  Control (n={len(control_ldh)}):")
print(f"    Mean: {control_ldh.mean():.1f} U/L")
print(f"    Median: {control_ldh.median():.1f} U/L")

u_stat, p_value = mannwhitneyu(cancer_ldh, control_ldh, alternative='two-sided')
print(f"\nMann-Whitney U test: U={u_stat:.1f}, p={p_value:.3e}")

if p_value < 0.05:
    diff = cancer_ldh.median() - control_ldh.median()
    print(f"✓ SIGNIFICANT: Cancer patients have {'higher' if diff > 0 else 'lower'} LDH (median diff: {diff:+.1f} U/L)")
else:
    print(f"✗ NOT SIGNIFICANT: No difference in LDH levels")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save processed data
output_path = Path("data/nhanes/nhanes_processed_insulin_ldh.csv")
df.to_csv(output_path, index=False)
print(f"✓ Saved processed data to {output_path}")

# Save summary statistics
summary = {
    'n_total': len(df),
    'n_cancer': int(n_cancer),
    'n_control': int(n_control),
    'cancer_homa_mean': float(cancer_homa.mean()),
    'cancer_homa_median': float(cancer_homa.median()),
    'control_homa_mean': float(control_homa.mean()),
    'control_homa_median': float(control_homa.median()),
    'homa_pvalue': float(p_value),
    'cancer_ldh_mean': float(cancer_ldh.mean()),
    'cancer_ldh_median': float(cancer_ldh.median()),
    'control_ldh_mean': float(control_ldh.mean()),
    'control_ldh_median': float(control_ldh.median()),
    'ldh_pvalue': float(p_value),
    'homa_ldh_corr_all': float(r_all),
    'homa_ldh_corr_cancer': float(r_cancer) if len(cancer_df) > 2 else None,
    'homa_ldh_corr_control': float(r_control) if len(control_df) > 2 else None,
}

import json
summary_path = Path("data/nhanes/nhanes_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved summary statistics to {summary_path}")

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND INTERPRETATION")
print("="*80)

print("""
KEY FINDINGS FROM MIMIC-IV DEMO:
  ✓ Healthy controls: LDH-lactate correlation = 0.940 (very strong)
  ✓ Cancer patients: LDH-lactate correlation = 0.009 (essentially zero)
  → MASSIVE CORRELATION BREAKDOWN IN CANCER

NHANES ANALYSIS (partial data - no lactate):
  • Tested insulin resistance (HOMA-IR) in cancer vs controls
  • Tested HOMA-IR correlation with LDH
  • Can't directly test LDH-lactate correlation (no lactate data)

BIOLOGICAL INTERPRETATION:
  If cancer patients have:
    1. Higher HOMA-IR (insulin resistance) → YES/NO from analysis above
    2. Higher LDH → YES/NO from analysis above
    3. HOMA-IR correlates with LDH → YES/NO from analysis above

  Then insulin resistance may drive:
    • Compensatory hyperinsulinemia
    • Increased glucose uptake (despite resistance)
    • Warburg effect (aerobic glycolysis)
    • Lactate production via non-LDH pathways (e.g., LDHA vs LDHB isoforms)
    • → Decorrelation of LDH and lactate

NEXT STEPS:
  1. Review results above
  2. If hypothesis supported, create visualizations
  3. Consider full MIMIC-IV access for complete dataset with lactate
  4. Investigate LDH isoforms (LDHA vs LDHB) in cancer
""")

print("\n✓ Analysis complete!")
