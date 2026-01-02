"""
Analyze Cancer Fermentation Hypothesis

HYPOTHESIS (User's):
Cancer cells ferment both:
1. Glucose → Lactate (via LDH) - Warburg effect
2. Glutamine → Succinate (via reductive carboxylation)

PREDICTION: Strong LDH/Lactate/Succinate correlation in cancer

OBSERVED: LDH-Lactate DECORRELATION (r=0.94 → 0.009)

PARADOX: Why does active fermentation BREAK the correlation?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

print("="*80)
print("CANCER FERMENTATION HYPOTHESIS ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

# NHANES
nhanes = pd.read_csv('data/nhanes/nhanes_2017_2018_processed.csv')
print(f"✓ NHANES: {len(nhanes)} participants")

# MIMIC (if we have the distribution params)
mimic_params_file = Path('data/mimic_distribution_params.pkl')
if mimic_params_file.exists():
    import pickle
    with open(mimic_params_file, 'rb') as f:
        mimic_params = pickle.load(f)
    print(f"✓ MIMIC distribution parameters loaded")
else:
    mimic_params = None
    print("  (MIMIC params not found)")

# ============================================================================
# THE PARADOX
# ============================================================================
print("\n" + "="*80)
print("THE FERMENTATION PARADOX")
print("="*80)

print("""
CANCER METABOLISM (Established Science):

1. WARBURG EFFECT (Glucose Fermentation):
   Glucose → Pyruvate → LACTATE (via LDH)
   ↑ Glycolysis even with O2 present
   ↑ LDH expression (especially LDHA)
   ↑ Lactate production

2. GLUTAMINE ADDICTION (Glutamine Fermentation):
   Glutamine → Glutamate → α-KG → SUCCINATE
   Reductive carboxylation
   Feeds TCA cycle in reverse
   ↑ Succinate production

LOGICAL PREDICTION:
  In cancer: ↑LDH, ↑Lactate, ↑Succinate → STRONG POSITIVE CORRELATION
  Because they're all products of active fermentation!

ACTUAL OBSERVATION:
  LDH-Lactate correlation BREAKS DOWN in cancer!
  • Controls: r = +0.940 (very strong)
  • Cancer:   r = +0.009 (essentially zero)
  • Δr = -0.931 (MASSIVE DECORRELATION)

THE PARADOX:
  Why does ACTIVE FERMENTATION → DECORRELATION?
  Shouldn't more fermentation → stronger correlation?
""")

# ============================================================================
# POSSIBLE EXPLANATIONS
# ============================================================================
print("\n" + "="*80)
print("POSSIBLE EXPLANATIONS FOR THE PARADOX")
print("="*80)

explanations = {
    "1. LDH Isoform Switching": """
    Normal cells: LDHB (converts Lactate → Pyruvate, aerobic)
    Cancer cells: LDHA (converts Pyruvate → Lactate, glycolytic)

    → Total serum LDH measures BOTH isoforms
    → But only LDHA produces lactate
    → LDHA↑ in cancer, but if LDHB also ↑ (from tissue damage),
       total LDH may not correlate with lactate

    TEST: Measure LDHA/LDHB ratio instead of total LDH
    """,

    "2. Lactate Export (MCT Transporters)": """
    Cancer cells EXPORT lactate rapidly via MCT1/MCT4
    → High intracellular lactate production
    → But lactate leaves cells quickly
    → Serum lactate may not reflect LDH activity
    → Spatial/temporal mismatch

    TEST: Measure intratumoral lactate vs serum lactate
    """,

    "3. Tumor Heterogeneity": """
    Different cancer cell populations:
    • Hypoxic core: High lactate production
    • Well-perfused rim: Lactate consumption (reverse Warburg)
    • Stromal cells: Normal metabolism

    → Population averaging destroys correlation
    → Each subpopulation has correlation, but mixed signal = decorrelation

    TEST: Single-cell metabolomics
    """,

    "4. Lactate as Fuel (Reverse Warburg)": """
    Some cancer cells CONSUME lactate:
    • Oxidative cancer cells use lactate from glycolytic cells
    • "Metabolic symbiosis"
    • Lactate shuttle between cell types

    → Some cells: ↑LDH + ↑Lactate (producers)
    → Other cells: ↑LDH + ↓Lactate (consumers)
    → Net effect: Decorrelation

    TEST: Lactate uptake studies
    """,

    "5. Alternative Lactate Sources": """
    Lactate can be produced by:
    • LDH pathway (main)
    • Malate-Aspartate shuttle
    • Cori cycle (liver)
    • Gut microbiome
    • Exercise/stress

    → In cancer, alternative pathways may dominate
    → LDH no longer predicts lactate

    TEST: Isotope tracing (13C-glucose)
    """,

    "6. Temporal Dynamics": """
    Cancer metabolism is DYNAMIC:
    • LDH levels: Relatively stable (enzyme half-life ~hours)
    • Lactate levels: Rapidly changing (metabolite turnover ~minutes)

    → Single timepoint measurement misses correlation
    → Need continuous monitoring

    TEST: Time-series measurements
    """,

    "7. Succinate Competition": """
    If glutamine → succinate pathway is very active:
    • Succinate can inhibit LDH activity
    • Product inhibition
    • Shifts metabolism away from lactate production

    → High fermentation, but lactate diverted
    → LDH present but not producing lactate

    TEST: Measure succinate levels
    """
}

for title, explanation in explanations.items():
    print(f"\n{title}:")
    print(explanation)

# ============================================================================
# CHECK NHANES FOR RELATED BIOMARKERS
# ============================================================================
print("\n" + "="*80)
print("AVAILABLE BIOMARKERS IN NHANES")
print("="*80)

print(f"\nNHANES columns: {list(nhanes.columns)}")
print("\nMetabolic markers we have:")
print("  ✓ Glucose")
print("  ✓ Insulin")
print("  ✓ LDH")
print("  ✓ CRP")
print("  ✓ HOMA-IR")

print("\nMetabolic markers we DON'T have:")
print("  ✗ Lactate")
print("  ✗ Succinate")
print("  ✗ Glutamine")
print("  ✗ LDH isoforms (LDHA, LDHB)")
print("  ✗ MCT transporters")
print("  ✗ Pyruvate")
print("  ✗ α-ketoglutarate")

# ============================================================================
# PREDICTIVE MODEL
# ============================================================================
print("\n" + "="*80)
print("FERMENTATION SIGNATURE PREDICTION")
print("="*80)

print("""
IF your hypothesis is correct, we should see in cancer:

1. HIGH FERMENTATION GROUP (glucose + glutamine fermenters):
   • High LDH
   • High Lactate (if we had it)
   • High Succinate (if we had it)
   • High Glucose uptake
   • All strongly correlated

2. But we observe DECORRELATION, suggesting:
   • Heterogeneous populations
   • Different metabolic states
   • Compartmentalization
   • OR: Fermentation is more complex than simple correlation

WHAT WE CAN TEST WITH CURRENT DATA:
  • Cluster analysis: Do high LDH cancer patients cluster?
  • Glucose-LDH in subgroups
  • Age-stratified analysis (older = more heterogeneous?)
""")

# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CLUSTERING ANALYSIS: METABOLIC PHENOTYPES")
print("="*80)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select cancer patients only
cancer = nhanes[nhanes['cancer'] == 1].copy()
print(f"\nCancer patients: {len(cancer)}")

# Features for clustering
cluster_features = ['glucose', 'ldh', 'insulin', 'HOMA_IR']
X_cluster = cancer[cluster_features].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-means clustering (k=3: low, medium, high fermentation)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cancer['metabolic_cluster'] = kmeans.fit_predict(X_scaled)

print("\nMetabolic Clusters in Cancer Patients:")
print("-" * 60)

for i in range(3):
    cluster = cancer[cancer['metabolic_cluster'] == i]
    print(f"\nCluster {i} (n={len(cluster)}):")
    print(f"  Glucose:  {cluster['glucose'].mean():.1f} ± {cluster['glucose'].std():.1f} mg/dL")
    print(f"  LDH:      {cluster['ldh'].mean():.1f} ± {cluster['ldh'].std():.1f} U/L")
    print(f"  Insulin:  {cluster['insulin'].mean():.1f} ± {cluster['insulin'].std():.1f} µU/mL")
    print(f"  HOMA-IR:  {cluster['HOMA_IR'].mean():.2f} ± {cluster['HOMA_IR'].std():.2f}")
    print(f"  Age:      {cluster['age'].mean():.1f} ± {cluster['age'].std():.1f} years")

    # Correlation within cluster
    if len(cluster) > 10:
        r_glu_ldh, _ = pearsonr(cluster['glucose'], cluster['ldh'])
        print(f"  Glucose-LDH correlation: r = {r_glu_ldh:+.3f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Fermentation hypothesis diagram
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
ax1.text(0.5, 0.9, 'CANCER FERMENTATION HYPOTHESIS', ha='center', fontsize=16, fontweight='bold')

# Draw pathways
ax1.text(0.15, 0.7, 'GLUCOSE', ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue'))
ax1.arrow(0.15, 0.65, 0, -0.15, head_width=0.03, head_length=0.05, fc='blue', ec='blue')
ax1.text(0.15, 0.45, 'Pyruvate', ha='center', fontsize=10)
ax1.arrow(0.15, 0.40, 0, -0.15, head_width=0.03, head_length=0.05, fc='red', ec='red')
ax1.text(0.15, 0.20, 'LACTATE', ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightcoral'))
ax1.text(0.05, 0.32, 'LDH', ha='center', fontsize=10, color='red', fontweight='bold')

ax1.text(0.5, 0.7, 'GLUTAMINE', ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax1.arrow(0.5, 0.65, 0, -0.15, head_width=0.03, head_length=0.05, fc='green', ec='green')
ax1.text(0.5, 0.45, 'Glutamate → α-KG', ha='center', fontsize=10)
ax1.arrow(0.5, 0.40, 0, -0.15, head_width=0.03, head_length=0.05, fc='green', ec='green')
ax1.text(0.5, 0.20, 'SUCCINATE', ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow'))

ax1.text(0.85, 0.5, 'PREDICTION:\nStrong correlation\nLDH↑ + Lactate↑ + Succinate↑',
         ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

ax1.text(0.5, 0.05, 'OBSERVED: LDH-Lactate DECORRELATION (r=0.94→0.009) - WHY?',
         ha='center', fontsize=12, fontweight='bold', color='red',
         bbox=dict(boxstyle='round', facecolor='mistyrose'))

# 2. Metabolic clusters - Glucose vs LDH
ax2 = fig.add_subplot(gs[1, 0])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(3):
    cluster = cancer[cancer['metabolic_cluster'] == i]
    ax2.scatter(cluster['glucose'], cluster['ldh'], alpha=0.6, s=50,
               color=colors[i], label=f'Cluster {i} (n={len(cluster)})')
ax2.set_xlabel('Glucose (mg/dL)', fontsize=11)
ax2.set_ylabel('LDH (U/L)', fontsize=11)
ax2.set_title('Cancer Metabolic Clusters\n(Are there distinct phenotypes?)',
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Cluster characteristics
ax3 = fig.add_subplot(gs[1, 1])
cluster_means = []
for i in range(3):
    cluster = cancer[cancer['metabolic_cluster'] == i]
    cluster_means.append([
        cluster['glucose'].mean() / 120,  # Normalize
        cluster['ldh'].mean() / 200,
        cluster['insulin'].mean() / 20,
        cluster['HOMA_IR'].mean() / 5
    ])

cluster_means = np.array(cluster_means)
x = np.arange(4)
width = 0.25

for i in range(3):
    ax3.bar(x + i*width, cluster_means[i], width, label=f'Cluster {i}', color=colors[i])

ax3.set_xticks(x + width)
ax3.set_xticklabels(['Glucose\n(/120)', 'LDH\n(/200)', 'Insulin\n(/20)', 'HOMA-IR\n(/5)'])
ax3.set_ylabel('Normalized Mean', fontsize=11)
ax3.set_title('Cluster Metabolic Profiles\n(Relative levels)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. The paradox visualization
ax4 = fig.add_subplot(gs[1, 2])
scenarios = ['Expected\n(Your Hypothesis)', 'Observed\n(Reality)']
expected = [0.8, 0.8, 0.8]  # Strong correlation expected
observed = [0.94, 0.009, np.nan]  # What we actually see (no succinate data)
labels = ['LDH-Lactate\n(Control)', 'LDH-Lactate\n(Cancer)', 'LDH-Succinate\n(Unknown)']

x = np.arange(len(labels))
width = 0.35

bars1 = ax4.bar(x - width/2, expected, width, label='Expected', color='lightgreen', edgecolor='black')
bars2 = ax4.bar(x + width/2, observed, width, label='Observed', color='lightcoral', edgecolor='black')

ax4.set_ylabel('Correlation (r)', fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=9)
ax4.set_title('Fermentation Hypothesis vs Reality\n(The Paradox)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.grid(axis='y', alpha=0.3)

# Add question mark for unknown
ax4.text(2, 0.5, '?', ha='center', fontsize=30, color='red', fontweight='bold')

plt.suptitle('Cancer Fermentation Hypothesis: Prediction vs Observation',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('results/fermentation_hypothesis_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS TO TEST YOUR HYPOTHESIS")
print("="*80)

print("""
PRIORITY EXPERIMENTS:

1. MEASURE SUCCINATE (Critical!):
   • Get metabolomics data with succinate
   • Test LDH-Lactate-Succinate correlations in cancer
   • If they're correlated → fermentation signature exists
   • If decorrelated → more complex mechanism

2. MEASURE LDH ISOFORMS:
   • Separate LDHA (glycolytic) from LDHB (oxidative)
   • Test LDHA-Lactate correlation
   • May restore correlation if isoform switching is the cause

3. ISOTOPE TRACING:
   • 13C-glucose → track to lactate
   • 13C-glutamine → track to succinate
   • Quantify flux through each pathway
   • Determine if decorrelation is due to mixed sources

4. SINGLE-CELL ANALYSIS:
   • Measure LDH and lactate at single-cell level
   • May find correlation within cells but not across population
   • Test heterogeneity hypothesis

5. TEMPORAL DYNAMICS:
   • Time-series measurements over hours
   • See if LDH and lactate oscillate together
   • Test if single timepoint misses correlation

6. TUMOR BIOPSY STUDIES:
   • Intratumoral lactate vs serum lactate
   • Spatial distribution of LDH vs lactate
   • Test compartmentalization hypothesis

DATASETS TO LOOK FOR:
  • Metabolomics databases (Human Metabolome Database)
  • Cancer metabolomics studies (published data)
  • Clinical trials with metabolic profiling
  • The Cancer Genome Atlas (TCGA) - metabolomics subset
""")

# Save results
results = {
    "hypothesis": "Cancer ferments glucose→lactate and glutamine→succinate, predicting strong correlation",
    "prediction": "High LDH + High Lactate + High Succinate in cancer, all correlated",
    "observation": "LDH-Lactate decorrelation (r=0.94→0.009)",
    "paradox": "Active fermentation should increase correlation, but we see decorrelation",
    "possible_explanations": list(explanations.keys()),
    "metabolic_clusters": {
        f"cluster_{i}": {
            "n": int((cancer['metabolic_cluster'] == i).sum()),
            "glucose_mean": float(cancer[cancer['metabolic_cluster'] == i]['glucose'].mean()),
            "ldh_mean": float(cancer[cancer['metabolic_cluster'] == i]['ldh'].mean()),
        }
        for i in range(3)
    },
    "missing_data": ["Lactate", "Succinate", "LDH isoforms", "Glutamine"],
    "recommendations": [
        "Measure succinate levels",
        "Measure LDH isoforms (LDHA/LDHB)",
        "Isotope tracing studies",
        "Single-cell metabolomics",
        "Temporal dynamics studies"
    ]
}

import json
with open('results/fermentation_hypothesis_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Analysis complete!")
print("\nFiles created:")
print("  • results/fermentation_hypothesis_analysis.png")
print("  • results/fermentation_hypothesis_analysis.json")
