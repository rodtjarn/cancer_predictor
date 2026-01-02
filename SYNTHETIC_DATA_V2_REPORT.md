# Synthetic Data V2 (Realistic Stochastics) - Analysis Report

**Date**: 2026-01-01
**Purpose**: Test whether more realistic synthetic data generation improves real-world model performance
**Status**: ❌ NEGATIVE RESULT - V2 performed significantly worse on real MIMIC-IV data

---

## Executive Summary

We hypothesized that using more realistic stochastics (skewed distributions, correlated features, outliers) in synthetic data generation would reduce the sim-to-real gap and improve model performance on real MIMIC-IV patients.

**Result**: The hypothesis was **REJECTED**. V2 model performed 20.4 percentage points WORSE on real patients compared to V1.

---

## Background

### Original Problem (V1)
- **V1 synthetic data**: Simple uniform distributions for all biomarkers
- **V1 performance**: 99.21% synthetic, 73.3% MIMIC-IV
- **Sim-to-real gap**: 25.9 pp
- **Diagnosis**: Distribution overfitting (synthetic data too simple, didn't match real-world complexity)

### Hypothesis
More realistic synthetic data (V2) with:
- Skewed distributions (not uniform)
- Correlated metabolic biomarkers (glucose, lactate, LDH)
- Outliers (5-15% per feature)
- Multimodal cancer subtypes

Should reduce sim-to-real gap and improve real-world generalization.

---

## V2 Improvements to Synthetic Data Generation

### 1. Distribution Types
**V1 (Uniform):**
```python
np.random.uniform(100, 250)  # LDH (rectangular distribution)
```

**V2 (Realistic):**
```python
skewnorm.rvs(a=2, loc=180, scale=50)  # LDH (right-skewed)
```

### 2. Feature Correlations
**V1:** All features independent
**V2:** Glucose, lactate, and LDH correlated via multivariate normal
```python
# Correlation matrix:
# Glucose-Lactate: 0.4 (moderate positive)
# Glucose-LDH: 0.3 (weak positive)
# Lactate-LDH: 0.6 (strong positive - biologically linked)
```

### 3. Outliers
**V1:** No outliers (uniform distributions can't create outliers)
**V2:** 3-15% outliers per feature, magnitude 2-3x normal range

### 4. Population Heterogeneity
**V1:** 5 patient subtypes with uniform distributions within each
**V2:** 5 patient subtypes with multiple sub-subtypes (multimodal within cancer)

---

## Results

### Synthetic Test Set Performance

| Metric | V1 (Uniform) | V2 (Realistic) | Difference |
|--------|--------------|----------------|------------|
| Accuracy | 99.21% | 96.00% | -3.21 pp |
| Sensitivity | ~99% | 94.5% | -4.5 pp |
| Specificity | ~99% | 96.8% | -2.2 pp |

**Interpretation**: V2 is slightly harder (more realistic), as expected.

---

### Real MIMIC-IV Performance

| Metric | V1 (Uniform) | V2 (Realistic) | Difference |
|--------|--------------|----------------|------------|
| **Accuracy** | **73.3%** | **52.9%** | **-20.4 pp** ❌ |
| **Sensitivity** | **63.6%** | **12.5%** | **-51.1 pp** ❌❌❌ |
| **Specificity** | **78.9%** | **88.9%** | **+10.0 pp** ✅ |
| **CV Mean** | **64.0%** | **52.7%** | **-11.3 pp** ❌ |
| **CV Std** | **4.9%** | **13.4%** | **+8.5 pp** ❌ |

**Critical Finding**: V2 model is extremely conservative:
- Detects only 12.5% of cancer patients (vs 63.6% for V1)
- Correctly identifies 88.9% of controls (vs 78.9% for V1)
- Confusion matrix: 7/8 cancers missed!

---

### Sim-to-Real Gap

| Model | Synthetic Accuracy | Real Accuracy | Gap |
|-------|-------------------|---------------|-----|
| V1 (Uniform) | 99.21% | 73.3% | **25.9 pp** |
| V2 (Realistic) | 96.00% | 52.9% | **43.1 pp** ❌ |

**Gap increased by 17.1 pp** - opposite of our hypothesis!

---

### Feature Importance Comparison

**V1 Model (4 biomarkers):**
```
LDH:        37.4%  ← Dominant
Age:        25.5%
Glucose:    19.1%
Lactate:    17.9%
```

**V2 Model (4 biomarkers):**
```
Lactate:    37.5%  ← Dominant (switched!)
LDH:        31.4%
Age:        21.0%
Glucose:    10.0%
```

**Observation**: V2 learned different feature importance, prioritizing Lactate over LDH.

---

## Root Cause Analysis

### Why Did V2 Fail?

#### 1. **Wrong Distribution Assumptions**
Our "realistic" distributions were based on:
- Published literature (healthy populations)
- Our assumptions about cancer distributions
- Biological plausibility

But **MIMIC-IV patients are critically ill ICU patients**, not general population:
- Baseline lactate/LDH already elevated (critical illness)
- Cancer vs control distinction less clear in ICU setting
- Confounding conditions (sepsis, organ failure, inflammation)

#### 2. **Overfitting to Wrong Distribution**
V2 learned more specific patterns from V2 synthetic data:
- Correlated biomarkers → specific correlation structure
- Skewed distributions → specific skewness parameters
- Multimodal subtypes → specific subtype patterns

When real MIMIC-IV had **different** correlations/skewness/subtypes, V2 failed catastrophically.

#### 3. **Conservative Decision Boundary**
V2's feature importance shift (Lactate > LDH) suggests:
- V2 learned to rely on lactate as primary signal
- But in MIMIC-IV, lactate is elevated in many critically ill controls
- Result: model too conservative, requires very high lactate to predict cancer
- Outcome: 88.9% specificity but only 12.5% sensitivity

#### 4. **Sample Size Amplified Variance**
- Small MIMIC-IV test set (n=17)
- V2 CV std: 13.4% (vs V1: 4.9%)
- V2 is less stable → more sensitive to small sample variance

---

## Key Insights

### 1. **Uniform Distributions Were Actually Better**
Paradoxically, V1's "overly simple" uniform distributions allowed the model to learn **general patterns** that transferred better to the specific (but different) MIMIC-IV distribution.

**V1 learning**: "Cancer has higher LDH than controls" (general pattern)
**V2 learning**: "Cancer has LDH with skewness=2, correlation=0.6 with lactate" (specific pattern)

Real MIMIC-IV had different skewness/correlations → V2 failed, V1 succeeded.

### 2. **Distribution Overfitting Can Get Worse**
More realistic ≠ better if "realistic" means "specific to wrong distribution"

**Overfitting hierarchy**:
1. Classical overfitting (train set) ← Random Forest prevents this ✅
2. Distribution overfitting (synthetic→real) ← Both V1 and V2 have this
3. **Parameter overfitting** (specific distribution parameters) ← V2 introduced this! ❌

### 3. **The Problem Wasn't Stochastics**
The real issue is **population mismatch**:
- Synthetic: General population (healthy + various cancers)
- MIMIC-IV: Critically ill ICU patients

No amount of realistic stochastics can fix a fundamental population mismatch.

### 4. **Simpler Can Be More Robust**
V1's uniform distributions created **wider, more general decision boundaries** that happened to work better on the specific (but different) MIMIC-IV distribution.

V2's realistic distributions created **narrow, specific decision boundaries** optimized for V2's particular distribution parameters.

---

## Conclusion

### Primary Finding
**More realistic synthetic data generation DECREASED real-world performance** from 73.3% to 52.9% (-20.4 pp), with sim-to-real gap INCREASING from 25.9 pp to 43.1 pp.

### Why This Happened
1. "Realistic" was based on wrong assumptions (general population, not ICU patients)
2. More specific distributions → overfitting to specific distribution parameters
3. Simpler V1 distributions → more general patterns → better transfer

### Implications

**For this project:**
- **Use V1 model** (73.3% accuracy) for real-world deployment
- V2 model is not fit for purpose (only 12.5% sensitivity)
- The problem is population mismatch, not data generation method

**For synthetic data generally:**
- More realistic ≠ automatically better
- Need to match **target distribution**, not just "be realistic"
- Simpler distributions can be more robust when target is uncertain
- Validate on real data early to avoid wasted effort

**For future work:**
- Need to train on actual MIMIC-IV data (or similar ICU population)
- Or: Use domain adaptation techniques to adjust for population shift
- Or: Collect synthetic data specifically matching ICU patient characteristics

---

## Next Steps

### Immediate
1. ✅ Document findings (this report)
2. ❌ Do NOT use V2 model for any real-world application
3. ✅ Continue using V1 validated model (73.3% accuracy)

### Short-term
1. Apply for full MIMIC-IV access (73,181 patients)
2. Train directly on real ICU patient data
3. Expected improvement: 73.3% → 80-85% (better data quality + larger sample)

### Long-term (if pursuing general population screening)
1. Collect data from outpatient/screening populations (not ICU)
2. Generate synthetic data matching **that** distribution
3. Validate transfer learning from synthetic to real outpatient data

---

## Files Generated

- `src/generate_data_v2.py` - Improved synthetic data generator
- `data/training_data_v2.npz` - 24,500 training samples with realistic stochastics
- `data/test_data_v2.npz` - 10,500 test samples with realistic stochastics
- `models/model_v2_realistic_stochastics.pkl` - V2 model (7 biomarkers)
- `train_v2_model.py` - Training script for V2
- `test_v2_on_mimic.py` - MIMIC-IV validation script
- `results/v1_vs_v2_mimic_comparison.pkl` - Comparison results
- `results/v2_feature_importance.png` - Feature importance plot
- `SYNTHETIC_DATA_V2_REPORT.md` - This report

---

## Lessons Learned

1. **Validate early**: We should have tested V2 on MIMIC-IV immediately after generating data
2. **Negative results are valuable**: This failure teaches us that realistic ≠ better when distributions don't match
3. **Simpler is often better**: Don't add complexity unless you're sure it helps
4. **Know your target population**: Synthetic data must match deployment population, not just "be realistic"
5. **Random Forest protects against overfitting BUT**: It doesn't protect against distribution mismatch or parameter overfitting

---

**Status**: V2 experiment complete, hypothesis rejected, V1 model remains recommended approach

**Recommendation**: Proceed with full MIMIC-IV access application to train on real ICU patient data directly rather than improving synthetic data generation.
