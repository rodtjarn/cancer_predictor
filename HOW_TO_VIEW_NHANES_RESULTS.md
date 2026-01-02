# How to View NHANES Individual Results

**You have 2,312 REAL NHANES participants with predictions!**

---

## Quick Start: View in Spreadsheet (EASIEST)

### Option 1: Excel (Mac/Windows)
1. Open Microsoft Excel
2. File → Open
3. Navigate to: `data/nhanes/nhanes_with_predictions.csv`
4. Done! You can now sort, filter, and explore all 2,312 individuals

### Option 2: Numbers (Mac)
1. Open Numbers
2. File → Open
3. Navigate to: `data/nhanes/nhanes_with_predictions.csv`
4. Done!

### Option 3: Google Sheets
1. Go to sheets.google.com
2. File → Import → Upload
3. Select: `data/nhanes/nhanes_with_predictions.csv`
4. Done!

---

## What's in the CSV File?

Each row is ONE real NHANES participant with:

| Column | Description |
|--------|-------------|
| **Patient_ID** | NHANES participant ID |
| **Age** | Age in years |
| **Gender** | Male or Female |
| **Glucose_mg_dL** | Fasting glucose (mg/dL) |
| **Insulin_uU_mL** | Fasting insulin (µU/mL) |
| **LDH_U_L** | Lactate dehydrogenase (U/L) |
| **CRP_mg_L** | C-reactive protein (mg/L) |
| **HOMA_IR** | Insulin resistance index |
| **HOMA_IR_Quartile** | Q1 (low) to Q4 (high) |
| **Actual_Diagnosis** | Real diagnosis (Cancer or No Cancer) |
| **Predicted_Diagnosis** | Model prediction (Cancer or No Cancer) |
| **Cancer_Probability_%** | Model's confidence (0-100%) |
| **Risk_Category** | Low, Medium, High, or Very High Risk |
| **Prediction_Correct** | TRUE if model was right, FALSE if wrong |

---

## Useful Filters/Sorts in Excel

### Find High-Risk Individuals
- Filter: `Risk_Category` = "Very High Risk"
- Shows: 62 people with >75% cancer probability
- Result: 51 of them (82%) actually have cancer

### Find Missed Cancers (False Negatives)
- Filter: `Actual_Diagnosis` = "Cancer" AND `Predicted_Diagnosis` = "No Cancer"
- Shows: 53 cancer patients the model missed

### Find False Alarms (False Positives)
- Filter: `Actual_Diagnosis` = "No Cancer" AND `Predicted_Diagnosis` = "Cancer"
- Shows: 239 healthy people incorrectly flagged

### Sort by Cancer Probability
- Sort: `Cancer_Probability_%` descending
- See who has the highest cancer risk

### View by Insulin Resistance
- Sort: `HOMA_IR_Quartile`
- Compare Q1 (low) vs Q4 (high) cancer rates

---

## Python/Jupyter Notebook

If you prefer Python:

```python
import pandas as pd

# Load the data
df = pd.read_csv('data/nhanes/nhanes_with_predictions.csv')

# View first 10 rows
print(df.head(10))

# Show summary statistics
print(df.describe())

# Find cancer patients
cancer_patients = df[df['Actual_Diagnosis'] == 'Cancer']
print(f"Found {len(cancer_patients)} cancer patients")

# Find high-risk individuals
high_risk = df[df['Cancer_Probability_%'] > 75]
print(f"Found {len(high_risk)} high-risk individuals")
print(f"Of these, {high_risk[high_risk['Actual_Diagnosis']=='Cancer'].shape[0]} have cancer")

# View specific patient
patient_93758 = df[df['Patient_ID'] == 93758.0]
print(patient_93758)

# Misclassifications
false_negatives = df[(df['Actual_Diagnosis'] == 'Cancer') &
                      (df['Predicted_Diagnosis'] == 'No Cancer')]
print(f"Missed {len(false_negatives)} cancers")

# Group by risk category
risk_summary = df.groupby('Risk_Category').agg({
    'Patient_ID': 'count',
    'Actual_Diagnosis': lambda x: (x == 'Cancer').sum()
}).rename(columns={'Patient_ID': 'Total', 'Actual_Diagnosis': 'Cancer_Count'})
print(risk_summary)
```

---

## Interactive Python Script

For an interactive menu-driven exploration:

```bash
python view_nhanes_individual_predictions.py
```

This gives you options to:
1. View cancer patients
2. View high-risk individuals
3. View misclassified cases
4. Search by patient ID
5. Show summary stats

---

## Command Line (Quick Look)

```bash
# View first 10 rows
head -n 11 data/nhanes/nhanes_with_predictions.csv

# Count total rows
wc -l data/nhanes/nhanes_with_predictions.csv

# Find high-risk cancer cases (grep for "Very High Risk" and "Cancer")
grep "Very High Risk.*,Cancer," data/nhanes/nhanes_with_predictions.csv
```

---

## Sample Queries

### Example 1: Find Patient 93758 (False Negative)
**In Excel**: Filter Patient_ID = 93758
**Result**: 55-year-old female with cancer (probability 37.7%, incorrectly predicted as No Cancer)

### Example 2: Compare HOMA-IR Quartiles
**In Excel**:
1. Filter HOMA_IR_Quartile = Q1, count cancers
2. Filter HOMA_IR_Quartile = Q4, count cancers
3. Compare rates

**Expected**: Higher insulin resistance (Q4) → more cancers

### Example 3: View All Correct Cancer Predictions
**Filter**:
- Actual_Diagnosis = "Cancer"
- Prediction_Correct = TRUE

**Shows**: 171 correctly identified cancer patients

---

## Summary Statistics

From the 2,312 REAL NHANES participants:

### Overall Performance
- **Accuracy**: 87.4%
- **Sensitivity**: 76.3% (caught 171/224 cancers)
- **Specificity**: 88.6% (correctly ID'd 1,849/2,088 healthy)

### Risk Categories
| Category | Count | Cancer Cases | Cancer Rate |
|----------|-------|--------------|-------------|
| Low Risk | 1,317 | 12 | 0.9% |
| Medium Risk | 582 | 41 | 7.0% |
| High Risk | 348 | 120 | 34.5% |
| **Very High Risk** | **62** | **51** | **82.3%** ⭐ |

### Key Insight
**Very High Risk individuals (>75% probability):**
- Only 62 people (2.7% of population)
- But 51 of them (82%) actually have cancer!
- This shows the model is good at identifying high-risk individuals

---

## Files Available

| File | Description | Best For |
|------|-------------|----------|
| `data/nhanes/nhanes_2017_2018_processed.csv` | Raw NHANES data without predictions | Original data |
| `data/nhanes/nhanes_with_predictions.csv` | **NHANES data WITH predictions** | **Exploring results** ⭐ |

---

## Need Help?

**Regenerate the predictions file:**
```bash
python generate_nhanes_predictions.py
```

**Interactive exploration:**
```bash
python view_nhanes_individual_predictions.py
```

**View in spreadsheet:**
- Excel: File → Open → `data/nhanes/nhanes_with_predictions.csv`
- Numbers: File → Open → `data/nhanes/nhanes_with_predictions.csv`
- Google Sheets: Import → Upload → Select file

---

## Recommended: Open in Excel

**The easiest way to explore is Excel/Numbers/Google Sheets:**
1. Open the CSV in your spreadsheet software
2. Use filters to explore different groups
3. Sort by cancer probability to see high-risk individuals
4. Compare actual vs predicted diagnoses
5. Look at biomarker patterns in correct vs incorrect predictions

**You now have all 2,312 real patients with full biomarker data and model predictions!**
