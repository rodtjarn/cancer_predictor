# BigQuery Setup Guide for MIMIC-IV Access

**Purpose**: Set up Google BigQuery access to analyze full MIMIC-IV dataset
**Time**: 30 minutes
**Prerequisites**: Full MIMIC-IV access approved by PhysioNet

---

## Overview

Once your MIMIC-IV access is approved, the fastest way to analyze the data is through Google BigQuery (cloud-based). This avoids downloading 40+ GB of data and provides instant query access to 73,181 patients.

---

## Step 1: Create Google Cloud Account

### 1.1 Sign Up for Google Cloud

1. **Go to**: https://cloud.google.com/
2. **Click**: "Get started for free"
3. **Sign in** with your Google account (or create one)
4. **Provide**:
   - Country
   - Payment method (credit card required)
   - **NOTE**: You get $300 free credits for 90 days

### 1.2 Create a Project

1. **Go to**: https://console.cloud.google.com/
2. **Click**: "Select a project" (top of page)
3. **Click**: "New Project"
4. **Enter**:
   - Project name: `mimic-iv-analysis` (or your choice)
   - Organization: Leave blank (or select if you have one)
5. **Click**: "Create"
6. **Note your Project ID** (e.g., `mimic-iv-analysis-123456`)

---

## Step 2: Enable BigQuery API

### 2.1 Enable the API

1. **Go to**: https://console.cloud.google.com/apis/library
2. **Search**: "BigQuery API"
3. **Click**: "BigQuery API"
4. **Click**: "Enable"
5. **Wait**: ~30 seconds for activation

### 2.2 Verify Access

1. **Go to**: https://console.cloud.google.com/bigquery
2. **You should see**: BigQuery interface with empty workspace
3. **If prompted**: Accept terms of service

---

## Step 3: Link PhysioNet to Google Cloud

### 3.1 Get Your Google Cloud Email

1. **Go to**: https://console.cloud.google.com/iam-admin/iam
2. **Look for**: Your email address (e.g., `your-email@gmail.com`)
3. **Copy** this email address

### 3.2 Link on PhysioNet

1. **Log in to PhysioNet**: https://physionet.org/login/
2. **Go to**: https://physionet.org/settings/cloud/
3. **Under "Google Cloud Platform"**:
   - **Paste** your Google Cloud email
   - **Click**: "Link account"
4. **Verify**: You should see "Successfully linked"

---

## Step 4: Access MIMIC-IV in BigQuery

### 4.1 Verify Access

1. **Go to BigQuery**: https://console.cloud.google.com/bigquery
2. **In the Explorer** (left sidebar):
   - **Click**: "+ ADD DATA"
   - **Select**: "Star a project by name"
   - **Enter**: `physionet-data`
   - **Click**: "Star"

3. **Expand** `physionet-data` in the Explorer
4. **You should see**: `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_ed`, etc.
5. **Click on** `mimiciv_hosp` → `patients`
6. **You should see**: Table schema with columns

### 4.2 Test Query

Run this test query to verify access:

```sql
SELECT COUNT(*) as patient_count
FROM `physionet-data.mimiciv_hosp.patients`
```

**Expected result**: ~73,181 patients

If you see this number, **you're ready!** ✅

---

## Step 5: Install Python Libraries

### 5.1 Install Required Packages

```bash
pip install google-cloud-bigquery pandas numpy scikit-learn matplotlib seaborn
```

### 5.2 Authenticate Python

Run this command to authenticate:

```bash
gcloud auth application-default login
```

**Steps**:
1. Browser will open
2. Sign in with your Google account
3. Click "Allow"
4. Return to terminal

**Verify**:
```bash
gcloud config list
```

You should see your project ID.

### 5.3 Set Default Project

```bash
gcloud config set project YOUR_PROJECT_ID
```

Replace `YOUR_PROJECT_ID` with your actual project ID (e.g., `mimic-iv-analysis-123456`).

---

## Step 6: Configure Analysis Script

### 6.1 Update Script

1. **Open**: `analyze_full_mimic_iv.py`
2. **Find line 40**: `PROJECT_ID = "YOUR_PROJECT_ID"`
3. **Replace** with your actual project ID:
   ```python
   PROJECT_ID = "mimic-iv-analysis-123456"  # Your actual ID
   ```
4. **Save** the file

### 6.2 Test Connection

Run a quick test:

```bash
python3 -c "
from google.cloud import bigquery
client = bigquery.Client(project='YOUR_PROJECT_ID')
query = 'SELECT COUNT(*) FROM \`physionet-data.mimiciv_hosp.patients\`'
result = client.query(query).to_dataframe()
print(f'✅ Connected! Found {result.iloc[0,0]:,} patients')
"
```

Replace `YOUR_PROJECT_ID` with your actual project ID.

**Expected output**: `✅ Connected! Found 73,181 patients`

---

## Step 7: Run Full Analysis

### 7.1 Execute Script

```bash
python3 analyze_full_mimic_iv.py
```

**What happens**:
1. Connects to BigQuery
2. Extracts all patient demographics (~73,181 patients)
3. Identifies cancer patients (~5,000-10,000 expected)
4. Extracts biomarker data (Glucose, Lactate, LDH, CRP)
5. Calculates real BMI from height/weight
6. Trains 4-biomarker model
7. Trains 6-biomarker model (with real CRP and BMI)
8. Analyzes by cancer type
9. Generates comprehensive report and visualizations

**Estimated time**: 10-30 minutes (depending on internet speed)

### 7.2 Monitor Progress

The script prints progress updates:
```
STEP 1: Connecting to BigQuery...
✅ Connected! Found 73,181 patients
STEP 2: Extracting patient demographics...
✅ Extracted demographics for 73,181 patients
STEP 3: Identifying cancer patients...
✅ Found 12,543 cancer diagnoses
...
```

### 7.3 Check Output

Results saved to `full_mimic_iv_results/`:
- `full_mimic_iv_biomarker_data.csv` - Raw data
- `model_4biomarkers_full_mimic.pkl` - Validated 4-biomarker model
- `model_6biomarkers_full_mimic.pkl` - 6-biomarker model with CRP+BMI
- `cancer_type_analysis.csv` - Performance by cancer type
- `full_mimic_iv_validation.png` - Comprehensive visualization
- `FULL_MIMIC_IV_VALIDATION_REPORT.md` - Detailed report

---

## Cost Estimate

### BigQuery Costs

**Query pricing**: $5 per TB processed

**Expected usage for this analysis**:
- Patient demographics: ~1 GB
- Cancer diagnoses: ~500 MB
- Lab events: ~10 GB (biomarker extraction)
- Height/weight: ~100 MB
- **Total**: ~12 GB

**Cost**: 12 GB × $5/TB = **$0.06** (6 cents!)

### Free Tier

Google Cloud free tier includes:
- **$300 free credits** for 90 days (new accounts)
- **10 GB/month free** BigQuery queries (permanent)

**You'll likely spend $0** on this analysis! ✅

---

## Troubleshooting

### Issue 1: "Permission denied"

**Problem**: Can't access `physionet-data`

**Solutions**:
1. Verify PhysioNet approval email received
2. Check you signed Data Use Agreement
3. Verify Google Cloud email linked on PhysioNet
4. Wait 24 hours after linking (propagation delay)
5. Contact PhysioNet support: contact@physionet.org

### Issue 2: "Project not set"

**Problem**: `gcloud` commands fail

**Solution**:
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
```

### Issue 3: "BigQuery API not enabled"

**Problem**: API calls fail

**Solution**:
1. Go to: https://console.cloud.google.com/apis/library
2. Search: "BigQuery API"
3. Click: "Enable"

### Issue 4: "Module not found: google.cloud.bigquery"

**Problem**: Python import fails

**Solution**:
```bash
pip install --upgrade google-cloud-bigquery
```

### Issue 5: Query costs too high

**Problem**: Concerned about costs

**Solutions**:
- Use **query preview** to estimate costs before running
- Limit data with `WHERE` clauses
- Use **query caching** (automatic in BigQuery)
- Monitor usage: https://console.cloud.google.com/billing

### Issue 6: Slow queries

**Problem**: Queries taking too long

**Solutions**:
- BigQuery automatically optimizes queries
- Use smaller date ranges for testing
- Partition tables by date if needed
- Check query execution plan in BigQuery console

---

## Alternative: Local Download (Not Recommended)

If you absolutely cannot use BigQuery:

### Download MIMIC-IV Locally

1. **Go to**: https://physionet.org/content/mimiciv/
2. **Files section**: Click to view files
3. **Download** all `.csv.gz` files (~40 GB compressed)
4. **Extract** files (~120 GB uncompressed)
5. **Load** into PostgreSQL or SQLite

**Downsides**:
- Very slow download (hours)
- Requires 120+ GB disk space
- Manual database setup
- Slower queries

**Only do this if**:
- You need offline access
- You have limited cloud budget
- You're doing intensive local processing

---

## Next Steps After Analysis

Once the script completes:

1. **Review results**: Check `FULL_MIMIC_IV_VALIDATION_REPORT.md`

2. **Compare to demo**:
   - Demo: 73.3% accuracy (n=100)
   - Full: TBD (n=73,181)
   - Expected: 75-85% with better data quality

3. **Test metabolic theory**:
   - Check if performance consistent across cancer types
   - Verify LDH dominance (should be ~35-40%)
   - Confirm Warburg markers account for ~75% importance

4. **Evaluate CRP and BMI**:
   - CRP importance with real data (vs 4.9% in demo)
   - BMI importance with real data (vs 0.0% in demo)
   - Expected: 10-20% each if data quality good

5. **Prepare publication**:
   - Comprehensive dataset (73,181 patients)
   - Robust statistics (narrow confidence intervals)
   - Cancer-specific analysis (n≥30 per type)
   - Validated on real-world EHR data

---

## Tips for Success

### ✅ DO:
- Start with small test queries to verify access
- Use query preview to estimate costs
- Save query results to avoid re-running
- Export data to CSV for local analysis if needed
- Monitor your billing: https://console.cloud.google.com/billing

### ❌ DON'T:
- Run queries without estimating cost first
- Select `*` from large tables unnecessarily
- Download data if you can use BigQuery
- Share your Google Cloud credentials
- Leave unused projects running

---

## Quick Reference

### Useful BigQuery Queries

**Count patients**:
```sql
SELECT COUNT(*) FROM `physionet-data.mimiciv_hosp.patients`
```

**Count cancer patients**:
```sql
SELECT COUNT(DISTINCT subject_id)
FROM `physionet-data.mimiciv_hosp.diagnoses_icd`
WHERE icd_version = 10 AND icd_code LIKE 'C%'
```

**Check glucose availability**:
```sql
SELECT COUNT(DISTINCT subject_id)
FROM `physionet-data.mimiciv_hosp.labevents`
WHERE itemid IN (50809, 50931)
```

**Check LDH availability**:
```sql
SELECT COUNT(DISTINCT subject_id)
FROM `physionet-data.mimiciv_hosp.labevents`
WHERE itemid = 50954
```

### Useful Commands

**Authenticate**:
```bash
gcloud auth application-default login
```

**Set project**:
```bash
gcloud config set project YOUR_PROJECT_ID
```

**Check configuration**:
```bash
gcloud config list
```

**Run analysis**:
```bash
python3 analyze_full_mimic_iv.py
```

---

## Support Resources

**Google Cloud**:
- Documentation: https://cloud.google.com/bigquery/docs
- Pricing: https://cloud.google.com/bigquery/pricing
- Free tier: https://cloud.google.com/free

**MIMIC-IV**:
- Documentation: https://mimic.mit.edu/docs/iv/
- Tutorials: https://github.com/MIT-LCP/mimic-code
- Forum: https://github.com/MIT-LCP/mimic-code/issues

**PhysioNet**:
- Support: contact@physionet.org
- Cloud access: https://physionet.org/settings/cloud/

---

**Ready to analyze 73,181 patients!** 🚀

**Status**: Setup guide complete
**Next**: Get PhysioNet approval, then follow this guide
