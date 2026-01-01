# MIMIC-IV Full Access Application Guide

**Goal**: Secure access to full MIMIC-IV dataset (73,181 patients vs 100 in demo)
**Timeline**: 3-7 business days (possibly longer due to staffing delays)
**Cost**: Free

---

## Overview: 3-Step Process

1. ✅ **Complete CITI Training** (2-4 hours)
2. ✅ **Submit PhysioNet Credentialing** (10-15 minutes)
3. ✅ **Sign Data Use Agreement** (5 minutes)

---

## Step 1: Complete CITI Training

### What You Need

**Course**: CITI "Data or Specimens Only Research"
- Covers human subjects research ethics
- Includes HIPAA requirements
- Valid for 3 years
- **Duration**: 2-4 hours (self-paced, online)

### How to Complete

1. **Go to CITI Program**: https://about.citiprogram.org/

2. **Register/Login**:
   - Click "Register"
   - Select your affiliated institution (or "Independent Learner" if none)
   - Create account with your email

3. **Select Course**:
   - Choose: **"Data or Specimens Only Research"**
   - Alternative names: "Human Subjects Research - Data or Specimens Only"
   - This is the shortest track (no direct patient contact)

4. **Complete Modules**:
   - Work through all required modules
   - Must pass quizzes (usually 80% threshold)
   - Can retake quizzes if needed
   - Save your progress (can complete over multiple sessions)

5. **Download Completion Report**:
   - After completing all modules, go to "My Reports"
   - Download the **Completion Report PDF**
   - **IMPORTANT**: Keep this PDF - you'll upload it to PhysioNet

### Cost
- **Free** at most institutions
- Some independent learners may be charged (~$50)

---

## Step 2: Submit PhysioNet Credentialing

### Create PhysioNet Account

1. **Go to PhysioNet**: https://physionet.org/

2. **Register**:
   - Click "Register" (top right)
   - Fill in your details:
     - Email (use professional/academic email if possible)
     - Username
     - Password
   - Verify email

### Submit Credentialing Application

1. **Log in to PhysioNet**: https://physionet.org/login/

2. **Go to Credentialing Page**: https://physionet.org/settings/credentialing/

3. **Fill Out Application**:

   **Personal Information:**
   - Full name
   - Current affiliation (organization/institution)
   - Professional title/position
   - Contact information

   **Reference Information** (REQUIRED if you are a student/postdoc):
   - Supervisor's name
   - Supervisor's email
   - Supervisor's affiliation
   - **Note**: Your supervisor may be contacted to verify your research

   **Research Summary:**
   - Brief description of your research purpose
   - How you plan to use MIMIC-IV data
   - Expected outcomes

4. **Upload CITI Training Report**:
   - Go to: https://physionet.org/settings/training/
   - Upload the CITI Completion Report PDF
   - Ensure it's the correct course (Data or Specimens Only)
   - Ensure it's not expired (valid for 3 years)

5. **Submit Application**:
   - Review all information carefully
   - Incomplete applications will be delayed or rejected
   - Click "Submit"

---

## Step 3: Sign Data Use Agreement (DUA)

**Wait for Approval First**: You'll receive an email when credentialing is approved (3-7+ business days)

### Once Approved

1. **Go to MIMIC-IV Project Page**: https://physionet.org/content/mimiciv/

2. **Navigate to Files Section**:
   - Look for "Files" tab/section
   - You'll see the Data Use Agreement

3. **Review DUA**:
   - Read the PhysioNet Credentialed Health Data Use Agreement 1.5.0
   - Key requirements:
     - ✅ Use data only for research purposes
     - ✅ Protect patient privacy (HIPAA compliance)
     - ✅ Do not attempt to re-identify patients
     - ✅ Do not share data with non-credentialed users
     - ✅ Cite MIMIC-IV in publications
     - ✅ Report any data breaches immediately

4. **Sign Agreement**:
   - Click "Sign" button
   - Electronic signature is legally binding
   - **You're now approved for access!**

5. **Access Data**:
   - Files will become visible in the Files section
   - **Recommended**: Use cloud access (Google BigQuery or AWS)
   - **Alternative**: Download locally (very large files)

---

## Research Purpose Statement (Draft for You)

Use this as a template when filling out the "Research Summary" section:

```
Title: Validation of Metabolic Biomarker Panel for Cancer Detection

Purpose:
I am developing a machine learning model to detect cancer using routine blood
test biomarkers (glucose, lactate, LDH, age, CRP, BMI) based on the Warburg
effect. I have completed initial validation on the MIMIC-IV Demo dataset
(100 patients, 73.3% accuracy) and need access to the full MIMIC-IV dataset
to:

1. Validate model performance on larger sample (n≥1,000 cancer patients)
2. Test model across different cancer types (lung, GI, breast, hematologic)
3. Improve biomarker panel with better data quality (real CRP, BMI measurements)
4. Analyze metabolic patterns by cancer stage and type
5. Develop cancer-specific decision thresholds

Expected Outcomes:
- Validated cancer detection model with robust performance metrics
- Publication of findings in peer-reviewed journal
- Open-source model for research community
- Insights into metabolic theory of cancer (testing Seyfried hypothesis)

Data Security:
- Data will be stored on encrypted local machine
- No attempt to re-identify patients
- No data sharing with unauthorized individuals
- Compliance with HIPAA and PhysioNet DUA requirements
- Results will be reported only in aggregate (no individual patient data)

Timeline: 6-12 months
```

**Adjust this based on your actual situation** (student/independent researcher/affiliated institution).

---

## Important Information to Have Ready

### Personal Details
- [ ] Full legal name
- [ ] Email address (preferably institutional/professional)
- [ ] Current affiliation/institution
- [ ] Professional title/position
- [ ] Phone number

### Reference Information (if student/postdoc)
- [ ] Supervisor's full name
- [ ] Supervisor's email
- [ ] Supervisor's institution
- [ ] Supervisor's title

### Documents
- [ ] CITI Training Completion Report (PDF)
- [ ] Proof of affiliation (may be requested)

---

## Timeline Expectations

| Step | Duration |
|------|----------|
| **CITI Training** | 2-4 hours (self-paced) |
| **Submit Credentialing** | 10-15 minutes |
| **Approval Wait** | 3-7 business days (possibly longer)* |
| **Sign DUA** | 5 minutes |
| **Total Time** | ~1 week |

**⚠️ Warning**: PhysioNet recently noted "applications for credentialed access are likely to be subject to significant delays" due to staffing changes. Budget 1-2 weeks for approval.

---

## Common Issues & Solutions

### Issue 1: "My institution is not listed"
**Solution**: Select "Not Listed" or "Independent Learner" for CITI training

### Issue 2: "I don't have a supervisor" (independent researcher)
**Solution**: You can still apply - use professional references or leave reference section blank. Explain your independent research status in the research summary.

### Issue 3: "CITI training is asking for payment"
**Solution**:
- Check if your institution has an agreement with CITI (most universities do)
- Try registering through your institution's IRB/research office
- Cost is typically ~$50 for independent learners (one-time)

### Issue 4: "Application is taking too long"
**Solution**:
- Check that CITI report is uploaded correctly
- Ensure all required fields are filled
- Email PhysioNet support: contact@physionet.org
- Be patient - staffing delays are expected

### Issue 5: "I'm a student - will my application be approved?"
**Solution**: Yes! Students are welcome. Just make sure to:
- Provide supervisor contact information
- Clearly explain your research purpose
- Have a legitimate research project

---

## After Approval: Accessing MIMIC-IV

### Option 1: Cloud Access (RECOMMENDED)

**Google BigQuery** (Fastest for analysis):
1. Create Google Cloud account
2. Link PhysioNet account to Google Cloud
3. Access MIMIC-IV directly in BigQuery
4. Run SQL queries on 73,181 patients
5. **Cost**: Free tier available, ~$5-10 for typical analysis

**AWS** (Alternative):
1. Create AWS account
2. Link PhysioNet account to AWS
3. Access via S3 or Athena
4. Similar pricing to BigQuery

**Advantage**: No download time, instant access, scalable

### Option 2: Local Download (NOT Recommended)

**Size**: ~30-40 GB compressed
**Time**: Several hours to download
**Storage**: Need 100+ GB for uncompressed data
**Only do this if**:
- You need offline access
- You have limited cloud budget
- You're doing intensive local processing

---

## What You Get With Full MIMIC-IV

### Dataset Size Comparison

| Feature | MIMIC-IV Demo | MIMIC-IV Full |
|---------|---------------|---------------|
| **Patients** | 100 | **73,181** |
| **Admissions** | 275 | **523,740** |
| **Lab Events** | 107,727 | **122,103,667** |
| **Diagnoses** | 4,506 | **4,756,326** |
| **Cancer Patients** | 9 | **~5,000-10,000** (estimated) |

### Expected Improvements for Your Model

**Current (Demo, n=100)**:
- 9 cancer patients (7 with complete data)
- 73.3% accuracy
- Cannot test by cancer type (n too small)
- CRP: 81% imputed
- BMI: constant approximation

**With Full MIMIC-IV (n=73,181)**:
- Expected: 5,000-10,000 cancer patients
- Better CRP coverage (expected 40-60% real measurements)
- Real BMI data (height/weight measurements)
- n≥30 per cancer type (robust statistical analysis)
- Cancer-specific models possible
- Narrow confidence intervals
- **Expected accuracy: 80-85%** with CRP+BMI re-added

---

## Next Steps After This Guide

1. [ ] **Start CITI training today** (2-4 hours)
   - https://about.citiprogram.org/
   - Course: "Data or Specimens Only Research"

2. [ ] **Download CITI completion report**

3. [ ] **Create PhysioNet account**
   - https://physionet.org/register/

4. [ ] **Submit credentialing application**
   - https://physionet.org/settings/credentialing/
   - Upload CITI report
   - Fill out research summary (use template above)

5. [ ] **Wait for approval** (check email daily)

6. [ ] **Sign DUA when approved**

7. [ ] **Access data via BigQuery** (recommended)

8. [ ] **Run cancer type analysis on full dataset**
   - Validate 4-biomarker model
   - Test metabolic theory hypothesis
   - Re-add CRP and BMI with better data

---

## Resources & Links

### Official Documentation
- [PhysioNet Home](https://physionet.org/)
- [MIMIC-IV Project Page](https://physionet.org/content/mimiciv/)
- [MIMIC Documentation](https://mimic.mit.edu/docs/gettingstarted/)
- [Application Process](https://physionet.org/news/post/395)
- [CITI Program](https://about.citiprogram.org/)

### Support
- **PhysioNet Support**: contact@physionet.org
- **MIMIC Forum**: https://github.com/MIT-LCP/mimic-code/issues
- **CITI Support**: https://support.citiprogram.org/

### Your Project Files
- **Demo Validation**: `external_datasets/mimic_iv_demo/FINAL_VALIDATION_REPORT.md`
- **Cancer Type Analysis**: `external_datasets/mimic_iv_demo/CANCER_TYPE_ANALYSIS_REPORT.md`
- **CRP Subset Analysis**: `external_datasets/mimic_iv_demo/CRP_SUBSET_ANALYSIS_REPORT.md`

---

## Tips for Success

✅ **DO**:
- Use institutional email if possible (increases credibility)
- Write clear, scientific research purpose
- Provide supervisor contact if student/postdoc
- Double-check all information before submitting
- Keep CITI certificate (valid 3 years)
- Read DUA carefully before signing
- Use cloud access for faster analysis

❌ **DON'T**:
- Rush the CITI training (understand the content)
- Leave required fields blank
- Provide vague research purpose
- Share data with non-credentialed users
- Attempt to re-identify patients
- Download data to insecure locations
- Share login credentials

---

## Estimated Timeline for Your Project

**Week 1**: Complete CITI training + submit credentialing
**Week 2**: Approval received, sign DUA, set up BigQuery
**Week 3-4**: Re-run all analyses on full dataset
**Week 5-8**: Test cancer-specific models, add CRP+BMI, optimize
**Week 9-12**: Validate on external datasets, write paper

**Total**: 3 months from application to publication-ready results

---

## Questions?

If you have questions during the application process, I can help with:
- Drafting research purpose statements
- Explaining CITI training modules
- Troubleshooting application issues
- Setting up BigQuery access after approval
- Writing SQL queries for MIMIC-IV analysis
- Re-running your cancer prediction models on full dataset

**Let's get you that access!** 🚀

---

**Status**: Ready to start
**Next Action**: Begin CITI training at https://about.citiprogram.org/
