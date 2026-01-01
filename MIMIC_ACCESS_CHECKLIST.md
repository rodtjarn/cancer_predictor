# MIMIC-IV Access Application Checklist

**Goal**: Secure full MIMIC-IV access
**Target**: Complete by end of Week 1
**Last Updated**: 2026-01-01

---

## ✅ Step 1: CITI Training (2-4 hours)

- [ ] Go to https://about.citiprogram.org/
- [ ] Register account
- [ ] Select course: "Data or Specimens Only Research"
- [ ] Complete all required modules
- [ ] Pass all quizzes (80%+ required)
- [ ] Download completion report PDF
- [ ] Save PDF to safe location: `~/Downloads/CITI_Completion_Report.pdf`

**Status**: ⏳ Not Started
**Due**: Within 1-2 days

---

## ✅ Step 2: PhysioNet Credentialing (15 minutes)

- [ ] Create PhysioNet account: https://physionet.org/register/
- [ ] Verify email address
- [ ] Go to credentialing page: https://physionet.org/settings/credentialing/
- [ ] Fill out personal information
- [ ] Add reference information (if student/postdoc)
- [ ] Write research summary (use template from guide)
- [ ] Upload CITI completion report: https://physionet.org/settings/training/
- [ ] Review all information
- [ ] Submit application
- [ ] Save confirmation email

**Status**: ⏳ Waiting for Step 1
**Due**: Same day as Step 1 completion

---

## ✅ Step 3: Wait for Approval (3-7 days)

- [ ] Check email daily for approval notification
- [ ] If no response after 7 days, email contact@physionet.org
- [ ] Prepare cloud environment while waiting (optional)

**Status**: ⏳ Waiting for Step 2
**Expected**: 3-7 business days after submission

---

## ✅ Step 4: Sign Data Use Agreement (5 minutes)

- [ ] Receive approval email from PhysioNet
- [ ] Log in to PhysioNet
- [ ] Go to MIMIC-IV page: https://physionet.org/content/mimiciv/
- [ ] Navigate to Files section
- [ ] Read Data Use Agreement carefully
- [ ] Click "Sign" button
- [ ] Confirm access to files

**Status**: ⏳ Waiting for Step 3
**Due**: Same day as approval received

---

## ✅ Step 5: Set Up Data Access (30 minutes)

### Option A: Google BigQuery (Recommended)

- [ ] Create Google Cloud account: https://cloud.google.com/
- [ ] Enable BigQuery API
- [ ] Link PhysioNet account to Google Cloud
- [ ] Verify access to MIMIC-IV dataset in BigQuery
- [ ] Test query: `SELECT COUNT(*) FROM physionet-data.mimiciv_hosp.patients`

### Option B: Local Download (Not Recommended)

- [ ] Download MIMIC-IV files from PhysioNet (~40 GB)
- [ ] Extract files
- [ ] Set up local PostgreSQL database
- [ ] Load data into database

**Status**: ⏳ Waiting for Step 4
**Due**: Within 1 day of DUA signing

---

## ✅ Step 6: Validate Access (15 minutes)

- [ ] Run test query to count patients: `SELECT COUNT(*) FROM patients`
- [ ] Expected result: 73,181 patients
- [ ] Test cancer diagnosis query
- [ ] Verify biomarker availability (Glucose, Lactate, LDH, CRP)
- [ ] Estimate cancer patient count
- [ ] Confirm data quality is better than demo

**Status**: ⏳ Waiting for Step 5
**Due**: Same day as Step 5

---

## Progress Tracker

| Step | Status | Started | Completed | Duration |
|------|--------|---------|-----------|----------|
| 1. CITI Training | ⏳ Not Started | - | - | - |
| 2. Credentialing | ⏳ Not Started | - | - | - |
| 3. Approval Wait | ⏳ Not Started | - | - | - |
| 4. Sign DUA | ⏳ Not Started | - | - | - |
| 5. Setup Access | ⏳ Not Started | - | - | - |
| 6. Validate | ⏳ Not Started | - | - | - |

**Overall Progress**: 0/6 steps complete (0%)

---

## Key Information

### CITI Training
- **Course**: Data or Specimens Only Research
- **Link**: https://about.citiprogram.org/
- **Duration**: 2-4 hours
- **Validity**: 3 years
- **Cost**: Free (most institutions) or ~$50 (independent)

### PhysioNet
- **Account**: https://physionet.org/register/
- **Credentialing**: https://physionet.org/settings/credentialing/
- **Training Upload**: https://physionet.org/settings/training/
- **Support**: contact@physionet.org

### Research Purpose (Copy/Paste Ready)

```
Title: Validation of Metabolic Biomarker Panel for Cancer Detection

Purpose:
I am developing a machine learning model to detect cancer using routine blood
test biomarkers (glucose, lactate, LDH, age, CRP, BMI) based on the Warburg
effect. I have completed initial validation on the MIMIC-IV Demo dataset
(100 patients, 73.3% accuracy) and need access to the full MIMIC-IV dataset to:

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

---

## Next Actions

**TODAY**:
1. Start CITI training (2-4 hours)
2. Complete credentialing application

**THIS WEEK**:
- Wait for approval
- Prepare cloud environment

**NEXT WEEK**:
- Sign DUA
- Access data
- Re-run analyses

---

## Notes & Issues

<!-- Use this section to track any issues or notes during the process -->

**Date**: 2026-01-01
- Created application checklist
- Ready to start CITI training

---

**Status**: ⏳ Ready to Start
**Next Action**: Begin CITI training at https://about.citiprogram.org/
