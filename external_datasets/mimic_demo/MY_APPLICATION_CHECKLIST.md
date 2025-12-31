# My MIMIC-IV Application Checklist

**Started**: 2025-12-31
**Goal**: Get access to MIMIC-IV dataset with all 7 biomarkers
**Timeline**: 1-2 weeks

---

## ✅ Step 1: PhysioNet Account
- [x] Created PhysioNet account
- **Status**: DONE ✅

---

## Step 2: Create ORCID Account (5 minutes) - DO NOW

**Why**: Alternative to institutional email for independent researchers

**Go here**: https://orcid.org/register

**Action**:
1. Fill out registration form:
   - Email address
   - First name
   - Last name
   - Password
2. Verify email
3. Add to your profile:
   - Employment (any research/tech work)
   - Education (highest degree)
   - Biography: "Independent researcher working on cancer biomarker analysis and machine learning models for early cancer detection using metabolic markers"

**Save your ORCID iD**: Looks like `https://orcid.org/0000-0001-2345-6789`

- [ ] Created ORCID account
- [ ] Verified email
- [ ] Added biography
- [ ] Saved ORCID iD: ________________

---

## Step 3: Start CITI Training (3-4 hours) - START TODAY

**Why**: Required ethics training for using patient data

**Go here**: https://physionet.org/about/citi-course/

**Action**:
1. Click "CITI Course in Data or Specimens Only Research"
2. You'll be redirected to CITIPROGRAM.org
3. Create account (if needed) or login
4. Select organization: "Not affiliated" (for independent researchers)
5. Select course: "Data or Specimens Only Research"
6. Complete modules (~3-4 hours total):
   - Can skim most content
   - Take quizzes (80% to pass)
   - Can pause and resume
7. Download completion certificate PDF when done

**Tips**:
- Don't overthink it - basic research ethics
- Quizzes are multiple choice
- Can retry if you fail
- Power through in one session if possible

- [ ] Started CITI training
- [ ] Completed all modules
- [ ] Downloaded certificate PDF
- [ ] Certificate saved at: ________________

**Time spent**: _______ hours

---

## Step 4: PhysioNet Credentialing (15 minutes) - AFTER CITI

**Go here**: https://physionet.org/settings/credentialing/

**Action**:
1. Upload CITI completion certificate (from Step 3)
2. Link your ORCID account (from Step 2)
3. Fill out credentialing form:
   - Research purpose: "Cancer biomarker analysis and validation of machine learning models using metabolic markers (lactate, LDH, CRP, glucose) for early cancer detection"
4. Submit application

**Reference Options** (choose ONE):
- [x] Use ORCID (easiest for independent researchers)
- [ ] Get reference letter from colleague/professor
- [ ] Use institutional email (.edu or .org)

- [ ] Uploaded CITI certificate
- [ ] Linked ORCID account
- [ ] Filled out research purpose
- [ ] Submitted credentialing application
- [ ] Received confirmation email

**Wait time**: 1-2 days for approval

---

## Step 5: Sign MIMIC-IV Data Use Agreement (5 minutes) - AFTER CREDENTIALING

**Wait for**: Credentialing approval email

**Go here**: https://physionet.org/content/mimiciv/3.1/

**Action**:
1. After credentialing approved, you'll see "Sign the Data Use Agreement" button
2. Read the terms:
   - Use for research only ✓
   - Don't redistribute raw data ✓
   - Cite MIMIC-IV in publications ✓
   - Maintain patient privacy ✓
   - Don't try to re-identify patients ✓
3. Accept and submit

- [ ] Credentialing approved (received email)
- [ ] Signed Data Use Agreement
- [ ] Submitted DUA

**Wait time**: 5-7 days for MIMIC-IV access approval

---

## Step 6: Download MIMIC-IV Data (2-3 hours) - AFTER DUA APPROVED

**Wait for**: MIMIC-IV access approval email

**Action**:
1. Set credentials in terminal:
```bash
export PHYSIONET_USERNAME="your_username"
export PHYSIONET_PASSWORD="your_password"
```

2. Run download script:
```bash
cd /Users/per/work/claude/cancer_predictor_package/external_datasets/scripts
python download_mimic.py
```

**Files to download** (~8 GB total):
- patients.csv.gz (~20MB) - Demographics
- admissions.csv.gz (~50MB) - Hospital admissions
- diagnoses_icd.csv.gz (~200MB) - Diagnosis codes
- d_icd_diagnoses.csv.gz (~5MB) - Diagnosis dictionary
- labevents.csv.gz (~8GB) - Lab test results ⭐ LARGE!
- d_labitems.csv.gz (~100KB) - Lab test dictionary

- [ ] Received MIMIC-IV access approval email
- [ ] Set PhysioNet credentials
- [ ] Downloaded all files
- [ ] Extracted files

**Download time**: 2-3 hours (depending on connection)

---

## Step 7: Test Model on Real Data! (30 minutes)

**Action**:
1. Extract cancer patients:
```bash
cd /Users/per/work/claude/cancer_predictor_package
python -c "from external_datasets.scripts.download_mimic import MIMICDownloader; d = MIMICDownloader(); d.extract_cancer_patients()"
```

2. Run model test:
```bash
python test_model_on_mimic.py
```

**Expected Results**:
- Accuracy: 85-95% (vs 55% on UCI, 98.8% on synthetic)
- Sensitivity: 90-95%
- Specificity: 90-95%
- Sample size: 15,000-30,000 cancer patients with lactate
- All 7/7 biomarkers available

- [ ] Extracted cancer patients
- [ ] Ran model test
- [ ] Results: Accuracy = ______%
- [ ] Results saved
- [ ] Model validated! 🎉

---

## Timeline Summary

| Step | Time | Status | Date Completed |
|------|------|--------|----------------|
| 1. PhysioNet account | - | ✅ DONE | (already done) |
| 2. ORCID account | 5 min | ⏳ TODO | __________ |
| 3. CITI training | 3-4 hrs | ⏳ TODO | __________ |
| 4. Credentialing | 15 min | ⏳ TODO | __________ |
| 5. Sign DUA | 5 min | ⏳ WAITING | __________ |
| 6. Download data | 2-3 hrs | ⏳ WAITING | __________ |
| 7. Test model | 30 min | ⏳ WAITING | __________ |

**Total Work Time**: ~4-5 hours
**Total Wait Time**: ~1-2 weeks
**Total Cost**: $0 (FREE)

---

## Current Status

**Today's Date**: 2025-12-31

**Next Action**:
1. Create ORCID account: https://orcid.org/register
2. Start CITI training: https://physionet.org/about/citi-course/

**Can do both in parallel!**

---

## Troubleshooting

**Problem**: "Don't have a reference letter"
**Solution**: Use ORCID instead! Add employment/education to your profile.

**Problem**: "CITI course too long"
**Solution**: Can pause and resume anytime. Just power through - it's multiple choice.

**Problem**: "Credentialing taking too long"
**Solution**: Usually 1-2 days. If >3 days, email support@physionet.org

**Problem**: "Download too slow"
**Solution**: Use wired connection, download during off-peak hours.

---

## Key Links

- **ORCID Registration**: https://orcid.org/register
- **CITI Training**: https://physionet.org/about/citi-course/
- **PhysioNet Credentialing**: https://physionet.org/settings/credentialing/
- **MIMIC-IV Page**: https://physionet.org/content/mimiciv/3.1/
- **PhysioNet Support**: support@physionet.org

---

## Why This Is Worth It

**What you're getting**:
- ✅ 365,000 patient records
- ✅ 15,000-30,000 cancer patients with lactate
- ✅ All 7 biomarkers confirmed present
- ✅ Ability to validate Warburg effect hypothesis
- ✅ Expected 85-95% model accuracy (vs 55% on UCI)
- ✅ Real clinical validation for your research

**What it costs**:
- 4-5 hours of work
- 1-2 weeks of waiting
- $0 money

**ROI**: PRICELESS 🎯

---

## Notes

(Use this space to track progress, questions, issues)

**Application started**: 2025-12-31
**ORCID created**: __________
**CITI started**: __________
**CITI completed**: __________
**Credentialing submitted**: __________
**Credentialing approved**: __________
**DUA signed**: __________
**MIMIC access granted**: __________
**Data downloaded**: __________
**Model tested**: __________
**SUCCESS!**: __________

---

**Remember**: You've already proven the data exists (demo analysis). Now you just need to complete the paperwork to get access to the full dataset.

**You got this!** 🚀
