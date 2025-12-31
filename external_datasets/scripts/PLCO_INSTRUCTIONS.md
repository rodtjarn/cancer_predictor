# PLCO Cancer Screening Trial Data Access

The Prostate, Lung, Colorectal and Ovarian (PLCO) Cancer Screening Trial dataset contains clinical and biomarker data for cancer research.

## Overview

- **Study**: PLCO Cancer Screening Trial
- **Participants**: ~155,000 participants
- **Data**: Clinical, biospecimen, imaging, and biomarker data
- **Access Portal**: Cancer Data Access System (CDAS)
- **URL**: https://cdas.cancer.gov/

## What Data is Available

- Demographics and baseline characteristics
- Cancer diagnoses and outcomes
- Screening test results
- Biomarker measurements (varies by study)
- Clinical lab values
- Mortality and follow-up data

## Access Process

### Step 1: Create an Account

1. Go to https://cdas.cancer.gov/
2. Click "Register" to create an account
3. Complete the registration form
4. Verify your email address

### Step 2: Review Available Datasets

1. Log in to CDAS
2. Navigate to "Datasets" → "PLCO"
3. Browse available datasets:
   - PLCO-21: Lung Cancer Dataset
   - PLCO-22: Ovarian Cancer Dataset
   - PLCO-23: Prostate Cancer Dataset
   - PLCO-24: Colorectal Cancer Dataset
   - And many others

### Step 3: Review Data Dictionaries

Before requesting data, review the data dictionaries:

1. Click on a dataset (e.g., PLCO-21 Lung)
2. Download the Data Dictionary (publicly available)
3. Check which biomarkers and lab values are included
4. Identify variables of interest:
   - Lactate measurements
   - LDH (lactate dehydrogenase)
   - CRP (C-reactive protein)
   - Glucose
   - Other metabolic markers

### Step 4: Submit Data Request

1. Click "Begin Data Request"
2. Fill out the request form:
   - **Project Title**: Your research project name
   - **Research Description**: Brief description (2-3 paragraphs)
   - **Specific Aims**: What you plan to do with the data
   - **Variables Requested**: Select specific variables or "all"
   - **Data Format**: Choose CSV or SAS

3. Provide institutional information:
   - Institution name
   - IRB information (if applicable)
   - PI contact information

4. Accept Data Use Agreement

5. Submit request

### Step 5: Wait for Approval

- Review time: Typically 1-2 weeks
- You'll receive email notification
- May require additional documentation

### Step 6: Download Data

Once approved:

1. Log in to CDAS
2. Go to "My Requests"
3. Click on your approved request
4. Download data files (CSV or SAS format)
5. Download associated documentation

## Data Download Script

After approval, save your downloaded files to:
```
external_datasets/plco/
```

### Automated Processing

Once you have the CSV files, use our processing script:

```bash
python scripts/process_plco.py \
  --input ../plco/raw_data.csv \
  --output ../plco/processed_biomarkers.csv
```

## Expected Biomarkers

Based on PLCO datasets, you may find:

- **Standard Labs**:
  - Complete blood count (CBC)
  - Chemistry panel
  - Glucose
  - Some enzyme measurements

- **Special Biomarkers** (varies by sub-study):
  - Inflammatory markers (possibly CRP)
  - Tumor markers (CA-125, PSA, etc.)
  - Metabolic markers (varies)

**Note**: Lactate and LDH availability varies by dataset. Check data dictionaries carefully.

## Important Notes

1. **IRB Approval**: Some requests may require IRB approval
2. **Data Use Agreement**: Must comply with all terms
3. **No Redistribution**: Cannot share raw data with others
4. **Publication**: Must acknowledge PLCO in publications
5. **Annual Reports**: May need to submit annual progress reports

## Contact Information

- **Email**: NCICDASHelp@mail.nih.gov
- **Phone**: 240-276-6800
- **Website**: https://cdas.cancer.gov/help/

## Expected Timeline

| Step | Estimated Time |
|------|----------------|
| Account creation | Same day |
| Data dictionary review | 1-2 hours |
| Request submission | 1 hour |
| Request approval | 1-2 weeks |
| Data download | 1-2 hours |

## Data Format Example

Expected CSV structure after download:

```csv
subject_id,age,gender,cancer_diagnosis,glucose,ldh,crp,lactate,...
10001,65,M,1,5.6,210,12.5,2.1,...
10002,58,F,0,5.1,180,3.2,1.5,...
```

## Processing Script Placeholder

Create this script after you receive and review the actual data structure.

## Tips for Success

1. **Be specific**: Clearly state why you need each variable
2. **Research purpose**: Emphasize public health benefit
3. **Institutional email**: Use official institution email
4. **Follow up**: If no response in 2 weeks, email help desk
5. **Start small**: Request a subset of variables for faster approval

## Citation

If you use PLCO data in publications:

```
Data used in this research were obtained from the Cancer Data Access System
(CDAS) sponsored by the National Cancer Institute. The Prostate, Lung,
Colorectal and Ovarian (PLCO) Cancer Screening Trial was supported by
contracts from the Division of Cancer Prevention, National Cancer Institute,
National Institutes of Health.
```

## Related Resources

- PLCO Study Website: https://prevention.cancer.gov/major-programs/plco
- CDAS Documentation: https://cdas.cancer.gov/learn/
- PLCO Publications: https://prevention.cancer.gov/major-programs/plco/publications
