# UK Biobank Data Access

UK Biobank is a large-scale biomedical database containing genetic, metabolic, and clinical data from 500,000+ participants.

## Overview

- **Study**: UK Biobank
- **Participants**: 500,000+ (ages 40-69 at recruitment)
- **Data Types**: Genomics, imaging, metabolomics, clinical, lifestyle
- **Metabolomics**: 250+ circulating metabolites via NMR spectroscopy
- **Access**: Requires formal application and approval
- **URL**: https://www.ukbiobank.ac.uk/

## What Data is Available

### Metabolomics Data

- **Platform**: Nightingale Health NMR spectroscopy
- **Metabolites**: 250+ biomarkers including:
  - Lipids and lipoproteins
  - Amino acids
  - **Glucose**
  - Glycolysis-related metabolites
  - Ketone bodies
  - Inflammation markers

**Note**: Check if lactate and LDH are specifically included in the NMR panel.

### Clinical Data

- Cancer diagnoses (ICD-10 codes)
- Hospital episode statistics
- Cancer registry linkage
- Mortality data
- Primary care data

### Other Relevant Data

- Demographics (age, sex, ethnicity)
- Anthropometrics (BMI, weight, height)
- Lifestyle factors
- Physical activity
- Diet
- Blood pressure
- Genomic data (WGS available)

## Access Process

### Step 1: Determine Eligibility

UK Biobank data is available to:
- ✓ Academic researchers
- ✓ Commercial researchers
- ✓ Charities
- ✓ International applicants

All applicants must demonstrate research is in the public interest.

### Step 2: Register and Create Account

1. Go to https://www.ukbiobank.ac.uk/enable-your-research/register
2. Create an account on the Access Management System (AMS)
3. You'll receive a confirmation email

### Step 3: Prepare Your Application

You'll need:

1. **Research Proposal** (2-3 pages):
   - Background and rationale
   - Study aims and objectives
   - Proposed analysis methods
   - Public health impact

2. **Data Fields Required**:
   - Specific metabolites needed
   - Clinical diagnoses
   - Demographics
   - Other variables

3. **Institutional Information**:
   - Principal Investigator details
   - Institution name and address
   - Ethics committee information

4. **Collaborators** (if applicable)

### Step 4: Submit Application

1. Log in to AMS: https://ams.ukbiobank.ac.uk/
2. Click "New Application"
3. Complete the application form:
   - Project title
   - Duration (typically 3 years, renewable)
   - Lay summary (for public)
   - Research proposal
   - Data fields requested
   - Ethical approval information

4. Submit application

### Step 5: Application Review

- **Initial review**: 2-4 weeks
- **Possible outcomes**:
  - Approved
  - Revisions requested
  - Additional information needed
  - Rejected (rare if well-justified)

- **Review criteria**:
  - Scientific merit
  - Public benefit
  - Feasibility
  - Appropriate data requested

### Step 6: Pay Access Fee

Once approved, pay the access fee:

- **Academic**: £2,500 for 3 years (as of 2025)
- **Commercial**: Higher fees apply
- **Renewal**: Additional fees for extensions

### Step 7: Access Data

After payment:

1. Receive access to UK Biobank Research Analysis Platform (RAP)
2. Cloud-based environment (DNAnexus platform)
3. Data accessed via:
   - Web interface
   - Command-line tools
   - Python/R APIs

## Data Download and Access

### UK Biobank RAP (Research Analysis Platform)

Data is accessed via cloud platform, not direct download:

```bash
# Install UK Biobank tools
pip install dxpy

# Login to platform
dx login

# List available datasets
dx ls

# Download specific dataset
dx download project-xxx:/path/to/dataset.csv
```

### Data Structure

Typical data format:

```csv
eid,age,sex,cancer_icd10,glucose,metabolite_1,metabolite_2,...
1000001,55,F,C50.9,5.2,1.23,4.56,...
1000002,62,M,,5.8,1.45,4.23,...
```

- `eid`: Participant ID (encrypted)
- Cancer diagnoses: ICD-10 codes
- Metabolites: Continuous values

## Specific Data Fields for Cancer Metabolism Research

### Metabolomics Fields

Search for these field IDs in UK Biobank showcase (https://biobank.ndph.ox.ac.uk/showcase/):

- Field 23400-23650: NMR metabolomics results
- Look for:
  - Glucose (should be included)
  - Lactate (verify availability)
  - Glycolysis markers
  - Amino acids

### Cancer Diagnosis Fields

- Field 40006: Cancer ICD-10 codes
- Field 40005: Cancer diagnosis date
- Field 40011: Cancer histology
- Field 40012: Cancer behavior

### Anthropometrics

- Field 21001: BMI
- Field 31: Sex
- Field 21022: Age at recruitment

## Processing Script Template

After gaining access, create processing script:

```python
# scripts/process_ukbiobank.py
import pandas as pd

def load_ukbiobank_data(csv_path):
    """Load and process UK Biobank data"""
    df = pd.read_csv(csv_path)

    # Filter for cancer patients
    cancer_df = df[df['cancer_icd10'].notna()]

    # Extract metabolic markers
    biomarkers = [
        'glucose',
        'lactate',  # if available
        # add other metabolite field names
    ]

    return cancer_df[['eid', 'age', 'sex', 'cancer_icd10'] + biomarkers]

# Usage
data = load_ukbiobank_data('../ukbiobank/ukb_data.csv')
data.to_csv('../ukbiobank/processed_cancer_metabolites.csv', index=False)
```

## Important Restrictions

1. **No Data Export**: Data must stay on RAP platform
2. **No Redistribution**: Cannot share raw data
3. **Approved Use Only**: Only for stated research purpose
4. **Publication Requirements**: Must acknowledge UK Biobank
5. **Return Results**: Must return aggregate results to UK Biobank

## Cost Estimate

| Item | Cost (Academic) | Timeline |
|------|-----------------|----------|
| Application | Free | 2-4 weeks |
| Access fee (3 years) | £2,500 | - |
| Compute costs | Pay-as-you-go | Varies |
| Storage | Pay-as-you-go | Varies |

Compute costs: ~£100-500 depending on analysis complexity

## Citation

If you use UK Biobank data:

```
This research has been conducted using the UK Biobank Resource under
Application Number [YOUR_NUMBER].
```

## Checking Metabolite Availability

Before applying, check the data showcase:

1. Go to https://biobank.ndph.ox.ac.uk/showcase/
2. Search for "NMR metabolomics"
3. Review Field ID 23400-23650
4. Check if lactate and LDH are included
5. Download data dictionary

## Tips for Successful Application

1. **Clear hypothesis**: State specific research question
2. **Public benefit**: Emphasize health impact
3. **Realistic scope**: Don't request unnecessary data
4. **Preliminary data**: Include supporting evidence
5. **Timeline**: Provide realistic analysis timeline
6. **Collaboration**: UK-based collaborator can help

## Alternative: UK Biobank Data Showcase

For initial exploration:

- Browse data summaries at https://biobank.ndph.ox.ac.uk/showcase/
- No login required
- See aggregate statistics
- Identify relevant fields
- Plan your application

## Contact Information

- **Email**: access@ukbiobank.ac.uk
- **Phone**: +44 (0)161 475 5395
- **Address**: UK Biobank, Stockport, UK

## Expected Timeline

| Step | Duration |
|------|----------|
| Account creation | Same day |
| Application prep | 1-2 weeks |
| Submission | 1 hour |
| Review | 2-4 weeks |
| Payment processing | 1 week |
| Access granted | Same day after payment |
| **Total** | **4-8 weeks** |

## Related Resources

- UK Biobank Homepage: https://www.ukbiobank.ac.uk/
- Data Showcase: https://biobank.ndph.ox.ac.uk/showcase/
- Research Analysis Platform: https://ukbiobank.dnanexus.com/
- Publications: https://www.ukbiobank.ac.uk/enable-your-research/publications
- Tutorials: https://dnanexus.gitbook.io/uk-biobank-rap/

## Next Steps

1. ✓ Review data showcase to confirm metabolite availability
2. ✓ Draft research proposal (2-3 pages)
3. ✓ Identify all required data fields
4. ✓ Secure institutional support/PI
5. ✓ Submit application
6. ⏳ Wait for approval (2-4 weeks)
7. ⏳ Pay access fee
8. ⏳ Access data via RAP platform
