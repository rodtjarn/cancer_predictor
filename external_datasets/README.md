# External Cancer Datasets with Biomarkers

This directory contains scripts and instructions for downloading real patient data from major cancer research databases containing metabolic biomarkers (lactate, LDH, CRP, glucose).

## Quick Start

```bash
# 1. Install dependencies
cd external_datasets
pip install -r requirements.txt

# 2. List available datasets
cd scripts
python download_all.py --list

# 3. Download TCGA data (no credentials needed)
python download_all.py --tcga

# 4. Download MIMIC-IV (requires PhysioNet credentials)
export PHYSIONET_USERNAME="your_username"
export PHYSIONET_PASSWORD="your_password"
python download_all.py --mimic
```

## Directory Structure

```
external_datasets/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── scripts/                     # Download scripts
│   ├── download_all.py         # Master script
│   ├── download_tcga.py        # TCGA downloader
│   ├── download_mimic.py       # MIMIC-IV downloader
│   ├── PLCO_INSTRUCTIONS.md    # PLCO access guide
│   └── UKBIOBANK_INSTRUCTIONS.md  # UK Biobank guide
├── tcga/                       # TCGA data (auto-created)
├── mimic/                      # MIMIC-IV data (auto-created)
├── plco/                       # PLCO data (manual)
└── ukbiobank/                  # UK Biobank data (manual)
```

## Available Datasets

### 1. TCGA (The Cancer Genome Atlas) ✅ Automated

**What it contains:**
- 11,000+ cancer patients across 33 cancer types
- Clinical data: demographics, diagnosis, staging
- Limited laboratory values
- Genomic/transcriptomic data (not downloaded by default)

**Biomarkers available:**
- Age, BMI, basic clinical data
- Gene expression data for LDH genes (LDHA, LDHB)
- Note: Direct serum lactate/LDH measurements limited

**Access:**
- Free, no approval needed
- Downloads via GDC API

**Script:** `download_tcga.py`

**Time to data:** 30-60 minutes

### 2. MIMIC-IV (PhysioNet) ✅ Semi-Automated

**What it contains:**
- 365,000+ ICU/ED patients (2008-2022)
- Complete lab test results including:
  - ✅ Lactate (item ID 50813)
  - ✅ Glucose (item ID 50931)
  - ❓ CRP (availability TBD)
  - ❓ LDH (availability TBD)
- ICD-9/ICD-10 diagnosis codes (for cancer identification)

**Access:**
- Free after PhysioNet credentialing
- Requires CITI training (~3-4 hours)
- Approval takes ~1 week

**Script:** `download_mimic.py`

**Time to data:** 1 week setup + 1-3 hours download

### 3. PLCO Cancer Screening Trial ⏳ Manual

**What it contains:**
- 155,000+ participants
- Multiple cancer types: prostate, lung, colorectal, ovarian
- Clinical and biomarker data (varies by sub-study)

**Biomarkers available:**
- Varies by dataset
- Check data dictionaries before requesting
- May include standard lab panels

**Access:**
- Free, requires data request approval
- 1-2 weeks approval time

**Instructions:** `PLCO_INSTRUCTIONS.md`

**Time to data:** 2-4 weeks

### 4. UK Biobank ⏳ Manual + Paid

**What it contains:**
- 500,000+ participants
- 250+ metabolites via NMR spectroscopy
- Cancer registry linkage
- Comprehensive phenotyping

**Biomarkers available:**
- ✅ Glucose
- ❓ Lactate (verify in data showcase)
- ❓ LDH (verify in data showcase)
- Many other metabolites

**Access:**
- Requires formal application
- £2,500 fee (academic, 3 years)
- Cloud-based access (no direct download)

**Instructions:** `UKBIOBANK_INSTRUCTIONS.md`

**Time to data:** 4-8 weeks

## Installation

```bash
# From external_datasets directory
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- pandas
- requests
- numpy

## Usage Examples

### List All Datasets

```bash
cd scripts
python download_all.py --list
```

### Download TCGA Only

```bash
python download_all.py --tcga
```

Output: `../tcga/tcga_all_clinical.csv`

### Download MIMIC-IV (with credentials)

```bash
# Set credentials first
export PHYSIONET_USERNAME="your_username"
export PHYSIONET_PASSWORD="your_password"

# Download
python download_all.py --mimic
```

Output:
- `../mimic/patients.csv.gz`
- `../mimic/labevents.csv.gz`
- `../mimic/cancer_patient_labs.csv`

### Download Both TCGA and MIMIC

```bash
python download_all.py --tcga --mimic
```

### Show Manual Access Instructions

```bash
python download_all.py --plco
python download_all.py --ukbiobank
```

## Individual Script Usage

### TCGA Downloader

```bash
cd scripts
python download_tcga.py
```

Features:
- Downloads clinical data for all TCGA projects
- Saves individual project files + combined file
- Includes demographics, diagnosis, staging
- ~500MB total download

### MIMIC-IV Downloader

```bash
cd scripts

# Full download (includes large lab events file)
python download_mimic.py

# Or use as a library
python -c "
from download_mimic import MIMICDownloader
dl = MIMICDownloader()
dl.check_access()
"
```

Features:
- Verifies PhysioNet credentials
- Downloads core tables
- Extracts cancer patients automatically
- Filters lab values for biomarkers
- WARNING: labevents.csv.gz is ~8GB

## Data Processing

After downloading, process the data:

```bash
# TCGA
cd ../tcga
head -20 tcga_all_clinical.csv

# MIMIC
cd ../mimic
gunzip -c patients.csv.gz | head -20
```

### Merging Datasets

Create a unified dataset:

```python
import pandas as pd

# Load TCGA
tcga = pd.read_csv('tcga/tcga_all_clinical.csv')

# Load MIMIC cancer patients with labs
mimic = pd.read_csv('mimic/cancer_patient_labs.csv')

# Process and merge
# ... your processing code ...
```

## Expected Output Files

### TCGA
```
tcga/
├── TCGA-BRCA_clinical.csv      # Breast cancer
├── TCGA-LUAD_clinical.csv      # Lung adenocarcinoma
├── TCGA-PAAD_clinical.csv      # Pancreatic cancer
├── ...                         # Other cancer types
└── tcga_all_clinical.csv       # Combined (all projects)
```

### MIMIC-IV
```
mimic/
├── patients.csv.gz             # Patient demographics
├── admissions.csv.gz           # Hospital admissions
├── diagnoses_icd.csv.gz        # Diagnosis codes
├── d_icd_diagnoses.csv.gz      # ICD code dictionary
├── labevents.csv.gz            # All lab tests (LARGE)
├── d_labitems.csv.gz           # Lab test dictionary
├── cancer_patient_ids.csv      # Extracted cancer patients
└── cancer_patient_labs.csv     # Labs for cancer patients
```

### PLCO (after manual download)
```
plco/
├── plco_demographics.csv
├── plco_biomarkers.csv
└── plco_outcomes.csv
```

### UK Biobank (cloud access only)
```
Access via UK Biobank RAP platform
No local files (cloud-based analysis)
```

## Biomarker Coverage Summary

| Dataset | Lactate | Glucose | CRP | LDH | Other |
|---------|---------|---------|-----|-----|-------|
| TCGA | Gene expression | Age, BMI | - | Gene expression | Clinical |
| MIMIC-IV | ✅ | ✅ | ❓ | ❓ | Full labs |
| PLCO | ❓ | ❓ | ❓ | ❓ | Variable |
| UK Biobank | ❓ | ✅ | ❓ | ❓ | 250+ metabolites |

Legend: ✅ Confirmed | ❓ Check documentation | - Not available

## Troubleshooting

### TCGA Download Fails

```bash
# Check connection
curl -I https://api.gdc.cancer.gov/projects

# Try single project
python -c "
from download_tcga import TCGADownloader
dl = TCGADownloader()
dl.download_project_clinical_data('TCGA-BRCA')
"
```

### MIMIC-IV Authentication Error

```bash
# Verify credentials
echo $PHYSIONET_USERNAME
echo $PHYSIONET_PASSWORD

# Test access
curl -u $PHYSIONET_USERNAME:$PHYSIONET_PASSWORD \
  https://physionet.org/files/mimiciv/3.1/README.txt
```

### Slow Downloads

MIMIC-IV files are large. For faster downloads:
1. Use wired internet connection
2. Download during off-peak hours
3. Consider downloading only needed tables

## Data Use Agreements

All datasets require compliance with data use agreements:

- **TCGA**: Free, open access, cite in publications
- **MIMIC-IV**: Sign PhysioNet DUA, no redistribution
- **PLCO**: Sign CDAS DUA, restricted use
- **UK Biobank**: Formal agreement, approved use only

**IMPORTANT**: Never share raw data publicly or redistribute.

## Citation

If you use these datasets, cite appropriately:

### TCGA
```
Data used in this study were generated by The Cancer Genome Atlas (TCGA)
Research Network: https://www.cancer.gov/tcga
```

### MIMIC-IV
```
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R.
(2023). MIMIC-IV (version 3.1). PhysioNet.
https://doi.org/10.13026/kpb8-g736
```

### PLCO
```
Data used in this research were obtained from the Cancer Data Access System
(CDAS) sponsored by the National Cancer Institute.
```

### UK Biobank
```
This research has been conducted using the UK Biobank Resource under
Application Number [YOUR_NUMBER].
```

## Next Steps

1. **Download available datasets**
   ```bash
   python download_all.py --tcga --mimic
   ```

2. **Apply for manual access datasets**
   - Follow instructions in `*_INSTRUCTIONS.md`
   - Allow 1-8 weeks for approval

3. **Explore the data**
   - Check data quality
   - Identify available biomarkers
   - Calculate coverage statistics

4. **Retrain your model**
   - Use real patient data
   - Validate against test set
   - Compare to synthetic data performance

5. **Report findings**
   - Document any discrepancies
   - Share insights (aggregate data only)
   - Publish results with proper citations

## Support

### TCGA Issues
- GDC Support Portal: https://gdc.cancer.gov/support

### MIMIC-IV Issues
- PhysioNet Forums: https://physionet.org/forums/

### PLCO Issues
- Email: NCICDASHelp@mail.nih.gov

### UK Biobank Issues
- Email: access@ukbiobank.ac.uk

## Contributing

Found a bug or improvement? Please report in the main repository.

## License

Download scripts: MIT License
Datasets: Each has its own terms of use (see individual agreements)

---

Last updated: 2025-12-31
