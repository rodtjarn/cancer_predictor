# Literature Search Report: Patient-Level Metabolic Biomarker Datasets

**Date**: 2025-12-31  
**Objective**: Find publicly available datasets with individual patient-level measurements of lactate, glucose, LDH, CRP for cancer patients

---

## Executive Summary

❌ **Result**: No publicly available datasets found with complete individual patient-level data for all required biomarkers

✅ **What We Found**: Multiple databases and repositories with partial data or aggregated statistics

⚠️ **Conclusion**: Individual patient-level metabolic data remains largely unpublished or restricted

---

## Repositories Searched

### 1. MACdb (Cancer Metabolic Biomarker Database) ✅ EXPLORED
- **URL**: https://ngdc.cncb.ac.cn/macdb/
- **Coverage**: 40,710 metabolite measurements, 1,127 studies
- **Biomarkers Found**: Lactate ✅, Glucose ✅
- **Problem**: AGGREGATED data (group means), not individual patients
- **Usefulness**: Literature validation only

### 2. MetaboLights ⚠️ IDENTIFIED
- **URL**: https://www.ebi.ac.uk/metabolights/
- **Datasets Identified**:
  - MTBLS11849: Lactate in cancer stemness
  - MTBLS3338: Breast cancer metabolic reprogramming  
  - MTBLS7260: Pancreatic cancer microbiome
- **Problem**: Requires JavaScript; couldn't verify individual patient data
- **Status**: Needs manual exploration

### 3. cBioPortal 🔍 PARTIALLY EXPLORED
- **URL**: https://www.cbioportal.org/datasets
- **Focus**: Cancer genomics + clinical data
- **Clinical Data**: Free-form patient attributes (varies by study)
- **LDH/Lactate**: May be available in some studies
- **Problem**: Primarily genomic data; clinical metabolic markers not standardized
- **Status**: Requires dataset-by-dataset exploration

### 4. GEO (Gene Expression Omnibus) ❌ EXPLORED
- **Datasets Checked**: GSE15459 (gastric cancer, 200 samples)
- **Problem**: Gene expression data (mRNA), NOT metabolomics
- **Supplementary Files**: Clinical outcomes, but not metabolic measurements
- **Status**: Wrong data type for our needs

### 5. CRDC (Cancer Research Data Commons) 📋 IDENTIFIED
- **URL**: https://datacommons.cancer.gov/
- **Coverage**: NCI/NIH-funded datasets
- **Status**: Needs exploration; may require data access applications

---

## Published Studies Reviewed

### Studies with LDH + CRP Measurements:

1. **Prognostic Value of Combined Biomarkers in Metastatic Breast Cancer** (PMC8743925)
   - Biomarkers: LDH, CRP, CA 15-3, CA 125
   - Patients: Metastatic breast cancer
   - Data Availability: ❌ Supplementary files have summary stats only, no individual patient data

2. **Laboratory Cachexia Score in Lung Cancer** (BMC Cancer, 2025)
   - Biomarkers: LDH, CRP, Albumin
   - Patients: 261 advanced lung cancer patients
   - Data Availability: ❌ Not mentioned; likely aggregated

3. **New Pancreatic Cancer Survival Model** (PMC11491290, 2024)
   - Biomarkers: GGT, LDH, glucose
   - Patients: Pancreatic ductal adenocarcinoma
   - Data Availability: ❌ Clinical/biochemical data mentioned but not shared

---

## Key Findings

### ✅ What EXISTS in Literature:

1. **LDH is widely measured** in cancer studies
   - Common prognostic marker
   - Measured in blood/serum
   - Present in hundreds of studies

2. **Lactate is frequently studied**
   - Warburg effect research
   - Cell metabolism studies
   - Often measured in tissue or blood

3. **Glucose is standard**
   - Routine clinical measurement
   - Diabetes/metabolic syndrome link to cancer
   - Widely available

4. **CRP is common**
   - Inflammation marker
   - Used in prognostic models
   - Routinely measured

### ❌ What's MISSING:

1. **Individual patient-level data is rarely published**
   - Studies publish summary statistics
   - Privacy concerns
   - Journal policies

2. **Complete biomarker panels are uncommon**
   - Studies measure 1-3 markers
   - Rarely all 7 needed for your model
   - Different studies measure different things

3. **Metabolomics data is siloed**
   - Cell line data (not patients)
   - Tissue samples (not blood)
   - Partial biomarker panels

---

## Why Individual Patient Data is Hard to Find

### 1. Privacy Regulations
- HIPAA in US
- GDPR in Europe
- De-identification requirements

### 2. Journal Policies
- Summary statistics sufficient for publication
- Raw data sharing not mandated (though encouraged)
- Supplementary file size limits

### 3. Data Sharing Culture
- Improving but still incomplete
- Researchers reluctant to share
- Lack of standardization

### 4. Commercial Interests
- Biotech/pharma proprietary data
- Competitive advantage
- IP concerns

---

## Comparison: What We've Tried

| Source | Lactate | Glucose | LDH | CRP | Spec Grav | Age | BMI | Individual Data? | Can Test Model? |
|--------|---------|---------|-----|-----|-----------|-----|-----|------------------|-----------------|
| UCI Breast Cancer | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ Yes | ✅ Yes (55% acc) |
| TCGA | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❓ | ✅ Yes | ❌ Too incomplete |
| MACdb | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❓ | ❌ Aggregated | ❌ No |
| Literature Search | ✅ (studies exist) | ✅ | ✅ (studies exist) | ✅ (studies exist) | ❌ | ✅ | ✅ | ❌ Not published | ❌ No |
| MIMIC-IV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Yes | ⏳ Pending access |

---

## Recommendations

### Short-term (Now):
1. ✅ **Use MACdb for literature validation**
   - Show that lactate/glucose ARE studied in cancer
   - Cite the 50 studies with both biomarkers
   - Demonstrate biomarker importance

2. ✅ **Manually explore MetaboLights**
   - Visit MTBLS11849, MTBLS3338, MTBLS7260
   - Check if downloadable individual-level data exists
   - May find cell-line or tissue data

3. ⏳ **Wait for PhysioNet/MIMIC-IV**
   - Best chance for complete biomarker panel
   - Individual patient-level data guaranteed
   - All 7 biomarkers available

### Medium-term (If MIMIC-IV fails):
1. **Contact study authors directly**
   - Email corresponding authors from MACdb studies
   - Request de-identified individual patient data
   - May work for some researchers

2. **Apply for restricted datasets**
   - Some biobanks have data sharing agreements
   - UK Biobank, All of Us, etc.
   - Requires formal application

3. **Collaborate with clinical researchers**
   - Partner with hospital/clinic
   - Prospective data collection
   - IRB approval needed

---

## Specific Datasets Worth Exploring

### High Priority:
1. **MetaboLights MTBLS11849** - Lactate in cancer (needs manual check)
2. **MetaboLights MTBLS3338** - Breast cancer metabolomics (needs manual check)
3. **cBioPortal specific studies** - Check for clinical LDH/CRP data

### Medium Priority:
4. **UK Biobank** - Large cohort, may have metabolic data (restricted access)
5. **All of Us Research Program** - US precision medicine initiative (restricted)
6. **European Genome-phenome Archive (EGA)** - May have metabolomics (restricted)

---

## Key Insights

### What We Learned:

1. **Your biomarker choices are validated**
   - Lactate, glucose, LDH, CRP are all actively studied in cancer
   - Published in hundreds of papers
   - Recognized as important markers

2. **The data exists but isn't publicly shared**
   - Researchers DO measure these biomarkers
   - Data is kept private or aggregated for publication
   - Individual patient data is the exception, not the norm

3. **MIMIC-IV is genuinely special**
   - Most clinical databases don't share raw patient data
   - MIMIC-IV's open model is unusual
   - Worth the effort to access

---

## Bottom Line

**The literature search confirms**:
- ✅ Your biomarkers (lactate, glucose, LDH, CRP) ARE important in cancer research
- ✅ Dozens of studies measure these markers
- ❌ Individual patient-level data is NOT publicly available in standard repositories

**This means**:
1. You can cite extensive literature support for your biomarker choices
2. You CANNOT test your model on publicly available literature datasets
3. MIMIC-IV remains the best (and perhaps only) option for proper validation

**Next Action**: 
Continue pursuing MIMIC-IV access while using MACdb data for literature validation and background justification.

---

## Files Generated

- `macdb_metabolites.tsv` - MACdb metabolite data
- `macdb_studies.tsv` - MACdb study metadata
- `MACDB_ANALYSIS_REPORT.md` - MACdb analysis
- `LITERATURE_SEARCH_REPORT.md` - This report

---

**Sources**:
- [MACdb Database](https://ngdc.cncb.ac.cn/macdb/)
- [MetaboLights](https://www.ebi.ac.uk/metabolights/)
- [cBioPortal](https://www.cbioportal.org/)
- [GEO Database](https://www.ncbi.nlm.nih.gov/geo/)
- [CRDC](https://datacommons.cancer.gov/)
- [LDH in Breast Cancer Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8743925/)
- [Lung Cancer Cachexia Study](https://link.springer.com/article/10.1186/s12885-025-13426-3)
- [GEO GSE15459](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15459)
- [Metabolomics Review](https://www.nature.com/articles/s41467-024-46043-y)

