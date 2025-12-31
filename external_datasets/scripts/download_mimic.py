"""
Download MIMIC-IV data from PhysioNet

This script downloads MIMIC-IV clinical database including lab test results
(lactate, glucose, CRP) and patient diagnoses.

IMPORTANT: Requires PhysioNet credentialing and data access approval.

Setup Instructions:
1. Create account at https://physionet.org/register/
2. Complete CITI training: https://physionet.org/about/citi-course/
3. Sign data use agreement for MIMIC-IV
4. Get credentialed (requires ~1 week)
5. Set environment variables:
   export PHYSIONET_USERNAME="your_username"
   export PHYSIONET_PASSWORD="your_password"

Requirements: requests, pandas, wget (optional)
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
import subprocess


class MIMICDownloader:
    """Download MIMIC-IV data from PhysioNet"""

    def __init__(self, output_dir='../mimic'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = 'https://physionet.org/files/mimiciv/3.1'

        # Check for credentials
        self.username = os.getenv('PHYSIONET_USERNAME')
        self.password = os.getenv('PHYSIONET_PASSWORD')

        if not self.username or not self.password:
            print("WARNING: PhysioNet credentials not found!")
            print("Please set environment variables:")
            print("  export PHYSIONET_USERNAME='your_username'")
            print("  export PHYSIONET_PASSWORD='your_password'")
            print("\nOr create a .env file with these credentials.")
            self.authenticated = False
        else:
            self.authenticated = True

    def check_access(self):
        """Check if user has access to MIMIC-IV"""
        if not self.authenticated:
            return False

        # Try to access a small file to verify credentials
        test_url = f'{self.base_url}/README.txt'
        try:
            response = requests.get(
                test_url,
                auth=(self.username, self.password),
                timeout=10
            )
            if response.status_code == 200:
                print("✓ PhysioNet authentication successful!")
                return True
            elif response.status_code == 401:
                print("✗ Authentication failed. Check your credentials.")
                return False
            elif response.status_code == 403:
                print("✗ Access denied. You may need to sign the MIMIC-IV data use agreement.")
                print("  Visit: https://physionet.org/content/mimiciv/")
                return False
            else:
                print(f"✗ Unexpected response: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False

    def download_file(self, filename, subdir='hosp'):
        """Download a specific file from MIMIC-IV"""
        if not self.authenticated:
            print("Cannot download - not authenticated")
            return False

        url = f'{self.base_url}/{subdir}/{filename}'
        output_file = self.output_dir / filename

        print(f"Downloading {filename}...")

        try:
            response = requests.get(
                url,
                auth=(self.username, self.password),
                stream=True,
                timeout=300
            )

            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(output_file, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"  Progress: {percent:.1f}%", end='\r')

                print(f"\n✓ Downloaded {filename} ({downloaded / 1024 / 1024:.1f} MB)")
                return True
            else:
                print(f"✗ Error downloading {filename}: HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False

    def download_core_files(self):
        """Download core MIMIC-IV files needed for biomarker analysis"""
        if not self.check_access():
            print("\n✗ Cannot proceed without valid access to MIMIC-IV")
            return False

        # List of essential files
        files_to_download = [
            ('hosp', 'patients.csv.gz'),           # Patient demographics
            ('hosp', 'admissions.csv.gz'),         # Admission details
            ('hosp', 'diagnoses_icd.csv.gz'),      # ICD diagnosis codes
            ('hosp', 'd_icd_diagnoses.csv.gz'),    # ICD code dictionary
            ('hosp', 'labevents.csv.gz'),          # Lab test results (LARGE!)
            ('hosp', 'd_labitems.csv.gz'),         # Lab test dictionary
        ]

        print("\n" + "="*80)
        print("Downloading MIMIC-IV Core Files")
        print("="*80)
        print("\nWARNING: labevents.csv.gz is very large (~8GB compressed)")
        print("This may take significant time and bandwidth.\n")

        response = input("Continue with download? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Download cancelled.")
            return False

        success_count = 0
        for subdir, filename in files_to_download:
            if self.download_file(filename, subdir):
                success_count += 1
            else:
                print(f"Failed to download {filename}")

        print(f"\n✓ Successfully downloaded {success_count}/{len(files_to_download)} files")
        return success_count > 0

    def extract_cancer_patients(self):
        """Extract patients with cancer diagnoses from MIMIC-IV data"""
        print("\n" + "="*80)
        print("Extracting Cancer Patients")
        print("="*80)

        # Load diagnosis codes
        print("Loading diagnosis codes...")
        diagnoses_file = self.output_dir / 'diagnoses_icd.csv.gz'
        d_icd_file = self.output_dir / 'd_icd_diagnoses.csv.gz'

        if not diagnoses_file.exists() or not d_icd_file.exists():
            print("✗ Diagnosis files not found. Please download core files first.")
            return None

        diagnoses = pd.read_csv(diagnoses_file)
        icd_dict = pd.read_csv(d_icd_file)

        # Find cancer ICD codes (ICD-9: 140-239, ICD-10: C00-D49)
        cancer_icd9 = icd_dict[
            (icd_dict['icd_version'] == 9) &
            (icd_dict['icd_code'].str.match(r'^(1[4-9]\d|2[0-3]\d)'))
        ]
        cancer_icd10 = icd_dict[
            (icd_dict['icd_version'] == 10) &
            (icd_dict['icd_code'].str.match(r'^[CD]'))
        ]

        cancer_codes = pd.concat([cancer_icd9, cancer_icd10])
        print(f"Found {len(cancer_codes)} cancer-related ICD codes")

        # Get patients with cancer diagnoses
        cancer_patients = diagnoses[
            diagnoses['icd_code'].isin(cancer_codes['icd_code'])
        ]['subject_id'].unique()

        print(f"Found {len(cancer_patients)} patients with cancer diagnoses")

        # Save cancer patient IDs
        output_file = self.output_dir / 'cancer_patient_ids.csv'
        pd.DataFrame({'subject_id': cancer_patients}).to_csv(output_file, index=False)
        print(f"✓ Saved cancer patient IDs to {output_file}")

        return cancer_patients

    def extract_lab_values(self, patient_ids=None):
        """Extract lab values (lactate, glucose, CRP) for specified patients"""
        print("\n" + "="*80)
        print("Extracting Lab Values")
        print("="*80)

        labitems_file = self.output_dir / 'd_labitems.csv.gz'
        labevents_file = self.output_dir / 'labevents.csv.gz'

        if not labitems_file.exists() or not labevents_file.exists():
            print("✗ Lab files not found. Please download core files first.")
            return None

        # Load lab item dictionary
        labitems = pd.read_csv(labitems_file)

        # Find itemids for our biomarkers
        biomarkers = {
            'Lactate': labitems[labitems['label'].str.contains('Lactate', case=False, na=False)],
            'Glucose': labitems[labitems['label'].str.contains('Glucose', case=False, na=False)],
            'CRP': labitems[labitems['label'].str.contains('C Reactive Protein|CRP', case=False, na=False)],
            'LDH': labitems[labitems['label'].str.contains('LDH|Lactate Dehydrogenase', case=False, na=False)],
        }

        print("Found lab item IDs:")
        for marker, items in biomarkers.items():
            if len(items) > 0:
                print(f"  {marker}: {list(items['itemid'].values)}")
            else:
                print(f"  {marker}: NOT FOUND")

        # Get all relevant itemids
        itemids = []
        for items in biomarkers.values():
            itemids.extend(items['itemid'].tolist())

        print(f"\nExtracting lab events for {len(itemids)} lab tests...")
        print("This may take several minutes for large datasets...")

        # Read labevents in chunks (it's huge)
        chunk_size = 100000
        filtered_labs = []

        for i, chunk in enumerate(pd.read_csv(labevents_file, chunksize=chunk_size)):
            # Filter for our biomarkers
            chunk = chunk[chunk['itemid'].isin(itemids)]

            # Filter for our patients if specified
            if patient_ids is not None:
                chunk = chunk[chunk['subject_id'].isin(patient_ids)]

            if len(chunk) > 0:
                filtered_labs.append(chunk)

            if (i + 1) % 10 == 0:
                print(f"  Processed {(i + 1) * chunk_size:,} rows...")

        if filtered_labs:
            df = pd.concat(filtered_labs, ignore_index=True)
            print(f"\n✓ Extracted {len(df):,} lab measurements")

            # Merge with lab item names
            df = df.merge(labitems[['itemid', 'label', 'fluid', 'category']], on='itemid', how='left')

            # Save to file
            output_file = self.output_dir / 'cancer_patient_labs.csv'
            df.to_csv(output_file, index=False)
            print(f"✓ Saved to {output_file}")

            return df
        else:
            print("✗ No matching lab values found")
            return None


def main():
    """Main download function"""
    print("="*80)
    print("MIMIC-IV Cancer Biomarker Data Downloader")
    print("="*80)

    downloader = MIMICDownloader()

    if not downloader.authenticated:
        print("\n" + "="*80)
        print("SETUP REQUIRED")
        print("="*80)
        print("\nTo use this script, you need to:")
        print("1. Get credentialed on PhysioNet (takes ~1 week)")
        print("2. Sign the MIMIC-IV data use agreement")
        print("3. Set your credentials:")
        print("   export PHYSIONET_USERNAME='your_username'")
        print("   export PHYSIONET_PASSWORD='your_password'")
        print("\nFor detailed instructions, visit:")
        print("https://physionet.org/content/mimiciv/")
        return

    # Download core files
    if downloader.download_core_files():
        print("\n✓ Core files downloaded successfully")

        # Extract cancer patients
        cancer_patients = downloader.extract_cancer_patients()

        if cancer_patients is not None:
            # Extract lab values for cancer patients
            downloader.extract_lab_values(cancer_patients)

        print("\n" + "="*80)
        print("✓ MIMIC-IV Download Complete!")
        print("="*80)
    else:
        print("\n✗ Download failed")


if __name__ == '__main__':
    main()
