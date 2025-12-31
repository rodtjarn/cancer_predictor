"""
Download TCGA clinical and biomarker data via GDC API

This script downloads clinical data from The Cancer Genome Atlas (TCGA)
including laboratory test results and biomarkers for cancer patients.

Requirements: requests, pandas
"""

import requests
import json
import pandas as pd
import os
from pathlib import Path
import time


class TCGADownloader:
    """Download clinical data from TCGA via GDC API"""

    def __init__(self, output_dir='../tcga'):
        self.base_url = 'https://api.gdc.cancer.gov'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_projects(self):
        """Get list of all TCGA projects"""
        endpoint = f'{self.base_url}/projects'
        params = {
            'filters': json.dumps({
                'op': 'in',
                'content': {
                    'field': 'program.name',
                    'value': ['TCGA']
                }
            }),
            'size': 100
        }

        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            projects = data['data']['hits']
            print(f"Found {len(projects)} TCGA projects")
            return projects
        else:
            print(f"Error fetching projects: {response.status_code}")
            return []

    def get_clinical_files(self, project_id):
        """Get clinical data files for a specific project"""
        endpoint = f'{self.base_url}/files'

        filters = {
            'op': 'and',
            'content': [
                {
                    'op': 'in',
                    'content': {
                        'field': 'cases.project.project_id',
                        'value': [project_id]
                    }
                },
                {
                    'op': 'in',
                    'content': {
                        'field': 'data_type',
                        'value': ['Clinical Supplement']
                    }
                }
            ]
        }

        params = {
            'filters': json.dumps(filters),
            'fields': 'file_id,file_name,file_size',
            'size': 1000
        }

        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return response.json()['data']['hits']
        return []

    def get_cases_with_clinical_data(self, project_id, size=1000):
        """Get cases (patients) with clinical data for a project"""
        endpoint = f'{self.base_url}/cases'

        filters = {
            'op': 'and',
            'content': [
                {
                    'op': 'in',
                    'content': {
                        'field': 'project.project_id',
                        'value': [project_id]
                    }
                }
            ]
        }

        params = {
            'filters': json.dumps(filters),
            'fields': ','.join([
                'case_id',
                'submitter_id',
                'diagnoses.age_at_diagnosis',
                'diagnoses.primary_diagnosis',
                'diagnoses.tissue_or_organ_of_origin',
                'diagnoses.tumor_stage',
                'demographic.gender',
                'demographic.race',
                'demographic.ethnicity',
                'demographic.vital_status',
                'exposures.bmi',
                'exposures.weight',
                'exposures.height'
            ]),
            'size': size,
            'format': 'JSON'
        }

        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return response.json()['data']['hits']
        else:
            print(f"Error fetching cases: {response.status_code}")
            return []

    def download_project_clinical_data(self, project_id):
        """Download and save clinical data for a project"""
        print(f"\nDownloading clinical data for {project_id}...")

        cases = self.get_cases_with_clinical_data(project_id)
        print(f"  Found {len(cases)} cases")

        if not cases:
            return None

        # Parse cases into structured data
        rows = []
        for case in cases:
            row = {
                'case_id': case.get('case_id'),
                'submitter_id': case.get('submitter_id'),
                'project': project_id
            }

            # Demographic data
            if 'demographic' in case and case['demographic']:
                demo = case['demographic']
                row['gender'] = demo.get('gender')
                row['race'] = demo.get('race')
                row['ethnicity'] = demo.get('ethnicity')
                row['vital_status'] = demo.get('vital_status')

            # Diagnosis data (take first diagnosis)
            if 'diagnoses' in case and case['diagnoses']:
                diag = case['diagnoses'][0]
                row['age_at_diagnosis'] = diag.get('age_at_diagnosis')
                if row['age_at_diagnosis']:
                    row['age_at_diagnosis'] = row['age_at_diagnosis'] / 365.25  # Convert days to years
                row['primary_diagnosis'] = diag.get('primary_diagnosis')
                row['tissue_or_organ'] = diag.get('tissue_or_organ_of_origin')
                row['tumor_stage'] = diag.get('tumor_stage')

            # Exposure data (BMI, weight, height)
            if 'exposures' in case and case['exposures']:
                exp = case['exposures'][0] if isinstance(case['exposures'], list) else case['exposures']
                row['bmi'] = exp.get('bmi')
                row['weight'] = exp.get('weight')
                row['height'] = exp.get('height')

            rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Save to CSV
        output_file = self.output_dir / f'{project_id}_clinical.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved {len(df)} records to {output_file}")

        return df

    def download_all_projects(self, max_projects=None):
        """Download clinical data for all TCGA projects"""
        projects = self.get_projects()

        if max_projects:
            projects = projects[:max_projects]

        all_data = []

        for i, project in enumerate(projects, 1):
            project_id = project['project_id']
            print(f"\n[{i}/{len(projects)}] Processing {project_id}")

            df = self.download_project_clinical_data(project_id)
            if df is not None:
                all_data.append(df)

            # Be nice to the API
            time.sleep(1)

        # Combine all projects
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            output_file = self.output_dir / 'tcga_all_clinical.csv'
            combined_df.to_csv(output_file, index=False)
            print(f"\n✓ Combined data saved to {output_file}")
            print(f"  Total patients: {len(combined_df)}")
            print(f"  Total projects: {len(all_data)}")

            # Print summary statistics
            print("\nSummary by cancer type:")
            print(combined_df['project'].value_counts())

            return combined_df

        return None


def main():
    """Main download function"""
    print("="*80)
    print("TCGA Clinical Data Downloader")
    print("="*80)
    print("\nNOTE: This downloads clinical/demographic data.")
    print("Laboratory test values (LDH, lactate, etc.) may require additional")
    print("data files or may not be available for all patients.\n")

    downloader = TCGADownloader()

    # Download all TCGA projects (or set max_projects for testing)
    # For testing, use max_projects=5
    df = downloader.download_all_projects(max_projects=None)

    if df is not None:
        print("\n✓ Download complete!")
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    else:
        print("\n✗ No data downloaded")


if __name__ == '__main__':
    main()
