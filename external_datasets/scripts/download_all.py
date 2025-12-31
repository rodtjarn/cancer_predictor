"""
Master script to download all external cancer datasets

This script orchestrates downloads from multiple public cancer databases
containing biomarker and clinical data.

Usage:
    python download_all.py --all
    python download_all.py --tcga
    python download_all.py --mimic --tcga
    python download_all.py --list
"""

import argparse
import sys
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "CANCER BIOMARKER DATA DOWNLOADER")
    print("="*80)
    print("\nThis tool downloads cancer patient data with metabolic biomarkers from:")
    print("  • TCGA (The Cancer Genome Atlas) - Automated")
    print("  • MIMIC-IV (PhysioNet) - Requires credentials")
    print("  • PLCO Cancer Screening Trial - Manual access")
    print("  • UK Biobank - Manual access")
    print("\n" + "="*80 + "\n")


def check_setup():
    """Check if required directories and dependencies exist"""
    print("Checking setup...")

    # Check directories
    required_dirs = [
        Path('../tcga'),
        Path('../mimic'),
        Path('../plco'),
        Path('../ukbiobank')
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {dir_path}")

    # Check Python dependencies
    try:
        import requests
        import pandas
        print("  ✓ Required Python packages installed")
        return True
    except ImportError as e:
        print(f"  ✗ Missing required package: {e}")
        print("\n  Install requirements with:")
        print("  pip install -r requirements.txt")
        return False


def list_datasets():
    """List all available datasets with status"""
    print("\nAvailable Datasets:\n")

    datasets = [
        {
            'name': 'TCGA',
            'full_name': 'The Cancer Genome Atlas',
            'access': 'Automated',
            'script': 'download_tcga.py',
            'size': '~500MB (clinical data)',
            'patients': '~11,000',
            'biomarkers': 'Clinical data, demographics (limited lab values)',
            'time': '30-60 minutes'
        },
        {
            'name': 'MIMIC-IV',
            'full_name': 'Medical Information Mart for Intensive Care',
            'access': 'Requires PhysioNet credentials',
            'script': 'download_mimic.py',
            'size': '~8GB+ (with lab events)',
            'patients': '~365,000 (filter for cancer)',
            'biomarkers': 'Lactate, Glucose, possibly CRP, LDH',
            'time': '1-3 hours (depending on bandwidth)'
        },
        {
            'name': 'PLCO',
            'full_name': 'Prostate, Lung, Colorectal, Ovarian Cancer Screening',
            'access': 'Manual data request',
            'script': 'PLCO_INSTRUCTIONS.md',
            'size': 'Varies by request',
            'patients': '~155,000',
            'biomarkers': 'Varies by dataset (check data dictionary)',
            'time': '1-2 weeks approval + download'
        },
        {
            'name': 'UK Biobank',
            'full_name': 'UK Biobank',
            'access': 'Formal application required',
            'script': 'UKBIOBANK_INSTRUCTIONS.md',
            'size': 'Access via cloud platform',
            'patients': '~500,000',
            'biomarkers': '250+ metabolites (NMR), glucose, possibly lactate',
            'time': '4-8 weeks approval'
        }
    ]

    for ds in datasets:
        print(f"{'='*80}")
        print(f"Dataset: {ds['name']} - {ds['full_name']}")
        print(f"{'='*80}")
        print(f"  Access:      {ds['access']}")
        print(f"  Script:      {ds['script']}")
        print(f"  Size:        {ds['size']}")
        print(f"  Patients:    {ds['patients']}")
        print(f"  Biomarkers:  {ds['biomarkers']}")
        print(f"  Time:        {ds['time']}")
        print()


def download_tcga():
    """Download TCGA data"""
    print("\n" + "="*80)
    print("Downloading TCGA Data")
    print("="*80)

    try:
        from download_tcga import main as tcga_main
        tcga_main()
        return True
    except Exception as e:
        print(f"✗ Error downloading TCGA: {e}")
        return False


def download_mimic():
    """Download MIMIC-IV data"""
    print("\n" + "="*80)
    print("Downloading MIMIC-IV Data")
    print("="*80)

    try:
        from download_mimic import main as mimic_main
        mimic_main()
        return True
    except Exception as e:
        print(f"✗ Error downloading MIMIC: {e}")
        return False


def show_manual_instructions(dataset):
    """Show instructions for manual access datasets"""
    if dataset == 'plco':
        print("\n" + "="*80)
        print("PLCO Data Access Instructions")
        print("="*80)
        print("\nPLCO requires manual data access request.")
        print("Please see: PLCO_INSTRUCTIONS.md for detailed steps.")
        print("\nQuick summary:")
        print("  1. Create account at https://cdas.cancer.gov/")
        print("  2. Review data dictionaries")
        print("  3. Submit data request")
        print("  4. Wait for approval (1-2 weeks)")
        print("  5. Download approved data")
        print("\nFor full instructions:")
        print("  cat PLCO_INSTRUCTIONS.md")

    elif dataset == 'ukbiobank':
        print("\n" + "="*80)
        print("UK Biobank Data Access Instructions")
        print("="*80)
        print("\nUK Biobank requires formal application.")
        print("Please see: UKBIOBANK_INSTRUCTIONS.md for detailed steps.")
        print("\nQuick summary:")
        print("  1. Register at https://www.ukbiobank.ac.uk/")
        print("  2. Prepare research proposal")
        print("  3. Submit application")
        print("  4. Wait for approval (2-4 weeks)")
        print("  5. Pay access fee (£2,500 academic)")
        print("  6. Access data via cloud platform")
        print("\nFor full instructions:")
        print("  cat UKBIOBANK_INSTRUCTIONS.md")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Download cancer biomarker datasets from multiple sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python download_all.py --list

  # Download TCGA data only
  python download_all.py --tcga

  # Download TCGA and MIMIC-IV
  python download_all.py --tcga --mimic

  # Attempt to download all automated datasets
  python download_all.py --all

  # Show instructions for manual datasets
  python download_all.py --plco
  python download_all.py --ukbiobank
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Download all automated datasets (TCGA, MIMIC)')
    parser.add_argument('--tcga', action='store_true',
                       help='Download TCGA data')
    parser.add_argument('--mimic', action='store_true',
                       help='Download MIMIC-IV data')
    parser.add_argument('--plco', action='store_true',
                       help='Show PLCO access instructions')
    parser.add_argument('--ukbiobank', action='store_true',
                       help='Show UK Biobank access instructions')
    parser.add_argument('--list', action='store_true',
                       help='List all available datasets')

    args = parser.parse_args()

    # Print banner
    print_banner()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\nStart with: python download_all.py --list")
        return

    # List datasets
    if args.list:
        list_datasets()
        return

    # Check setup
    if not check_setup():
        print("\n✗ Setup check failed. Please install requirements first.")
        return

    # Track successes
    success_count = 0
    total_attempted = 0

    # Download automated datasets
    if args.all or args.tcga:
        total_attempted += 1
        if download_tcga():
            success_count += 1

    if args.all or args.mimic:
        total_attempted += 1
        if download_mimic():
            success_count += 1

    # Show manual instructions
    if args.plco:
        show_manual_instructions('plco')

    if args.ukbiobank:
        show_manual_instructions('ukbiobank')

    # Summary
    if total_attempted > 0:
        print("\n" + "="*80)
        print("Download Summary")
        print("="*80)
        print(f"  Attempted: {total_attempted}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {total_attempted - success_count}")

        if success_count == total_attempted:
            print("\n✓ All downloads completed successfully!")
        elif success_count > 0:
            print("\n⚠ Some downloads completed, but there were errors")
        else:
            print("\n✗ All downloads failed")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("  1. Check downloaded data in external_datasets/*/")
    print("  2. Review data quality and completeness")
    print("  3. Process and merge datasets as needed")
    print("  4. Train models on real patient data")
    print("\nFor manual datasets (PLCO, UK Biobank):")
    print("  - Follow instructions in *_INSTRUCTIONS.md files")
    print("  - Allow 1-8 weeks for access approval")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
