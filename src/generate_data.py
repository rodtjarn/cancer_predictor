"""
Generate synthetic training data for cancer prediction model

Based on published cancer metabolism research data.
"""

import numpy as np
import argparse
from pathlib import Path


def generate_cancer_data(n_samples=2000, random_seed=42):
    """
    Generate synthetic patient data for cancer detection model.
    
    Distribution based on real cancer research:
    - Warburg (1923): Aerobic glycolysis in cancer
    - Zu & Guppy (2004): Cancer metabolism measurements
    - Hirschhaeuser et al. (2011): Lactate in cancer
    
    Args:
        n_samples: Total number of patients to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Feature matrix (n_samples, 7)
        y: Labels (0=healthy, 1=cancer)
        feature_names: List of feature names
    """
    np.random.seed(random_seed)
    
    data = []
    labels = []
    
    # Healthy controls (40%)
    n_healthy = int(n_samples * 0.4)
    for _ in range(n_healthy):
        sample = [
            np.random.uniform(0.5, 2.0),      # Lactate (mM)
            np.random.uniform(0.5, 5.0),      # CRP (mg/L)
            np.random.uniform(1.010, 1.025),  # Specific Gravity
            np.random.uniform(4.0, 6.0),      # Glucose (mM)
            np.random.uniform(100, 250),      # LDH (U/L)
            np.random.randint(30, 80),        # Age
            np.random.uniform(18, 30),        # BMI
        ]
        data.append(sample)
        labels.append(0)
    
    # Early cancer (20%)
    n_early = int(n_samples * 0.2)
    for _ in range(n_early):
        sample = [
            np.random.uniform(2.0, 4.0),      # Lactate elevated
            np.random.uniform(5.0, 40.0),     # CRP elevated
            np.random.uniform(1.015, 1.032),  # SG elevated
            np.random.uniform(3.5, 6.5),      # Glucose variable
            np.random.uniform(250, 500),      # LDH elevated
            np.random.randint(40, 85),        # Age higher
            np.random.uniform(18, 35),        # BMI variable
        ]
        data.append(sample)
        labels.append(1)
    
    # Advanced cancer (15%)
    n_advanced = int(n_samples * 0.15)
    for _ in range(n_advanced):
        sample = [
            np.random.uniform(4.0, 10.0),     # Lactate very high
            np.random.uniform(30.0, 200.0),   # CRP very high
            np.random.uniform(1.020, 1.040),  # SG high (dehydration)
            np.random.uniform(3.0, 7.0),      # Glucose variable
            np.random.uniform(500, 2000),     # LDH very high
            np.random.randint(50, 90),        # Age higher
            np.random.uniform(16, 35),        # BMI lower (cachexia)
        ]
        data.append(sample)
        labels.append(1)
    
    # Diabetic controls (15%)
    n_diabetic = int(n_samples * 0.15)
    for _ in range(n_diabetic):
        sample = [
            np.random.uniform(1.0, 3.0),      # Lactate mildly elevated
            np.random.uniform(1.0, 15.0),     # CRP mildly elevated
            np.random.uniform(1.015, 1.035),  # SG high (glucosuria)
            np.random.uniform(7.0, 15.0),     # Glucose HIGH
            np.random.uniform(150, 350),      # LDH normal
            np.random.randint(40, 80),        # Age
            np.random.uniform(25, 40),        # BMI higher
        ]
        data.append(sample)
        labels.append(0)
    
    # Inflammatory conditions (10%)
    n_inflam = int(n_samples * 0.1)
    for _ in range(n_inflam):
        sample = [
            np.random.uniform(1.5, 3.5),      # Lactate mildly elevated
            np.random.uniform(10.0, 100.0),   # CRP HIGH (inflammation)
            np.random.uniform(1.012, 1.028),  # SG normal
            np.random.uniform(4.0, 7.0),      # Glucose normal
            np.random.uniform(200, 600),      # LDH elevated
            np.random.randint(25, 75),        # Age
            np.random.uniform(18, 35),        # BMI
        ]
        data.append(sample)
        labels.append(0)
    
    X = np.array(data)
    y = np.array(labels)
    
    feature_names = [
        'Lactate (mM)',
        'CRP (mg/L)',
        'Specific Gravity',
        'Glucose (mM)',
        'LDH (U/L)',
        'Age',
        'BMI'
    ]
    
    return X, y, feature_names


def save_data(X, y, feature_names, output_dir, split='train'):
    """Save data to .npz format"""
    output_path = Path(output_dir) / f'{split}_data.npz'
    np.savez(
        output_path,
        X=X,
        y=y,
        feature_names=feature_names
    )
    print(f"Saved {len(y)} samples to {output_path}")
    print(f"  Cancer: {np.sum(y)} ({100*np.sum(y)/len(y):.1f}%)")
    print(f"  Healthy: {len(y) - np.sum(y)} ({100*(len(y)-np.sum(y))/len(y):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate cancer prediction training data')
    parser.add_argument('--samples', type=int, default=2000,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--test-split', type=float, default=0.3,
                       help='Fraction of data for test set')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Generate full dataset
    print(f"Generating {args.samples} samples with seed {args.seed}...")
    X, y, feature_names = generate_cancer_data(
        n_samples=args.samples,
        random_seed=args.seed
    )
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed, stratify=y
    )
    
    # Save training data
    print("\nSaving training data...")
    save_data(X_train, y_train, feature_names, args.output, 'training')
    
    # Save test data
    print("\nSaving test data...")
    save_data(X_test, y_test, feature_names, args.output, 'test')
    
    print("\n✓ Data generation complete!")


if __name__ == '__main__':
    main()
