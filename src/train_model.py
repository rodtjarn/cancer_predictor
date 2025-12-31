"""
Train cancer prediction model
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(data_path):
    """Load training data from .npz file"""
    data = np.load(data_path, allow_pickle=True)
    return data['X'], data['y'], data['feature_names']


def train_model(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Train Random Forest model"""
    print(f"Training Random Forest with {n_estimators} trees...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    print(f"Training accuracy: {train_acc:.3f}")
    
    return model


def save_model(model, feature_names, output_path):
    """Save trained model to pickle file"""
    model_data = {
        'model': model,
        'features': feature_names,
        'cost': 120  # Total test cost
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train cancer prediction model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (.npz file)')
    parser.add_argument('--output', type=str, default='models/model.pkl',
                       help='Output path for trained model')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in random forest')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth of trees')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}...")
    X_train, y_train, feature_names = load_data(args.data)
    print(f"Loaded {len(y_train)} training samples")
    
    # Train model
    model = train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed
    )
    
    # Show feature importance
    print("\nFeature Importance:")
    for name, importance in sorted(zip(feature_names, model.feature_importances_),
                                   key=lambda x: x[1], reverse=True):
        print(f"  {name:<20}: {importance:.3f}")
    
    # Save model
    save_model(model, feature_names, args.output)


if __name__ == '__main__':
    main()
