"""
Train cancer prediction model on V2 synthetic data (realistic stochastics)

This script:
1. Loads the v2 training data (with realistic distributions)
2. Trains Random Forest model (same parameters as original)
3. Evaluates on v2 synthetic test set
4. Tests on real MIMIC-IV data
5. Compares v1 vs v2 performance
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    """Load data from npz file"""
    data = np.load(filepath, allow_pickle=True)
    return data['X'], data['y'], data['feature_names']


def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("Training Random Forest model...")
    print(f"  n_estimators=100, max_depth=10, random_state=42")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Feature importance
    importance = model.feature_importances_

    return model, importance


def evaluate_model(model, X_test, y_test, feature_names, dataset_name="Test"):
    """Evaluate model performance"""
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} data...")
    print(f"{'='*60}")

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Cancer']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"\nSensitivity (Recall): {sensitivity*100:.1f}%")
    print(f"Specificity: {specificity*100:.1f}%")

    return {
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'predictions': y_pred,
        'confusion_matrix': cm
    }


def plot_feature_importance(importance, feature_names, save_path=None):
    """Plot feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))

    indices = np.argsort(importance)[::-1]

    ax.bar(range(len(importance)), importance[indices])
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance (V2 Model - Realistic Stochastics)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")

    plt.close()


def compare_models(v1_results, v2_results):
    """Compare V1 vs V2 model performance"""
    print("\n" + "="*60)
    print("COMPARISON: V1 (Uniform Distributions) vs V2 (Realistic Stochastics)")
    print("="*60)

    print("\nSYNTHETIC TEST SET PERFORMANCE:")
    print(f"  V1 accuracy: {v1_results['synthetic']*100:.2f}%")
    print(f"  V2 accuracy: {v2_results['synthetic']*100:.2f}%")
    print(f"  Difference: {(v2_results['synthetic'] - v1_results['synthetic'])*100:+.2f} pp")

    print("\nREAL MIMIC-IV PERFORMANCE:")
    print(f"  V1 accuracy: {v1_results['mimic']*100:.2f}%")
    print(f"  V2 accuracy: {v2_results['mimic']*100:.2f}%")
    print(f"  Difference: {(v2_results['mimic'] - v1_results['mimic'])*100:+.2f} pp")

    print("\nSIM-TO-REAL GAP:")
    v1_gap = v1_results['synthetic'] - v1_results['mimic']
    v2_gap = v2_results['synthetic'] - v2_results['mimic']
    print(f"  V1 gap: {v1_gap*100:.2f} pp  (synthetic - real)")
    print(f"  V2 gap: {v2_gap*100:.2f} pp  (synthetic - real)")
    print(f"  Gap reduction: {(v1_gap - v2_gap)*100:.2f} pp  {'✅ IMPROVED' if v2_gap < v1_gap else '❌ WORSE'}")

    print("\nINTERPRETATION:")
    if v2_gap < v1_gap:
        print("  ✅ V2 synthetic data better matches real-world distribution")
        print("  ✅ Smaller sim-to-real gap indicates more realistic training data")
    else:
        print("  ⚠️  V2 did not reduce sim-to-real gap")
        print("  ⚠️  May need further improvements to stochastics")


def main():
    print("="*60)
    print("TRAINING V2 MODEL WITH REALISTIC STOCHASTICS")
    print("="*60)

    # Load V2 training data
    print("\nLoading V2 training data...")
    X_train, y_train, feature_names = load_data('data/training_data_v2.npz')
    print(f"  Loaded {len(y_train)} training samples")
    print(f"  Cancer: {np.sum(y_train)} ({100*np.sum(y_train)/len(y_train):.1f}%)")
    print(f"  Features: {', '.join(feature_names)}")

    # Train model
    model, importance = train_model(X_train, y_train)

    # Print feature importance
    print("\nFeature Importance:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {imp*100:5.1f}%")

    # Save model
    model_path = 'models/model_v2_realistic_stochastics.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'feature_importance': importance,
            'version': 'v2_realistic_stochastics'
        }, f)
    print(f"\n✓ Saved model to {model_path}")

    # Plot feature importance
    plot_feature_importance(importance, feature_names,
                           save_path='results/v2_feature_importance.png')

    # Evaluate on V2 synthetic test set
    print("\n" + "="*60)
    print("EVALUATION ON V2 SYNTHETIC TEST SET")
    print("="*60)
    X_test_v2, y_test_v2, _ = load_data('data/test_data_v2.npz')
    results_synthetic = evaluate_model(model, X_test_v2, y_test_v2, feature_names,
                                      dataset_name="V2 Synthetic Test")

    # Test on real MIMIC-IV data
    print("\n" + "="*60)
    print("EVALUATION ON REAL MIMIC-IV DATA")
    print("="*60)

    # Load MIMIC-IV demo data
    try:
        mimic_data = np.load('external_datasets/mimic_iv_demo/mimic_iv_demo_biomarker_data.npz')
        X_mimic = mimic_data['X']
        y_mimic = mimic_data['y']

        # Split MIMIC data (same as original validation)
        from sklearn.model_selection import train_test_split
        _, X_mimic_test, _, y_mimic_test = train_test_split(
            X_mimic, y_mimic, test_size=0.3, random_state=42, stratify=y_mimic
        )

        results_mimic = evaluate_model(model, X_mimic_test, y_mimic_test, feature_names,
                                      dataset_name="MIMIC-IV Real Patients")

    except FileNotFoundError:
        print("❌ MIMIC-IV data not found. Skipping real-world evaluation.")
        results_mimic = None

    # Compare with V1 results
    if results_mimic is not None:
        print("\n" + "="*60)
        print("COMPARING V1 vs V2 PERFORMANCE")
        print("="*60)

        # V1 results (from previous validation)
        v1_results = {
            'synthetic': 0.9921,  # Original synthetic test accuracy
            'mimic': 0.733        # MIMIC-IV validation accuracy
        }

        v2_results = {
            'synthetic': results_synthetic['accuracy'],
            'mimic': results_mimic['accuracy']
        }

        compare_models(v1_results, v2_results)

        # Save comparison results
        comparison = {
            'v1': v1_results,
            'v2': v2_results,
            'v2_synthetic_results': results_synthetic,
            'v2_mimic_results': results_mimic,
            'feature_importance': dict(zip(feature_names, importance))
        }

        with open('results/v1_vs_v2_comparison.pkl', 'wb') as f:
            pickle.dump(comparison, f)
        print("\n✓ Saved comparison results to results/v1_vs_v2_comparison.pkl")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Trained V2 model with realistic stochastics")
    print(f"✓ Synthetic test accuracy: {results_synthetic['accuracy']*100:.2f}%")
    if results_mimic:
        print(f"✓ MIMIC-IV accuracy: {results_mimic['accuracy']*100:.2f}%")
        gap = results_synthetic['accuracy'] - results_mimic['accuracy']
        print(f"✓ Sim-to-real gap: {gap*100:.2f} pp")

        # Hypothesis test
        if gap < 0.26:  # Original gap was 26 pp (99.21% - 73.3%)
            print("\n🎉 SUCCESS: Realistic stochastics REDUCED sim-to-real gap!")
        else:
            print("\n⚠️  Sim-to-real gap still large. May need further improvements.")


if __name__ == '__main__':
    main()
