"""
Test cancer prediction model on UCI Breast Cancer Coimbra dataset

This script tests our metabolic cancer prediction model on real patient data
from the UCI repository to see how it performs with partial biomarker coverage.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_model():
    """Load the trained cancer prediction model"""
    with open('models/metabolic_cancer_predictor.pkl', 'rb') as f:
        model_data = pickle.load(f)

    print("Model Information:")
    print("="*80)
    print(f"Model type: {type(model_data['model']).__name__}")
    print(f"Features expected: {model_data['features']}")
    print(f"Number of features: {len(model_data['features'])}")
    print()

    return model_data['model'], model_data['features']


def load_uci_data():
    """Load UCI Breast Cancer Coimbra dataset"""
    df = pd.read_csv('external_datasets/uci_breast_cancer_coimbra.csv')

    print("UCI Dataset Information:")
    print("="*80)
    print(f"Total samples: {len(df)}")
    print(f"Features available: {list(df.columns)}")

    # Convert classification labels: UCI uses 1=Healthy, 2=Cancer
    # Our model uses 0=Healthy, 1=Cancer
    df['label'] = (df['Classification'] == 2).astype(int)

    print(f"\nClass distribution:")
    print(f"  Healthy (label=0): {(df['label']==0).sum()}")
    print(f"  Cancer (label=1): {(df['label']==1).sum()}")
    print()

    return df


def convert_and_prepare_features(uci_df, model_features):
    """
    Convert UCI features to match model requirements

    Model expects: ['Lactate (mM)', 'CRP (mg/L)', 'Specific Gravity',
                    'Glucose (mM)', 'LDH (U/L)', 'Age', 'BMI']

    UCI has: ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin',
              'Adiponectin', 'Resistin', 'MCP.1']
    """

    print("Feature Mapping:")
    print("="*80)

    # Initialize feature matrix
    X = np.zeros((len(uci_df), len(model_features)))

    feature_mapping = {}

    for i, feature in enumerate(model_features):
        if 'Age' in feature:
            # Direct mapping
            X[:, i] = uci_df['Age'].values
            feature_mapping[feature] = 'UCI Age (direct)'
            print(f"✅ {feature:<25s} <- UCI Age (exact match)")

        elif 'BMI' in feature:
            # Direct mapping
            X[:, i] = uci_df['BMI'].values
            feature_mapping[feature] = 'UCI BMI (direct)'
            print(f"✅ {feature:<25s} <- UCI BMI (exact match)")

        elif 'Glucose' in feature:
            # Convert from mg/dL to mM (divide by 18)
            X[:, i] = uci_df['Glucose'].values / 18.0
            feature_mapping[feature] = 'UCI Glucose (converted mg/dL -> mM)'
            print(f"✅ {feature:<25s} <- UCI Glucose (converted mg/dL -> mM)")

        elif 'Lactate' in feature:
            # NOT AVAILABLE - use training data median
            median_val = 2.5  # Approximate from our training data
            X[:, i] = median_val
            feature_mapping[feature] = f'IMPUTED (median={median_val})'
            print(f"❌ {feature:<25s} <- MISSING (imputed with median={median_val})")

        elif 'CRP' in feature:
            # NOT AVAILABLE - use training data median
            median_val = 15.0  # Approximate from our training data
            X[:, i] = median_val
            feature_mapping[feature] = f'IMPUTED (median={median_val})'
            print(f"❌ {feature:<25s} <- MISSING (imputed with median={median_val})")

        elif 'Specific Gravity' in feature:
            # NOT AVAILABLE - use training data median
            median_val = 1.020  # Approximate from our training data
            X[:, i] = median_val
            feature_mapping[feature] = f'IMPUTED (median={median_val})'
            print(f"❌ {feature:<25s} <- MISSING (imputed with median={median_val})")

        elif 'LDH' in feature:
            # NOT AVAILABLE - use training data median
            median_val = 300.0  # Approximate from our training data
            X[:, i] = median_val
            feature_mapping[feature] = f'IMPUTED (median={median_val})'
            print(f"❌ {feature:<25s} <- MISSING (imputed with median={median_val})")

    print()
    return X, feature_mapping


def evaluate_model(model, X, y_true):
    """Evaluate model performance"""

    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("Model Performance on UCI Data:")
    print("="*80)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {tn} (correctly identified healthy)")
    print(f"  False Positives (FP): {fp} (healthy predicted as cancer)")
    print(f"  False Negatives (FN): {fn} (cancer predicted as healthy)")
    print(f"  True Positives (TP):  {tp} (correctly identified cancer)")

    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nPerformance Metrics:")
    print(f"  Sensitivity (Recall):  {sensitivity:.3f} ({sensitivity*100:.1f}% of cancers caught)")
    print(f"  Specificity:           {specificity:.3f} ({specificity*100:.1f}% of healthy correctly identified)")

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  AUC-ROC:               {auc:.3f}")
    except:
        print(f"  AUC-ROC:               Could not calculate")

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Cancer']))

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def visualize_results(y_true, results):
    """Create visualizations of results"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Cancer'],
                yticklabels=['Healthy', 'Cancer'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Probability Distribution
    y_prob = results['y_prob']
    axes[1].hist(y_prob[y_true==0], bins=20, alpha=0.5, label='Healthy', color='green')
    axes[1].hist(y_prob[y_true==1], bins=20, alpha=0.5, label='Cancer', color='red')
    axes[1].set_xlabel('Predicted Cancer Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Predicted Probabilities')
    axes[1].legend()
    axes[1].axvline(0.5, color='black', linestyle='--', label='Decision Threshold')

    plt.tight_layout()
    plt.savefig('external_datasets/uci_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: external_datasets/uci_test_results.png")

    plt.close()


def compare_to_synthetic():
    """Compare UCI performance to synthetic data performance"""

    print("\n" + "="*80)
    print("COMPARISON: UCI Real Data vs Synthetic Training Data")
    print("="*80)

    print("\nSynthetic Data Performance (from training):")
    print("  • Accuracy:    98.8%")
    print("  • Sensitivity: 98.6%")
    print("  • Specificity: 99.0%")
    print("  • AUC-ROC:     0.999")
    print("  • Sample size: 35,000 (training)")
    print("  • Biomarkers:  7/7 complete")

    print("\nUCI Real Data Performance (just calculated):")
    print("  • Accuracy:    [see above]")
    print("  • Sensitivity: [see above]")
    print("  • Specificity: [see above]")
    print("  • Sample size: 116 patients")
    print("  • Biomarkers:  3/7 available (4/7 IMPUTED!)")

    print("\n⚠️  IMPORTANT LIMITATIONS:")
    print("  • 4 out of 7 biomarkers are MISSING and were imputed")
    print("  • Imputed features: Lactate, CRP, LDH, Specific Gravity")
    print("  • These are the KEY Warburg effect biomarkers!")
    print("  • Performance metrics are NOT reliable due to imputation")
    print("  • Only 116 patients (vs 35,000 training samples)")


def main():
    """Main testing function"""

    print("\n" + "="*80)
    print("  TESTING CANCER PREDICTION MODEL ON UCI BREAST CANCER DATA")
    print("="*80)
    print()

    # Load model
    model, model_features = load_model()

    # Load UCI data
    uci_df = load_uci_data()

    # Prepare features
    X, feature_mapping = convert_and_prepare_features(uci_df, model_features)
    y_true = uci_df['label'].values

    # Evaluate
    results = evaluate_model(model, X, y_true)

    # Visualize
    visualize_results(y_true, results)

    # Compare to synthetic
    compare_to_synthetic()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Tested model on 116 real cancer patients")
    print(f"✓ Accuracy: {results['accuracy']*100:.1f}%")
    print(f"✓ Sensitivity: {results['sensitivity']*100:.1f}%")
    print(f"✓ Specificity: {results['specificity']*100:.1f}%")
    print()
    print("⚠️  CRITICAL LIMITATION:")
    print("   This test used IMPUTED values for 4/7 biomarkers!")
    print("   Results do NOT validate the full model.")
    print("   Need MIMIC-IV for complete validation with all biomarkers.")
    print()
    print("✓ Results saved to: external_datasets/uci_test_results.png")
    print("="*80)


if __name__ == '__main__':
    main()
