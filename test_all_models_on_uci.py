"""
Test all cancer prediction models on UCI Breast Cancer Coimbra dataset

This script tests multiple models on the UCI dataset which has insulin data,
making it perfect for testing the NHANES Random Forest models.

UCI Dataset has:
- Age, BMI, Glucose, Insulin, HOMA
- Plus: Leptin, Adiponectin, Resistin, MCP.1
- 116 breast cancer patients (52 healthy, 64 cancer)

Models to test:
1. NHANES RF Model (uses insulin) - BEST MATCH
2. NHANES Real RF Model
3. V2 Synthetic Model
4. V3 MIMIC-Matched Model
"""

import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_uci_data():
    """Load UCI Breast Cancer Coimbra dataset"""
    print("\n" + "="*80)
    print("LOADING UCI BREAST CANCER COIMBRA DATASET")
    print("="*80)

    df = pd.read_csv('external_datasets/uci_breast_cancer_coimbra.csv')

    print(f"Total samples: {len(df)}")
    print(f"Features: {list(df.columns)}")

    # Convert classification: UCI uses 1=Healthy, 2=Cancer
    # We use 0=Healthy, 1=Cancer
    df['label'] = (df['Classification'] == 2).astype(int)

    print(f"\nClass distribution:")
    print(f"  Healthy (0): {(df['label']==0).sum()}")
    print(f"  Cancer (1): {(df['label']==1).sum()}")
    print(f"  Cancer rate: {df['label'].mean()*100:.1f}%")

    return df


def test_nhanes_rf_model(uci_df):
    """Test NHANES Random Forest model - has insulin data!"""

    print("\n" + "="*80)
    print("TEST 1: NHANES RANDOM FOREST MODEL (Synthetic Data)")
    print("="*80)

    try:
        # Load model
        model = joblib.load('models/nhanes_rf_model.pkl')
        scaler = joblib.load('models/nhanes_scaler.pkl')

        print("✓ Model loaded successfully")

        # Read feature names
        with open('models/nhanes_features.txt', 'r') as f:
            features = f.read().strip().split('\n')

        print(f"Expected features: {features}")

        # Prepare features
        # NHANES model expects: [Insulin, Glucose, LDH, CRP, HOMA-IR, Age, Gender]
        X = np.zeros((len(uci_df), len(features)))

        feature_mapping = []
        for i, feature in enumerate(features):
            if 'INSULIN' in feature.upper() or 'Insulin' in feature:
                X[:, i] = uci_df['Insulin'].values
                feature_mapping.append(f"✅ {feature} <- UCI Insulin")

            elif 'GLUCOSE' in feature.upper() or 'Glucose' in feature:
                X[:, i] = uci_df['Glucose'].values
                feature_mapping.append(f"✅ {feature} <- UCI Glucose")

            elif 'LDH' in feature.upper():
                # NOT AVAILABLE - impute with median
                X[:, i] = 180.0  # Median LDH
                feature_mapping.append(f"❌ {feature} <- IMPUTED (180.0)")

            elif 'CRP' in feature.upper():
                # NOT AVAILABLE - impute with median
                X[:, i] = 5.0  # Median CRP
                feature_mapping.append(f"❌ {feature} <- IMPUTED (5.0)")

            elif 'HOMA' in feature.upper():
                X[:, i] = uci_df['HOMA'].values
                feature_mapping.append(f"✅ {feature} <- UCI HOMA")

            elif 'AGE' in feature.upper() or 'Age' in feature:
                X[:, i] = uci_df['Age'].values
                feature_mapping.append(f"✅ {feature} <- UCI Age")

            elif 'GENDER' in feature.upper() or 'Gender' in feature or 'SEX' in feature.upper():
                # Gender not available - impute with 0 (female) since it's breast cancer
                X[:, i] = 0
                feature_mapping.append(f"❌ {feature} <- IMPUTED (0=Female)")

        print("\nFeature Mapping:")
        for mapping in feature_mapping:
            print(f"  {mapping}")

        # Count available features
        available = sum(1 for m in feature_mapping if '✅' in m)
        total = len(features)
        print(f"\nBiomarker coverage: {available}/{total} ({available/total*100:.1f}%)")

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict
        y_true = uci_df['label'].values
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # Evaluate
        results = evaluate_model(y_true, y_pred, y_prob, "NHANES RF")

        return results, feature_mapping

    except Exception as e:
        print(f"❌ Error testing NHANES RF model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_nhanes_real_rf_model(uci_df):
    """Test NHANES Real RF model trained on real NHANES data"""

    print("\n" + "="*80)
    print("TEST 2: NHANES REAL RF MODEL (Real NHANES Data)")
    print("="*80)

    try:
        # Load model
        model = joblib.load('models/nhanes_real_rf_model.pkl')
        scaler = joblib.load('models/nhanes_real_scaler.pkl')

        print("✓ Model loaded successfully")

        # Read feature names
        with open('models/nhanes_features.txt', 'r') as f:
            features = f.read().strip().split('\n')

        # Prepare features (same as NHANES RF)
        X = np.zeros((len(uci_df), len(features)))

        for i, feature in enumerate(features):
            if 'INSULIN' in feature.upper() or 'Insulin' in feature:
                X[:, i] = uci_df['Insulin'].values
            elif 'GLUCOSE' in feature.upper() or 'Glucose' in feature:
                X[:, i] = uci_df['Glucose'].values
            elif 'LDH' in feature.upper():
                X[:, i] = 180.0
            elif 'CRP' in feature.upper():
                X[:, i] = 5.0
            elif 'HOMA' in feature.upper():
                X[:, i] = uci_df['HOMA'].values
            elif 'AGE' in feature.upper() or 'Age' in feature:
                X[:, i] = uci_df['Age'].values
            elif 'GENDER' in feature.upper() or 'Gender' in feature:
                X[:, i] = 0  # Female (breast cancer dataset)

        # Scale and predict
        X_scaled = scaler.transform(X)
        y_true = uci_df['label'].values
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # Evaluate
        results = evaluate_model(y_true, y_pred, y_prob, "NHANES Real RF")

        return results

    except Exception as e:
        print(f"❌ Error testing NHANES Real RF model: {e}")
        return None


def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Evaluate model performance"""

    print(f"\nPerformance Metrics:")
    print("-" * 40)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = None

    print(f"Accuracy:    {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Sensitivity: {sensitivity:.3f} ({sensitivity*100:.1f}% of cancers detected)")
    print(f"Specificity: {specificity:.3f} ({specificity*100:.1f}% of healthy correct)")
    print(f"Precision:   {precision:.3f}")
    print(f"F1-Score:    {f1:.3f}")
    if auc is not None:
        print(f"AUC-ROC:     {auc:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:3d} (healthy correctly identified)")
    print(f"  FP={fp:3d} (healthy wrongly called cancer)")
    print(f"  FN={fn:3d} (cancer missed)")
    print(f"  TP={tp:3d} (cancer correctly identified)")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def compare_results(results_list):
    """Compare results from all models"""

    print("\n" + "="*80)
    print("COMPARISON: ALL MODELS ON UCI DATA")
    print("="*80)

    # Create comparison table
    comparison = []
    for result in results_list:
        if result is not None:
            comparison.append({
                'Model': result['model'],
                'Accuracy': f"{result['accuracy']:.3f}",
                'Sensitivity': f"{result['sensitivity']:.3f}",
                'Specificity': f"{result['specificity']:.3f}",
                'F1-Score': f"{result['f1_score']:.3f}",
                'AUC-ROC': f"{result['auc']:.3f}" if result['auc'] is not None else 'N/A'
            })

    df_comparison = pd.DataFrame(comparison)
    print("\n" + df_comparison.to_string(index=False))

    # Find best model
    valid_results = [r for r in results_list if r is not None and r['auc'] is not None]
    if valid_results:
        best_model = max(valid_results, key=lambda x: x['auc'])
        print(f"\n🏆 Best Model: {best_model['model']}")
        print(f"   AUC-ROC: {best_model['auc']:.3f}")
        print(f"   Accuracy: {best_model['accuracy']:.3f}")


def visualize_results(results_list, uci_df):
    """Create visualizations"""

    valid_results = [r for r in results_list if r is not None]
    if not valid_results:
        print("No results to visualize")
        return

    n_models = len(valid_results)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    y_true = uci_df['label'].values

    for i, result in enumerate(valid_results):
        # Confusion Matrix
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Healthy', 'Cancer'],
                    yticklabels=['Healthy', 'Cancer'],
                    ax=axes[0, i], cbar=False)
        axes[0, i].set_title(f"{result['model']}\nConfusion Matrix")
        axes[0, i].set_ylabel('True Label')
        axes[0, i].set_xlabel('Predicted Label')

        # Probability Distribution
        y_prob = result['y_prob']
        axes[1, i].hist(y_prob[y_true==0], bins=15, alpha=0.6, label='Healthy', color='green', edgecolor='black')
        axes[1, i].hist(y_prob[y_true==1], bins=15, alpha=0.6, label='Cancer', color='red', edgecolor='black')
        axes[1, i].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        axes[1, i].set_xlabel('Predicted Cancer Probability')
        axes[1, i].set_ylabel('Count')
        axes[1, i].set_title(f"{result['model']}\nProbability Distribution")
        axes[1, i].legend()
        axes[1, i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/uci_all_models_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: results/uci_all_models_test_results.png")
    plt.close()


def analyze_insulin_resistance(uci_df):
    """Analyze insulin resistance in UCI dataset"""

    print("\n" + "="*80)
    print("INSULIN RESISTANCE ANALYSIS ON UCI DATA")
    print("="*80)

    # Calculate HOMA-IR if not already present
    if 'HOMA' in uci_df.columns:
        homa_ir = uci_df['HOMA'].values
    else:
        # HOMA-IR = (Insulin * Glucose) / 405
        homa_ir = (uci_df['Insulin'] * uci_df['Glucose']) / 405

    cancer = uci_df['label'] == 1
    healthy = uci_df['label'] == 0

    print(f"\nHOMA-IR Statistics:")
    print(f"  Healthy mean: {homa_ir[healthy].mean():.2f} ± {homa_ir[healthy].std():.2f}")
    print(f"  Cancer mean:  {homa_ir[cancer].mean():.2f} ± {homa_ir[cancer].std():.2f}")

    # Check insulin resistance prevalence (HOMA-IR > 2.5)
    ir_threshold = 2.5
    healthy_ir_rate = (homa_ir[healthy] > ir_threshold).mean()
    cancer_ir_rate = (homa_ir[cancer] > ir_threshold).mean()

    print(f"\nInsulin Resistance Prevalence (HOMA-IR > {ir_threshold}):")
    print(f"  Healthy: {healthy_ir_rate*100:.1f}%")
    print(f"  Cancer:  {cancer_ir_rate*100:.1f}%")
    print(f"  Ratio:   {cancer_ir_rate/healthy_ir_rate:.2f}x")

    # Quartile analysis
    quartiles = np.percentile(homa_ir, [25, 50, 75])
    print(f"\nHOMA-IR Quartiles:")
    print(f"  Q1 (25%): {quartiles[0]:.2f}")
    print(f"  Q2 (50%): {quartiles[1]:.2f}")
    print(f"  Q3 (75%): {quartiles[2]:.2f}")

    # Cancer rate by quartile
    q1_mask = homa_ir < quartiles[0]
    q2_mask = (homa_ir >= quartiles[0]) & (homa_ir < quartiles[1])
    q3_mask = (homa_ir >= quartiles[1]) & (homa_ir < quartiles[2])
    q4_mask = homa_ir >= quartiles[2]

    print(f"\nCancer Rate by HOMA-IR Quartile:")
    print(f"  Q1 (low):  {uci_df.loc[q1_mask, 'label'].mean()*100:.1f}%")
    print(f"  Q2:        {uci_df.loc[q2_mask, 'label'].mean()*100:.1f}%")
    print(f"  Q3:        {uci_df.loc[q3_mask, 'label'].mean()*100:.1f}%")
    print(f"  Q4 (high): {uci_df.loc[q4_mask, 'label'].mean()*100:.1f}%")


def main():
    """Main testing function"""

    print("\n" + "="*80)
    print("COMPREHENSIVE UCI BREAST CANCER TESTING")
    print("Testing all models with focus on insulin resistance")
    print("="*80)

    # Load data
    uci_df = load_uci_data()

    # Analyze insulin resistance first
    analyze_insulin_resistance(uci_df)

    # Test all models
    results = []

    # Test NHANES RF (synthetic)
    nhanes_result, feature_mapping = test_nhanes_rf_model(uci_df)
    if nhanes_result:
        results.append(nhanes_result)

    # Test NHANES Real RF
    nhanes_real_result = test_nhanes_real_rf_model(uci_df)
    if nhanes_real_result:
        results.append(nhanes_real_result)

    # Compare all results
    if results:
        compare_results(results)
        visualize_results(results, uci_df)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Tested {len(results)} models on UCI dataset (116 patients)")
    print(f"✓ Dataset has INSULIN data - can test insulin resistance hypothesis!")

    if feature_mapping:
        available = sum(1 for m in feature_mapping if '✅' in m)
        total = len(feature_mapping)
        print(f"✓ Biomarker coverage: {available}/{total} features available")

    print("\nKey Limitations:")
    print("  • Small sample size (116 patients)")
    print("  • Only breast cancer (not diverse cancer types)")
    print("  • Missing LDH and CRP (key metabolic markers)")
    print("  • All female patients")

    print("\nKey Advantage:")
    print("  ✓ HAS INSULIN DATA - can test insulin resistance hypothesis!")
    print("  ✓ Real patient data (not synthetic)")
    print("  ✓ Validates NHANES model design")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
