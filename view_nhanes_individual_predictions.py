"""
View Individual NHANES Predictions

This script loads the REAL NHANES data and shows predictions for each individual.
You can filter by cancer status, high-risk individuals, or specific patient IDs.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)


def load_data_and_model():
    """Load real NHANES data and the trained model"""

    print("="*80)
    print("LOADING REAL NHANES DATA AND MODEL")
    print("="*80)

    # Load processed NHANES data
    df = pd.read_csv('data/nhanes/nhanes_2017_2018_processed.csv')
    print(f"\n✓ Loaded {len(df)} REAL NHANES participants")
    print(f"  Cancer cases: {df['cancer'].sum()}")
    print(f"  Controls: {(df['cancer']==0).sum()}")

    # Load model
    model = joblib.load('models/nhanes_real_rf_model.pkl')
    scaler = joblib.load('models/nhanes_real_scaler.pkl')
    print(f"\n✓ Loaded NHANES Real RF model (trained on REAL data)")

    return df, model, scaler


def make_predictions(df, model, scaler):
    """Generate predictions for all individuals"""

    print("\nGenerating predictions for all individuals...")

    # Prepare features in correct order
    # ['insulin', 'glucose', 'ldh', 'crp', 'HOMA_IR', 'age', 'gender']
    features = ['insulin', 'glucose', 'ldh', 'crp', 'HOMA_IR', 'age', 'gender']

    X = df[features].values

    # Handle any missing values (should be minimal)
    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        print(f"⚠️  Warning: {n_missing} missing values detected, filling with column median")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Add to dataframe
    df = df.copy()
    df['predicted_cancer'] = y_pred
    df['cancer_probability'] = y_prob
    df['prediction_correct'] = (df['cancer'] == df['predicted_cancer'])

    # Categorize predictions
    df['risk_category'] = pd.cut(df['cancer_probability'],
                                   bins=[0, 0.25, 0.5, 0.75, 1.0],
                                   labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])

    print(f"✓ Predictions generated for all {len(df)} participants")

    return df


def show_summary_stats(df):
    """Show summary statistics"""

    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)

    accuracy = df['prediction_correct'].mean()
    print(f"\nOverall Accuracy: {accuracy:.1%}")

    # By actual cancer status
    cancer_df = df[df['cancer']==1]
    control_df = df[df['cancer']==0]

    sensitivity = cancer_df['prediction_correct'].mean()
    specificity = control_df['prediction_correct'].mean()

    print(f"Sensitivity (cancer detection): {sensitivity:.1%}")
    print(f"Specificity (healthy correctly ID'd): {specificity:.1%}")

    # Risk categories
    print(f"\nRisk Category Distribution:")
    print(df['risk_category'].value_counts().sort_index())

    # High risk individuals
    high_risk = df[df['cancer_probability'] > 0.75]
    print(f"\nHigh Risk Individuals (>75% probability):")
    print(f"  Total: {len(high_risk)}")
    print(f"  Actually have cancer: {high_risk['cancer'].sum()} ({high_risk['cancer'].mean()*100:.1f}%)")


def view_cancer_patients(df, n=10):
    """View cancer patients and their predictions"""

    print("\n" + "="*80)
    print(f"CANCER PATIENTS - First {n}")
    print("="*80)

    cancer_df = df[df['cancer']==1].head(n)

    display_cols = [
        'id', 'age', 'gender',
        'glucose', 'insulin', 'ldh', 'crp', 'HOMA_IR',
        'cancer_probability', 'predicted_cancer', 'prediction_correct'
    ]

    print(cancer_df[display_cols].to_string(index=False))

    return cancer_df


def view_high_risk_individuals(df, threshold=0.75):
    """View high-risk individuals (probability > threshold)"""

    print("\n" + "="*80)
    print(f"HIGH RISK INDIVIDUALS (Cancer Probability > {threshold:.0%})")
    print("="*80)

    high_risk = df[df['cancer_probability'] > threshold].sort_values('cancer_probability', ascending=False)

    print(f"\nFound {len(high_risk)} high-risk individuals\n")

    display_cols = [
        'id', 'age', 'gender',
        'glucose', 'insulin', 'ldh', 'crp', 'HOMA_IR', 'HOMA_IR_quartile',
        'cancer_probability', 'cancer', 'prediction_correct'
    ]

    print(high_risk[display_cols].head(20).to_string(index=False))

    return high_risk


def view_misclassified(df):
    """View misclassified cases"""

    print("\n" + "="*80)
    print("MISCLASSIFIED CASES")
    print("="*80)

    wrong = df[~df['prediction_correct']].copy()

    # False positives (predicted cancer but no cancer)
    fp = wrong[wrong['cancer']==0]
    print(f"\nFalse Positives: {len(fp)} (healthy predicted as cancer)")
    print("\nTop 10 by probability:")
    display_cols = [
        'id', 'age', 'gender',
        'glucose', 'insulin', 'ldh', 'crp', 'HOMA_IR',
        'cancer_probability', 'cancer'
    ]
    print(fp.sort_values('cancer_probability', ascending=False).head(10)[display_cols].to_string(index=False))

    # False negatives (predicted no cancer but has cancer)
    fn = wrong[wrong['cancer']==1]
    print(f"\nFalse Negatives: {len(fn)} (cancer missed)")
    print("\nAll missed cancers:")
    print(fn.sort_values('cancer_probability')[display_cols].to_string(index=False))

    return fp, fn


def search_by_id(df, patient_id):
    """Search for specific patient by ID"""

    patient = df[df['id'] == patient_id]

    if len(patient) == 0:
        print(f"\n❌ Patient ID {patient_id} not found")
        return None

    print("\n" + "="*80)
    print(f"PATIENT ID: {patient_id}")
    print("="*80)

    p = patient.iloc[0]

    print(f"\nDemographics:")
    print(f"  Age: {p['age']:.0f} years")
    print(f"  Gender: {'Male' if p['gender']==1 else 'Female'}")

    print(f"\nBiomarkers:")
    print(f"  Fasting Glucose: {p['glucose']:.1f} mg/dL")
    print(f"  Fasting Insulin: {p['insulin']:.2f} µU/mL")
    print(f"  LDH: {p['ldh']:.1f} U/L")
    print(f"  CRP: {p['crp']:.2f} mg/L")
    print(f"  HOMA-IR: {p['HOMA_IR']:.2f} ({p['HOMA_IR_quartile']})")

    print(f"\nCancer Status:")
    print(f"  Actual: {'CANCER' if p['cancer']==1 else 'No Cancer'}")
    print(f"  Predicted: {'CANCER' if p['predicted_cancer']==1 else 'No Cancer'}")
    print(f"  Probability: {p['cancer_probability']:.1%}")
    print(f"  Risk Category: {p['risk_category']}")
    print(f"  Prediction: {'✓ CORRECT' if p['prediction_correct'] else '✗ INCORRECT'}")

    return patient


def save_to_csv(df, filename='data/nhanes/nhanes_with_predictions.csv'):
    """Save results with predictions to CSV"""

    df.to_csv(filename, index=False)
    print(f"\n✓ Saved {len(df)} participants with predictions to: {filename}")
    print(f"  You can open this file in Excel, Numbers, or any spreadsheet software")


def main():
    """Main function with menu"""

    # Load data
    df, model, scaler = load_data_and_model()

    # Make predictions
    df = make_predictions(df, model, scaler)

    # Show summary
    show_summary_stats(df)

    # Interactive menu
    while True:
        print("\n" + "="*80)
        print("MENU - What would you like to view?")
        print("="*80)
        print("1. View cancer patients")
        print("2. View high-risk individuals")
        print("3. View misclassified cases")
        print("4. Search by patient ID")
        print("5. Save all results to CSV")
        print("6. Show summary stats again")
        print("7. Exit")

        choice = input("\nEnter choice (1-7): ").strip()

        if choice == '1':
            n = input("How many to show? (default 10): ").strip()
            n = int(n) if n else 10
            view_cancer_patients(df, n)

        elif choice == '2':
            threshold = input("Probability threshold? (default 0.75): ").strip()
            threshold = float(threshold) if threshold else 0.75
            view_high_risk_individuals(df, threshold)

        elif choice == '3':
            view_misclassified(df)

        elif choice == '4':
            patient_id = input("Enter patient ID: ").strip()
            try:
                patient_id = float(patient_id)
                search_by_id(df, patient_id)
            except:
                print("Invalid patient ID")

        elif choice == '5':
            filename = input("Filename (default: data/nhanes/nhanes_with_predictions.csv): ").strip()
            if not filename:
                filename = 'data/nhanes/nhanes_with_predictions.csv'
            save_to_csv(df, filename)

        elif choice == '6':
            show_summary_stats(df)

        elif choice == '7':
            print("\nExiting...")
            break

        else:
            print("Invalid choice")


if __name__ == '__main__':
    main()
