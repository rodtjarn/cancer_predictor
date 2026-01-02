"""
Generate NHANES Predictions Report

This script creates a comprehensive CSV file with predictions for all REAL NHANES participants.
Run this once, then open the CSV in Excel/Numbers to explore the data.
"""

import numpy as np
import pandas as pd
import joblib

print("="*80)
print("GENERATING NHANES PREDICTIONS REPORT")
print("="*80)

# Load data
print("\nLoading REAL NHANES data...")
df = pd.read_csv('data/nhanes/nhanes_2017_2018_processed.csv')
print(f"✓ Loaded {len(df)} participants (224 cancer, 2,088 controls)")

# Load model
print("\nLoading NHANES Real RF model...")
model = joblib.load('models/nhanes_real_rf_model.pkl')
scaler = joblib.load('models/nhanes_real_scaler.pkl')
print("✓ Model loaded")

# Prepare features
features = ['insulin', 'glucose', 'ldh', 'crp', 'HOMA_IR', 'age', 'gender']
X = df[features].values

# Handle missing values
n_missing = np.isnan(X).sum()
if n_missing > 0:
    print(f"\n⚠️  Filling {n_missing} missing values with column median")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

# Scale and predict
print("\nGenerating predictions...")
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# Add predictions to dataframe
df['predicted_cancer'] = y_pred
df['cancer_probability'] = (y_prob * 100).round(1)  # As percentage
df['prediction_correct'] = (df['cancer'] == df['predicted_cancer'])

# Add risk categories
df['risk_category'] = pd.cut(y_prob,
                              bins=[0, 0.25, 0.5, 0.75, 1.0],
                              labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])

# Add readable labels
df['actual_diagnosis'] = df['cancer'].map({0: 'No Cancer', 1: 'Cancer'})
df['predicted_diagnosis'] = df['predicted_cancer'].map({0: 'No Cancer', 1: 'Cancer'})
df['gender_label'] = df['gender'].map({1: 'Male', 2: 'Female'})

# Reorder columns for better viewing
output_cols = [
    'id',
    'age',
    'gender_label',
    'glucose',
    'insulin',
    'ldh',
    'crp',
    'HOMA_IR',
    'HOMA_IR_quartile',
    'actual_diagnosis',
    'predicted_diagnosis',
    'cancer_probability',
    'risk_category',
    'prediction_correct'
]

df_output = df[output_cols].copy()

# Rename for clarity
df_output.columns = [
    'Patient_ID',
    'Age',
    'Gender',
    'Glucose_mg_dL',
    'Insulin_uU_mL',
    'LDH_U_L',
    'CRP_mg_L',
    'HOMA_IR',
    'HOMA_IR_Quartile',
    'Actual_Diagnosis',
    'Predicted_Diagnosis',
    'Cancer_Probability_%',
    'Risk_Category',
    'Prediction_Correct'
]

# Save to CSV
output_file = 'data/nhanes/nhanes_with_predictions.csv'
df_output.to_csv(output_file, index=False)

print(f"\n✓ Saved predictions to: {output_file}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

accuracy = df['prediction_correct'].mean()
sensitivity = df[df['cancer']==1]['prediction_correct'].mean()
specificity = df[df['cancer']==0]['prediction_correct'].mean()

print(f"\nModel Performance on {len(df)} REAL patients:")
print(f"  Overall Accuracy: {accuracy:.1%}")
print(f"  Sensitivity: {sensitivity:.1%} (correctly identified {df[(df['cancer']==1) & (df['predicted_cancer']==1)].shape[0]}/{df['cancer'].sum()} cancers)")
print(f"  Specificity: {specificity:.1%} (correctly identified {df[(df['cancer']==0) & (df['predicted_cancer']==0)].shape[0]}/{(df['cancer']==0).sum()} healthy)")

print(f"\nRisk Category Breakdown:")
for cat in ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
    count = (df['risk_category'] == cat).sum()
    cancer_count = df[df['risk_category'] == cat]['cancer'].sum()
    print(f"  {cat:15s}: {count:4d} patients ({cancer_count:3d} with cancer = {cancer_count/count*100:.1f}%)")

print(f"\nHigh-Risk Individuals (>75% probability):")
high_risk = df[y_prob > 0.75]
print(f"  Total: {len(high_risk)}")
print(f"  Actually have cancer: {high_risk['cancer'].sum()} ({high_risk['cancer'].mean()*100:.1f}%)")

print(f"\nMisclassifications:")
fp = df[(df['cancer']==0) & (df['predicted_cancer']==1)]
fn = df[(df['cancer']==1) & (df['predicted_cancer']==0)]
print(f"  False Positives: {len(fp)} (healthy wrongly called cancer)")
print(f"  False Negatives: {len(fn)} (cancer patients missed)")

print("\n" + "="*80)
print("HOW TO VIEW THE RESULTS")
print("="*80)
print(f"\n1. Open in Excel/Numbers/Google Sheets:")
print(f"   File -> Open -> {output_file}")
print(f"\n2. Use Python/Pandas:")
print(f"   df = pd.read_csv('{output_file}')")
print(f"   df.head()")
print(f"\n3. Use the interactive viewer:")
print(f"   python view_nhanes_individual_predictions.py")

print("\n" + "="*80)
print("DONE!")
print("="*80)
