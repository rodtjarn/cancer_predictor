"""
Cancer prediction inference
"""

import numpy as np
import pickle
import argparse
import pandas as pd
from pathlib import Path


class CancerPredictor:
    def __init__(self, model_path='models/metabolic_cancer_predictor.pkl'):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.features = data['features']
            self.cost = data.get('cost', 120)
    
    def predict(self, patient_data):
        """
        Predict cancer probability for a patient
        
        Args:
            patient_data: dict with keys:
                - lactate (mM)
                - crp (mg/L)
                - specific_gravity
                - glucose (mM)
                - ldh (U/L)
                - age (years)
                - bmi
        
        Returns:
            dict with:
                - probability: cancer probability (0-1)
                - risk_category: LOW/MODERATE/HIGH/VERY HIGH
                - recommendation: clinical action
        """
        # Convert to feature vector
        X = np.array([[
            patient_data['lactate'],
            patient_data['crp'],
            patient_data['specific_gravity'],
            patient_data['glucose'],
            patient_data['ldh'],
            patient_data['age'],
            patient_data['bmi']
        ]])
        
        # Predict
        prob = self.model.predict_proba(X)[0, 1]
        
        # Categorize risk
        if prob < 0.2:
            category = "LOW"
            recommendation = "Routine screening appropriate"
        elif prob < 0.5:
            category = "MODERATE"
            recommendation = "Consider additional testing (CT/MRI)"
        elif prob < 0.8:
            category = "HIGH"
            recommendation = "Recommend diagnostic workup"
        else:
            category = "VERY HIGH"
            recommendation = "Urgent oncology referral"
        
        return {
            'probability': prob,
            'risk_category': category,
            'recommendation': recommendation
        }


def predict_from_cli(args):
    """Command line prediction"""
    predictor = CancerPredictor(args.model)
    
    patient_data = {
        'lactate': args.lactate,
        'crp': args.crp,
        'specific_gravity': args.sg,
        'glucose': args.glucose,
        'ldh': args.ldh,
        'age': args.age,
        'bmi': args.bmi
    }
    
    result = predictor.predict(patient_data)
    
    print("\n" + "="*50)
    print("CANCER RISK PREDICTION")
    print("="*50)
    print(f"\nBiomarkers:")
    print(f"  Lactate: {args.lactate} mM")
    print(f"  CRP: {args.crp} mg/L")
    print(f"  LDH: {args.ldh} U/L")
    print(f"  Specific Gravity: {args.sg}")
    print(f"  Glucose: {args.glucose} mM")
    print(f"  Age: {args.age} years")
    print(f"  BMI: {args.bmi}")
    print(f"\nResults:")
    print(f"  Cancer Probability: {result['probability']:.1%}")
    print(f"  Risk Category: {result['risk_category']}")
    print(f"  Recommendation: {result['recommendation']}")
    print("="*50)


def predict_from_csv(args):
    """Batch prediction from CSV"""
    predictor = CancerPredictor(args.model)
    
    # Load CSV
    df = pd.read_csv(args.input)
    
    # Predict for each row
    results = []
    for _, row in df.iterrows():
        patient_data = {
            'lactate': row['lactate'],
            'crp': row['crp'],
            'specific_gravity': row['specific_gravity'],
            'glucose': row['glucose'],
            'ldh': row['ldh'],
            'age': row['age'],
            'bmi': row['bmi']
        }
        
        result = predictor.predict(patient_data)
        results.append(result)
    
    # Add results to dataframe
    df['cancer_probability'] = [r['probability'] for r in results]
    df['risk_category'] = [r['risk_category'] for r in results]
    df['recommendation'] = [r['recommendation'] for r in results]
    
    # Save
    df.to_csv(args.output, index=False)
    print(f"✓ Predictions saved to {args.output}")
    print(f"  Processed {len(df)} patients")


def main():
    parser = argparse.ArgumentParser(description='Cancer risk prediction')
    parser.add_argument('--model', type=str, 
                       default='models/metabolic_cancer_predictor.pkl',
                       help='Path to trained model')
    
    # Single prediction
    parser.add_argument('--lactate', type=float, help='Lactate (mM)')
    parser.add_argument('--crp', type=float, help='CRP (mg/L)')
    parser.add_argument('--sg', type=float, help='Specific Gravity')
    parser.add_argument('--glucose', type=float, help='Glucose (mM)')
    parser.add_argument('--ldh', type=float, help='LDH (U/L)')
    parser.add_argument('--age', type=int, help='Age (years)')
    parser.add_argument('--bmi', type=float, help='BMI')
    
    # Batch prediction
    parser.add_argument('--input', type=str, help='Input CSV file')
    parser.add_argument('--output', type=str, help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.input:
        # Batch mode
        predict_from_csv(args)
    else:
        # Single prediction mode
        if not all([args.lactate, args.crp, args.sg, args.glucose, 
                   args.ldh, args.age, args.bmi]):
            parser.error("All biomarker values required for single prediction")
        predict_from_cli(args)


if __name__ == '__main__':
    main()
