"""
Quick evaluation script to verify model performance
"""

import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

# Load model
print("Loading model...")
with open('models/metabolic_cancer_predictor.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    features = data['features']

print(f"Features: {features}")

# Load test data
print("\nLoading test data...")
test_data = np.load('data/test_data.npz', allow_pickle=True)
X_test = test_data['X']
y_test = test_data['y']

print(f"Test samples: {len(y_test)}")
print(f"  Cancer cases: {sum(y_test)}")
print(f"  Healthy cases: {len(y_test) - sum(y_test)}")

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate sensitivity and specificity
sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
specificity = tn / (tn + fp)  # True Negative Rate

# Print results
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"\n📊 Overall Metrics:")
print(f"  Accuracy:    {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"  AUC-ROC:     {auc_roc:.3f}")
print(f"\n🎯 Cancer Detection Performance:")
print(f"  Sensitivity: {sensitivity:.3f} ({sensitivity*100:.1f}%)")
print(f"  Specificity: {specificity:.3f} ({specificity*100:.1f}%)")
print(f"\n📋 Confusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")
print(f"\n📈 Expected Results (from README):")
print(f"  Accuracy:    0.988 (98.8%)")
print(f"  Sensitivity: 0.986 (98.6%)")
print(f"  Specificity: 0.990 (99.0%)")
print(f"  AUC-ROC:     0.999")
print("="*60)

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Cancer']))
