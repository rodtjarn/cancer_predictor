# Quick Start Guide

## Installation (30 seconds)

```bash
# Clone repository
git clone https://github.com/yourusername/cancer-predictor.git
cd cancer-predictor

# Install dependencies
pip install -r requirements.txt
```

## Make a Prediction (10 seconds)

```bash
python src/predict.py \
  --lactate 3.5 \
  --crp 28.0 \
  --ldh 410 \
  --sg 1.024 \
  --glucose 4.6 \
  --age 65 \
  --bmi 24
```

Output:
```
Cancer Probability: 91.3%
Risk Category: VERY HIGH  
Recommendation: Urgent oncology referral
```

## Python API (3 lines)

```python
from src.predict import CancerPredictor

predictor = CancerPredictor()

result = predictor.predict({
    'lactate': 3.5,
    'crp': 28.0,
    'specific_gravity': 1.024,
    'glucose': 4.6,
    'ldh': 410,
    'age': 65,
    'bmi': 24
})

print(f"Risk: {result['probability']:.1%}")
```

## Train Your Own Model (1 minute)

```bash
# Generate training data
python src/generate_data.py --samples 5000

# Train model
python src/train_model.py --data data/training_data.npz

# Done!
```

## Batch Processing

```bash
# Create CSV with patient data
echo "lactate,crp,specific_gravity,glucose,ldh,age,bmi" > patients.csv
echo "3.5,28.0,1.024,4.6,410,65,24" >> patients.csv
echo "1.1,2.5,1.018,5.1,175,45,23" >> patients.csv

# Run batch prediction
python src/predict.py --input patients.csv --output results.csv

# View results
cat results.csv
```

## What's Next?

- Read full [README.md](README.md) for details
- Check [docs/fasting_guide.md](docs/fasting_guide.md) for patient instructions
- See [docs/clinical_protocol.md](docs/clinical_protocol.md) for clinical use

## Need Help?

- GitHub Issues: Report bugs or ask questions
- Email: your.email@example.com
