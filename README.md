# ML Churn Prediction

Binary classification to predict customer churn using XGBoost on the IBM Telco Customer Churn dataset.

## Setup
```bash
pip install -r requirements.txt
```

## Run Training
```bash
python src/train.py --lr 0.05 --n_estimators 300 --max_depth 5 --seed 42
```

## Run Evaluation
```bash
python src/evaluate.py
```

## Run Tests
```bash
pytest tests/ -v
```

## Dataset
IBM Telco Customer Churn — 7,043 samples, 20 features. Source: IBM GitHub (public).

## Results
- Val AUC: ~0.84
- Checkpoint: `checkpoints/xgb_best_v1.json`
- Logs: `logs/train_results.log`