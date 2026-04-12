# XGBoost Regression Example

Minimal XGBoost regression on the California Housing dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

| Script | Description |
|---|---|
| `train.py` | Train/evaluate with train-test split. Prints RMSE and R2. |
| `train_with_cv.py` | Cross-validation via `xgb.cv()`. Prints best RMSE. |
| `feature_importance.py` | Train and print top 5 features by importance. |

```bash
python train.py                   # default seed=42
python train_with_cv.py --nfold 10
python feature_importance.py --seed 123
```
