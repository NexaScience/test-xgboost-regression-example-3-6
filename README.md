# XGBoost Regression Example

Minimal XGBoost regression on the California Housing dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

### train.py

Train an XGBoost regressor and evaluate on a held-out test set.

```bash
python train.py
python train.py --n-estimators 200 --max-depth 4 --learning-rate 0.05
```

### train_with_cv.py

Train with cross-validation using XGBoost's built-in `xgb.cv()`.

```bash
python train_with_cv.py
python train_with_cv.py --nfold 10 --n-estimators 300
```

### feature_importance.py

Train a model and display feature importance scores.

```bash
python feature_importance.py
python feature_importance.py --n-estimators 200 --max-depth 4
```
