"""Train an XGBoost regressor on the California Housing dataset."""

import argparse

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on California Housing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    model = XGBRegressor(n_estimators=100, max_depth=6, random_state=args.seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")


if __name__ == "__main__":
    main()
