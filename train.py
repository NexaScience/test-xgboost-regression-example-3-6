"""Train an XGBoost regressor on the California Housing dataset."""

import argparse

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost regressor on California Housing data."
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100, help="Number of boosting rounds."
    )
    parser.add_argument(
        "--max-depth", type=int, default=6, help="Maximum tree depth."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Boosting learning rate."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data for testing."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # Train
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # Save model
    model_path = "model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
