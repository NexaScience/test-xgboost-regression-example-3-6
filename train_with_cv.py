"""Train an XGBoost regressor with cross-validation."""

import argparse

import xgboost as xgb
from sklearn.datasets import fetch_california_housing


def main():
    parser = argparse.ArgumentParser(description="XGBoost CV on California Housing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--nfold", type=int, default=5, help="Number of CV folds.")
    args = parser.parse_args()

    data = fetch_california_housing()
    dtrain = xgb.DMatrix(data.data, label=data.target)

    params = {
        "max_depth": 6,
        "objective": "reg:squarederror",
        "seed": args.seed,
    }

    cv_results = xgb.cv(
        params, dtrain,
        num_boost_round=100,
        nfold=args.nfold,
        seed=args.seed,
        metrics="rmse",
    )

    best_rmse = cv_results["test-rmse-mean"].min()
    print(f"Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
