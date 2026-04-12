"""Train an XGBoost regressor with built-in cross-validation."""

import argparse

import xgboost as xgb
from sklearn.datasets import fetch_california_housing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost with cross-validation on California Housing data."
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="Maximum number of boosting rounds."
    )
    parser.add_argument(
        "--max-depth", type=int, default=6, help="Maximum tree depth."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Boosting learning rate."
    )
    parser.add_argument(
        "--nfold", type=int, default=5, help="Number of CV folds."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    data = fetch_california_housing()
    dtrain = xgb.DMatrix(data.data, label=data.target, feature_names=list(data.feature_names))

    params = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "objective": "reg:squarederror",
        "seed": args.seed,
    }

    # Cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        nfold=args.nfold,
        seed=args.seed,
        metrics=["rmse", "mae"],
        early_stopping_rounds=20,
        verbose_eval=True,
    )

    best_idx = cv_results["test-rmse-mean"].idxmin()
    best_rmse = cv_results.loc[best_idx, "test-rmse-mean"]
    best_mae = cv_results.loc[best_idx, "test-mae-mean"]

    print(f"\nBest iteration: {best_idx}")
    print(f"  RMSE: {best_rmse:.4f}")
    print(f"  MAE:  {best_mae:.4f}")


if __name__ == "__main__":
    main()
