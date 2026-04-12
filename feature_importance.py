"""Train an XGBoost model and display feature importance."""

import argparse

import pandas as pd
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost and show feature importance for California Housing."
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100, help="Number of boosting rounds."
    )
    parser.add_argument(
        "--max-depth", type=int, default=6, help="Maximum tree depth."
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

    # Train
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
    )
    model.fit(X, y)

    # Feature importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame(
        {"feature": data.feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    print("Feature Importance")
    print("=" * 35)
    for _, row in feat_imp.iterrows():
        print(f"  {row['feature']:>12s}: {row['importance']:.4f}")

    # Save to CSV
    out_path = "feature_importance.csv"
    feat_imp.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
