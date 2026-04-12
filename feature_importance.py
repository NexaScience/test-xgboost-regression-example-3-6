"""Train an XGBoost model and display top feature importances."""

import argparse

from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor


def main():
    parser = argparse.ArgumentParser(description="XGBoost feature importance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data = fetch_california_housing()
    model = XGBRegressor(n_estimators=100, max_depth=6, random_state=args.seed)
    model.fit(data.data, data.target)

    pairs = sorted(
        zip(data.feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    print("Top 5 Features")
    print("=" * 30)
    for name, score in pairs[:5]:
        print(f"  {name:>12s}: {score:.4f}")


if __name__ == "__main__":
    main()
