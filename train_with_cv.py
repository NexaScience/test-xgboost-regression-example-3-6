import argparse
import sys
import platform
import time
import traceback

def main():
    parser = argparse.ArgumentParser(description="XGBoost CV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nfold", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("XGBoost Cross-Validation - Verbose Mode")
    print("=" * 60)

    print("\n[ENV] Python version:", sys.version)
    print("[ENV] Platform:", platform.platform())

    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.datasets import fetch_california_housing
        print(f"[ENV] XGBoost: {xgb.__version__}, NumPy: {np.__version__}")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("[DEVICE] XGBoost using CPU")

    print(f"\n[DATA] Loading California Housing...")
    try:
        data = fetch_california_housing()
        dtrain = xgb.DMatrix(data.data, label=data.target)
        print(f"[DATA] Samples: {data.data.shape[0]}, Features: {data.data.shape[1]}")
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    params = {"max_depth": 6, "objective": "reg:squarederror", "seed": args.seed}
    num_boost_round = 100
    print(f"\n[TRAIN] CV params: {params}")
    print(f"[TRAIN] nfold={args.nfold}, num_boost_round={num_boost_round}")

    start_time = time.time()
    try:
        cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round,
                           nfold=args.nfold, metrics="rmse", seed=args.seed,
                           verbose_eval=10)
        train_time = time.time() - start_time
        print(f"\n[TRAIN] CV completed in {train_time:.2f}s")
        rmse_values = cv_results["test-rmse-mean"]
        best_rmse = min(rmse_values)
        best_round = rmse_values.index(best_rmse) if isinstance(rmse_values, list) else rmse_values.idxmin()
        print(f"[RESULT] Best RMSE: {best_rmse:.4f} at round {best_round}")
    except Exception as e:
        print(f"[ERROR] CV failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[DONE] Experiment completed (seed={args.seed}, nfold={args.nfold})")

if __name__ == "__main__":
    main()
