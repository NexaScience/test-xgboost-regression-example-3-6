import argparse
import sys
import platform
import time
import traceback

def main():
    parser = argparse.ArgumentParser(description="XGBoost Feature Importance")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("XGBoost Feature Importance - Verbose Mode")
    print("=" * 60)

    print("\n[ENV] Python version:", sys.version)
    print("[ENV] Platform:", platform.platform())

    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.datasets import fetch_california_housing
        print(f"[ENV] XGBoost: {xgb.__version__}")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("[DEVICE] XGBoost using CPU")

    print(f"\n[DATA] Loading California Housing...")
    try:
        data = fetch_california_housing()
        X, y = data.data, data.target
        print(f"[DATA] Samples: {X.shape[0]}, Features: {X.shape[1]}")
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[TRAIN] Training XGBRegressor(n_estimators=100, max_depth=6)...")
    start_time = time.time()
    try:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=args.seed)
        model.fit(X, y)
        print(f"[TRAIN] Completed in {time.time()-start_time:.2f}s")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n[RESULT] Feature Importances (top 5):")
    try:
        importances = model.feature_importances_
        feature_imp = sorted(zip(data.feature_names, importances), key=lambda x: -x[1])
        for i, (name, imp) in enumerate(feature_imp[:5]):
            bar = "█" * int(imp * 40)
            print(f"[RESULT]   {i+1}. {name:15s}: {imp:.4f} {bar}")
    except Exception as e:
        print(f"[ERROR] Feature importance failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[DONE] Experiment completed (seed={args.seed})")

if __name__ == "__main__":
    main()
