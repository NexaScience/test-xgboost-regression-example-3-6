import argparse
import sys
import platform
import time
import traceback

def main():
    parser = argparse.ArgumentParser(description="XGBoost Regression")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("XGBoost Regression (Holdout) - Verbose Mode")
    print("=" * 60)

    print("\n[ENV] Python version:", sys.version)
    print("[ENV] Platform:", platform.platform())

    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        import sklearn
        print(f"[ENV] XGBoost version: {xgb.__version__}")
        print(f"[ENV] scikit-learn version: {sklearn.__version__}")
        print(f"[ENV] NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # GPU Detection
    print("\n[DEVICE] Checking for GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[DEVICE] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("[DEVICE] CUDA not available")
    except ImportError:
        print("[DEVICE] PyTorch not installed - GPU detection via XGBoost")
    print("[DEVICE] XGBoost using CPU (tree_method=auto)")

    # Data
    print("\n[DATA] Loading California Housing dataset...")
    try:
        data = fetch_california_housing()
        X, y = data.data, data.target
        print(f"[DATA] Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"[DATA] Feature names: {', '.join(data.feature_names)}")
        print(f"[DATA] Target range: [{y.min():.4f}, {y.max():.4f}]")
    except Exception as e:
        print(f"[ERROR] Dataset loading failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Split
    print(f"\n[SPLIT] test_size=0.2, seed={args.seed}")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
        print(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")
    except Exception as e:
        print(f"[ERROR] Split failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Training
    n_estimators = 100
    max_depth = 6
    print(f"\n[TRAIN] XGBRegressor(n_estimators={n_estimators}, max_depth={max_depth}, seed={args.seed})")
    start_time = time.time()
    try:
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=args.seed, verbosity=1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)
        train_time = time.time() - start_time
        print(f"\n[TRAIN] Training completed in {train_time:.2f}s")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluation
    print("\n[EVAL] Evaluating on test set...")
    try:
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        print(f"[EVAL] RMSE: {rmse:.4f}")
        print(f"[EVAL] R2: {r2:.4f}")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[DONE] Experiment completed (seed={args.seed})")

if __name__ == "__main__":
    main()
