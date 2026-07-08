import subprocess
import sys
import time

MODELS = [
    ("model_01_logistic_regression.py", "Logistic Regression"),
    ("model_02_random_forest.py", "Random Forest"),
    ("model_03_xgboost.py", "XGBoost"),
    ("model_04_linear_svm.py", "Linear SVM"),
    ("model_07_simple_cnn.py", "Simple CNN"),
    ("model_08_deep_cnn.py", "Deep CNN"),
]

TIMEOUT = 300

results = []
for script, name in MODELS:
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")
    sys.stdout.flush()
    t0 = time.time()
    try:
        subprocess.run([sys.executable, script], timeout=TIMEOUT, check=True)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        results.append((name, elapsed, "SUCCESS"))
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {TIMEOUT}s")
        results.append((name, TIMEOUT, "TIMEOUT"))
    except subprocess.CalledProcessError as e:
        print(f"  FAILED (exit code {e.returncode})")
        results.append((name, time.time() - t0, "FAILED"))
    sys.stdout.flush()

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
for name, elapsed, status in results:
    print(f"    {name:<30s} {elapsed:>6.1f}s  {status}")
print(f"\n  {sum(1 for _, _, s in results if s == 'SUCCESS')}/{len(results)} succeeded")
