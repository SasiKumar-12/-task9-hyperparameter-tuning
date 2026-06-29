import numpy as np
import json
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── 1. REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 2. LOAD DATA + ADD NOISE (makes tuning matter) ──────────────
data = load_breast_cancer()
X, y = data.data, data.target

# Add random noise columns to make it harder
noise = np.random.normal(0, 5, size=(X.shape[0], 20))
X = np.hstack([X, noise])

print(f"Dataset shape with noise: {X.shape}")

# ── 3. TRAIN / VAL / TEST SPLIT ─────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=SEED)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ── 4. SCALE FEATURES ───────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. BASELINE (default SVM) ───────────────────────────────────
baseline = SVC(random_state=SEED)
baseline.fit(X_train_sc, y_train)
baseline_score = f1_score(y_test, baseline.predict(X_test_sc))
print(f"\n✅ Baseline Test F1: {baseline_score:.4f}")

# ── 6. SEARCH SPACE ─────────────────────────────────────────────
param_dist = {
    'C':      [0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma':  ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'degree': [2, 3, 4],
}

# ── 7. RANDOMIZED SEARCH ────────────────────────────────────────
print("\n🔍 Running hyperparameter search...")
search = RandomizedSearchCV(
    estimator=SVC(random_state=SEED),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=SEED,
    verbose=1
)

search.fit(X_train_sc, y_train)

print(f"\n🏆 Best CV F1   : {search.best_score_:.4f}")
print(f"Best Params     : {search.best_params_}")

# ── 8. CONFIRM ON TEST SET ───────────────────────────────────────
best_model = search.best_estimator_
tuned_score = f1_score(y_test, best_model.predict(X_test_sc))

print(f"\n📊 Results Summary")
print(f"Baseline Test F1 : {baseline_score:.4f}")
print(f"Tuned    Test F1 : {tuned_score:.4f}")
print(f"Gain             : +{tuned_score - baseline_score:.4f}")
print("\n" + classification_report(y_test, best_model.predict(X_test_sc)))

# ── 9. SAVE BEST CONFIG ──────────────────────────────────────────
best_config = {
    "model": "SVC",
    "best_params": search.best_params_,
    "cv_best_f1": round(search.best_score_, 4),
    "baseline_test_f1": round(baseline_score, 4),
    "tuned_test_f1": round(tuned_score, 4),
    "gain": round(tuned_score - baseline_score, 4)
}

with open("results/best_config.json", "w") as f:
    json.dump(best_config, f, indent=2)

print("\n💾 Best config saved to results/best_config.json")