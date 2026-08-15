"""
ML Assignment 2 - Model Training Script
Dataset: Breast Cancer Wisconsin (Diagnostic) - built into scikit-learn
569 instances, 30 features, binary classification (malignant / benign)

Trains 5 classifiers, evaluates each on a held-out test split, saves:
  - trained models -> model/*.joblib
  - test data (features + true label) -> test_data.csv (used by the Streamlit app)
  - metrics summary -> metrics_summary.csv (used to fill the README table)
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # project-folder/

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class balance: {y.value_counts().to_dict()}")

# ---------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features (helps Logistic Regression, kNN especially)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))
joblib.dump(feature_names, os.path.join(HERE, "feature_names.joblib"))

# ---------------------------------------------------------------
# 3. Define models
#    (LogReg, kNN, and to a lesser extent NB benefit from scaled input;
#     tree-based models don't need scaling but scaled input doesn't hurt them)
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # save model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    joblib.dump(model, os.path.join(HERE, fname))

# ---------------------------------------------------------------
# 4. Save metrics summary (used to fill README table)
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(ROOT, "metrics_summary.csv"), index=False)
print("\nSaved metrics_summary.csv")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------
# 5. Save test data as CSV (features + true label) for the Streamlit app
#    NOTE: unscaled features, since the app re-applies the saved scaler
# ---------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"\nSaved test_data.csv with {len(test_df)} rows")
