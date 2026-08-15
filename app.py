"""
ML Assignment 2 - Streamlit App
Breast Cancer Wisconsin (Diagnostic) - Binary Classification

Features:
 - Upload test data CSV
 - Select which trained model to evaluate
 - View evaluation metrics
 - View confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname))
              for name, fname in MODEL_FILES.items()}
    return scaler, feature_names, models


st.title("🔬 Breast Cancer Classification — Model Comparison")
st.write(
    "Upload the test data CSV (features + `target` column), pick a model, "
    "and see how it performs."
)

scaler, feature_names, models = load_artifacts()

# --- 1. Dataset upload ---
st.subheader("1. Upload test data (CSV)")
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded yet — using the bundled test_data.csv as a default.")
    df = pd.read_csv("test_data.csv")

st.dataframe(df.head(), use_container_width=True)

if "target" not in df.columns:
    st.error("Uploaded CSV must contain a 'target' column with the true labels.")
    st.stop()

X = df[feature_names]
y_true = df["target"]
X_scaled = scaler.transform(X)

# --- 2. Model selection ---
st.subheader("2. Select a model")
model_name = st.selectbox("Choose a classifier", list(models.keys()))
model = models[model_name]

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# --- 3. Metrics ---
st.subheader("3. Evaluation metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
col2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

# --- 4. Confusion matrix + classification report ---
st.subheader("4. Confusion matrix & classification report")
c1, c2 = st.columns(2)

with c1:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    st.pyplot(fig)

with c2:
    report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"])
    st.text("Classification Report")
    st.code(report)

st.caption(
    "Dataset: Breast Cancer Wisconsin (Diagnostic), 569 instances, 30 features. "
    "Models trained on an 80/20 stratified split; this app evaluates on the held-out test split."
)
