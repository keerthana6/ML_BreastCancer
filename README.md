# ML Assignment 2 — Breast Cancer Classification

## a. Problem Statement

This project builds and compares five classification models to predict whether
a breast tumor is **malignant** or **benign** based on features computed from
digitized images of a fine needle aspirate (FNA) of a breast mass. This is a
binary classification problem with direct clinical relevance — an accurate
model can support early diagnosis and triage.

## b. Dataset Description

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset — a well-known
  public dataset originally from the UCI Machine Learning Repository, and
  bundled directly with scikit-learn (`sklearn.datasets.load_breast_cancer`).
- **Instances**: 569
- **Features**: 30 numeric features (mean, standard error, and "worst"/largest
  value for 10 real-valued properties of each cell nucleus — radius, texture,
  perimeter, area, smoothness, compactness, concavity, concave points,
  symmetry, fractal dimension)
- **Target**: Binary — `0 = malignant`, `1 = benign`
- **Class balance**: 212 malignant, 357 benign
- **Split**: 80% train / 20% test, stratified by class, `random_state=42`
- Features were standardized (zero mean, unit variance) using
  `StandardScaler` fit on the training set before modeling.

## c. GitHub Repository Link

> https://github.com/keerthana6/ML_BreastCancer

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*(All metrics computed on the same held-out 20% test split, `random_state=42`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best performer on nearly every metric (Accuracy 0.9825, MCC 0.9623). The 30 features are largely linearly separable after standardization, and with only ~455 training rows, a low-variance linear model generalizes better than more flexible ones. |
| Decision Tree | Weakest model overall (Accuracy 0.9123, MCC 0.8174, lowest AUC 0.9157). A single unpruned tree overfits the training split and is the most sensitive to noise in individual features — no regularization/ensembling to smooth its decision boundary. |
| kNN | Strong and consistent (Accuracy 0.9561, F1 0.9655). Performs well because standardized Euclidean distance is meaningful here — malignant/benign cases form fairly compact clusters in feature space. |
| Naive Bayes | Decent Accuracy (0.9298) but the highest AUC among the non-Logistic-Regression models (0.9868), meaning its *ranking* of probabilities is very good even though its default 0.5 threshold isn't optimal. The independence assumption between the 30 (correlated) features costs it some precision/recall at the default cutoff. |
| Random Forest (Ensemble) | Matches kNN on Accuracy/F1 (0.9561/0.9655) and has the second-best AUC (0.9932) after Logistic Regression. Bagging clearly fixes the single Decision Tree's overfitting problem — a good illustration of why ensembling helps. |
| **Overall Winner for this dataset** | **Logistic Regression** — highest Accuracy, AUC, Precision, Recall, F1, and MCC. On this dataset, standardized features are close to linearly separable, so the simplest model generalizes best. Random Forest is the strongest non-linear alternative and would likely be more robust if the dataset were noisier or had non-linear feature interactions. |

## Project Structure

```
project-folder/
│-- app.py                  # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split (features + target)
│-- metrics_summary.csv     # metrics for all 5 models
│-- model/
│   │-- train_models.py     # trains all 5 models + computes metrics
│   │-- scaler.joblib
│   │-- feature_names.joblib
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- random_forest_ensemble.joblib
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models, test_data.csv, metrics_summary.csv
streamlit run app.py
```

## Streamlit App Features

- CSV upload for test data (`test_data.csv` bundled as default)
- Model selection dropdown (all 5 trained models)
- Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap + full classification report

## Live App Link

> https://breast-cancer-ml-assign.streamlit.app/
