# 🫀 Heart Disease Classification

A machine learning project that classifies heart disease using the **XGBoost** algorithm on the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Pipeline](#pipeline)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

---

## 📌 Introduction

This project builds a machine learning model to **predict the risk of heart disease** based on patients' clinical indicators. The problem is framed as a **binary classification task** — predicting whether a patient has heart disease (1) or not (0).

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository — [Heart Disease Dataset](https://doi.org/10.24432/C52P4X)

> A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano. *"Heart Disease,"* UCI Machine Learning Repository, 1989.

**Data file:** `data/processed.cleveland.data`

### Features

| Column     | Description                                                        |
|------------|--------------------------------------------------------------------|
| `age`      | Age of the patient                                                 |
| `sex`      | Sex (1 = male, 0 = female)                                         |
| `cp`       | Chest pain type (0–3)                                              |
| `trestbps` | Resting blood pressure in mmHg — *dropped after EDA*              |
| `chol`     | Serum cholesterol in mg/dl — *dropped after EDA*                  |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true)                        |
| `restecg`  | Resting electrocardiographic results (0–2)                        |
| `thalach`  | Maximum heart rate achieved                                        |
| `exang`    | Exercise-induced angina (1 = yes)                                  |
| `oldpeak`  | ST depression induced by exercise relative to rest                |
| `slope`    | Slope of the peak exercise ST segment (0–2)                       |
| `ca`       | Number of major vessels colored by fluoroscopy (0–3)              |
| `thal`     | Thalassemia result (3 = normal, 6 = fixed defect, 7 = reversable) |
| `num`      | **Target label** — 0: no disease, 1: disease present              |

---

## 📁 Project Structure

```
heart-disease-classification/
│
├── data/
│   └── processed.cleveland.data       # Raw data from UCI
│
├── heart_disease_classification.ipynb # Main notebook
└── README.md
```

---

## ⚙️ Requirements

Python 3.8+ and the following libraries:

```bash
pip install xgboost pandas numpy matplotlib seaborn scikit-learn
```

Or install all at once:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
xgboost
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🔄 Pipeline

### 0. Import Libraries
Load all required libraries: `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

### 1. Get the Data
- Read the raw CSV file (no header) and assign column names manually.
- Convert the `num` target column into binary labels: `0` (no disease) and `1` (disease present).
- Remove rows with `?` values in the `ca` and `thal` columns, then cast them to `float`.

### 2. Exploratory Data Analysis (EDA)
- Plot **histograms** to visualize the distribution of all features.
- Detect outliers using **Boxplots** on continuous variables: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`.
- Plot **Countplots** for categorical variables.
- Compare each feature against the target label to assess influence.
- Generate a **Correlation Heatmap** to analyze feature relationships.
- **Drop** `trestbps` and `chol` due to low correlation with the target.

### 3. Train/Test Split
- Split the data with an **80/20 ratio** (`test_size=0.2`, `random_state=42`).

### 4. Normalization
- Scale training data using **StandardScaler** (`fit_transform` on `X_train`).

### 5. Model Training
- Train an `XGBClassifier` with default parameters.
- Evaluate initial accuracy on both the training and test sets.

### 6. Hyperparameter Tuning (GridSearchCV)
Search for the best hyperparameters using **5-fold Cross Validation**:

```python
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [100, 200],
    'subsample': [0.7, 1.0]
}
```

### 7. Evaluation Metrics
- **Classification Report**: Precision, Recall, and F1-score per class.
- **ROC Curve**: Plot of True Positive Rate vs. False Positive Rate.

---

## 📈 Results

| Stage                        | Accuracy |
|------------------------------|----------|
| Baseline (default params)    | ~83%     |
| After GridSearchCV tuning    | ~85%+    |

> *Results may vary slightly due to randomness in the training process.*

---

## ▶️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deckardLong/heart-disease.git
   cd heart-disease
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook heart_disease_classification.ipynb
   ```

4. Run all cells sequentially from top to bottom.

---

## 📚 References

- A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano. *"Heart Disease,"* UCI Machine Learning Repository, 1989. https://doi.org/10.24432/C52P4X
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

> 📝 **Disclaimer:** This project is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.