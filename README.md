# Leukemia-ML-Prediction
Machine Learning-Based Prediction and Evaluation of Models for  Leukemia Using the CuMiDa Dataset

A Machine Learning-based approach to predict and classify leukemia subtypes using RNA-Seq gene expression data from the CuMiDa dataset. The project demonstrates a full ML workflow including data preprocessing, exploratory data analysis, model training, evaluation, and interpretation using various classification algorithms.

---

##  Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models Used](#-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Contributors](#-contributors)
- [License](#-license)

---

##  Overview

Leukemia is a life-threatening cancer that disrupts blood and bone marrow through abnormal white blood cell proliferation. Early and accurate classification is essential. We apply supervised machine learning models to analyze gene expression patterns and predict leukemia subtypes.

---

##  Dataset

- **Source**: [Kaggle - CuMiDa Dataset](https://www.kaggle.com/datasets/brunogrisci/leukemia-gene-expression-cumida)
- **Description**: RNA-Seq gene expression dataset for leukemia subtype classification
- **Samples**: 64
- **Classes**: 
  - AML (Acute Myeloid Leukemia)
  - Peripheral Blood (PB)
  - Peripheral Blood Stem Cells CD34 (PBSC CD34)
  - Bone Marrow
  - Bone Marrow CD34

**Note**: Dataset is not included in this repo due to size constraints. Download from Kaggle and place it in the `/data` directory.

---

## Project Workflow

This project follows a complete machine learning pipeline tailored for a **multi-class classification** problem using biological RNA-Seq data.

### 🔹 1. Data Loading & Cleaning
- Load the CuMiDa dataset into a Pandas DataFrame.
- Inspect structure, shape, and data types.
- Check for missing values or anomalies.

### 🔹 2. Exploratory Data Analysis (EDA)
- Visualize class distribution, outliers, and feature trends.
- Use histograms, boxplots, correlation heatmaps, and scatter plots.
- Identify dominant gene expressions and feature interrelationships.

### 🔹 3. Feature Scaling & Encoding
- Encode target labels using `LabelEncoder`.
- Normalize features using `StandardScaler` for consistent model input.
- Split into features (`X`) and labels (`y`).

### 🔹 4. Model Training
Train and evaluate the following classification models:
- **Random Forest**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**

Split data into train/test sets using `train_test_split` with an 80:20 ratio.

### 🔹 5. Hyperparameter Tuning
(Optional but recommended)
- Use `GridSearchCV` or `RandomizedSearchCV` to fine-tune parameters:
  - `n_estimators`, `max_depth`, `learning_rate`, etc.
- 5-fold cross-validation used for stability.

### 🔹 6. Model Evaluation
- Evaluate using:
  - Accuracy
  - AUC-ROC
  - F1-Score
  - Confusion Matrix
- Plot ROC curves and confusion matrices for visual analysis.

### 🔹 7. Feature Importance Analysis
- Use **XGBoost's built-in feature importance** methods:
  - `weight`: frequency of feature used
  - `gain`: average gain from feature splits
- Visualized with:
  ```python
  xgb.plot_importance(model)
