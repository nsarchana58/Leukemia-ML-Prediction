# 🧬 Leukemia Type Classification using Gene Expression Data (CuMiDa)

A Machine Learning-based classification project focused on predicting leukemia subtypes using RNA-Seq gene expression data from the CuMiDa dataset. This repository includes preprocessing, exploratory analysis, model training, evaluation, and feature interpretation using multiple ML algorithms.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models Used](#-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Contributor](#-contributor)
  
---

## 🧠 Overview

Leukemia is a cancer of the blood and bone marrow that impairs the normal production and function of white blood cells. Accurate and early classification of leukemia subtypes is crucial for effective treatment planning and prognosis. With advancements in bioinformatics, RNA-Seq gene expression profiling offers a data-rich source for computational disease modeling.

In this project, we apply various supervised machine learning models to predict the type of leukemia based on gene expression features. Our goal is to evaluate multiple algorithms for performance and interpretability, ultimately identifying the most effective method for this biomedical classification task.

---

## 📊 Dataset

- **Name**: CuMiDa – _Curated Microarray Database_
- **Source**: [Kaggle - Leukemia Gene Expression Dataset](https://www.kaggle.com/datasets/brunogrisci/leukemia-gene-expression-cumida)
- **Samples**: 64 gene expression profiles
- **Features**: Preprocessed gene expression values from RNA-Seq
- **Target Classes**:
  - AML (Acute Myeloid Leukemia)
  - Peripheral Blood (PB)
  - Peripheral Blood Stem Cells CD34 (PBSC CD34)
  - Bone Marrow
  - Bone Marrow CD34

> ⚠️ The dataset is not included in this repository due to size and licensing constraints. Please download it manually from the Kaggle link above and place it inside the `/data/` directory.

---

## 🔁 Project Workflow

This project follows a standard ML pipeline adapted for biomedical data:

### 1️⃣ Data Loading & Cleaning
- Load the CuMiDa dataset into a DataFrame
- Handle missing values and verify class distributions
- Drop irrelevant or constant features if present

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualize class balance
- Plot gene expression distributions using boxplots and histograms
- Use correlation heatmaps to assess feature relationships

### 3️⃣ Feature Engineering
- Encode categorical labels into numerical values
- Scale features using `StandardScaler` for uniform input

### 4️⃣ Model Training
- Split data into training (80%) and test (20%) sets
- Train multiple classification models
- Use consistent seeds and evaluation splits for reproducibility

### 5️⃣ Hyperparameter Tuning
- Apply `GridSearchCV` to optimize model parameters
- Validate using 5-fold cross-validation (optional for small data)

### 6️⃣ Model Evaluation
- Evaluate models on test data using accuracy, ROC-AUC, and confusion matrix
- Compare performance across all models

### 7️⃣ Feature Importance
- Use XGBoost’s `plot_importance()` to rank gene features
- Interpret which genes contribute most to predictions

---

## 🤖 Models Used

We implemented and compared the following classification algorithms:

| Model                 | Description |
|----------------------|-------------|
| **Logistic Regression** | A linear classifier used as a baseline |
| **K-Nearest Neighbors (KNN)** | Distance-based classification using nearest data points |
| **Random Forest** | Ensemble learning with decision trees |
| **XGBoost** | Gradient Boosted Trees optimized for performance and interpretability |

Each model was trained and evaluated under consistent conditions for fair comparison.

---

## 📈 Evaluation Metrics

We assessed each model using the following metrics:

- **Accuracy** – Percentage of correct predictions
- **Confusion Matrix** – Counts of TP, FP, TN, FN for each class
- **Precision, Recall, F1-Score** – Class-specific performance
- **AUC-ROC** – Area under the ROC curve for multi-class classification
- **XGBoost Feature Importance** – Gene-level contribution to predictions

Visual plots (heatmaps, confusion matrices, ROC curves) are saved in `/plots/`.

---

## 📊 Results
✅ Logistic Regression

Accuracy: 94%
AUC-ROC: 0.97

✅ K-Nearest Neighbors (KNN)

Accuracy: 91%
AUC-ROC: 0.94

✅ Random Forest

Accuracy: 96%
AUC-ROC: 0.99

✅ XGBoost

Accuracy: 98%
AUC-ROC: 1.00

## ⚠️ Note: A perfect AUC-ROC of 1.00 for XGBoost may indicate overfitting, especially given the small dataset size. Additional validation and more samples are recommended for reliable generalization.

## 👨‍🔬 Contributor
Archana NS 
