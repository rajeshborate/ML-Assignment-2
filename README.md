# 🧠 Adult Income Classification -- ML Assignment 2

## 📌 Problem Statement

The objective of this assignment is to implement and compare multiple
machine learning classification models to predict whether an
individual's annual income exceeds \$50,000 based on census demographic
data.

This assignment also includes the development of an interactive Streamlit
web application that allows users to: - Select different machine
learning models - Upload test datasets (CSV) - View evaluation metrics -
Visualize confusion matrices

The project demonstrates a complete end-to-end Machine Learning workflow
including preprocessing, model training, evaluation, and deployment.

------------------------------------------------------------------------

## 📊 Dataset Description

Dataset Used: UCI Adult Income Dataset

-   Number of Instances: 48,842
-   Number of Features: 14
-   Task Type: Binary Classification
-   Dataset Type: Multivariate

### Target Variable

-   `<=50K` → Income less than or equal to \$50,000
-   `>50K` → Income greater than \$50,000

### Preprocessing Steps Performed

-   Missing values handled
-   Categorical features encoded
-   Feature scaling applied
-   Train-test split (80:20)

------------------------------------------------------------------------

## 🤖 Machine Learning Models Implemented

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbor (KNN)\
4.  Gaussian Naive Bayes\
5.  Random Forest (Ensemble Model)\
6.  XGBoost (Boosting Ensemble Model)

------------------------------------------------------------------------

## 📈 Model Comparison Table

| ML Model             | Accuracy | AUC    | Precision | Recall | F1 Score | MCC   |
|----------------------|----------|--------|----------|--------|----------|-------|
| Logistic Regression  | 0.8212   | 0.7038 | 0.7347   | 0.4648 | 0.5694   | 0.4830 |
| Decision Tree        | 0.8142   | 0.7556 | 0.6340   | 0.6365 | 0.6353   | 0.5106 |
| KNN                  | 0.8270   | 0.7498 | 0.6846   | 0.5926 | 0.6353   | 0.5250 |
| Naive Bayes          | 0.7971   | 0.6495 | 0.7038   | 0.3491 | 0.4667   | 0.3922 |
| Random Forest        | 0.8586   | 0.7863 | 0.7660   | 0.6391 | 0.6968   | 0.6098 |
| XGBoost              | **0.8729** | **0.8059** | **0.7979** | **0.6696** | **0.7281** | **0.6502** |

------------------------------------------------------------------------

## 📌 Observations

| ML Model             | Performance Observation |
|----------------------|--------------------------|
| Logistic Regression  | Performs well as a baseline linear model but has lower recall for the high-income class. |
| Decision Tree        | Captures non-linear relationships but may slightly overfit the data. |
| KNN                  | Provides balanced performance but is sensitive to feature scaling. |
| Naive Bayes          | Fast and simple, but independence assumption reduces predictive power. |
| Random Forest        | Improves generalization and reduces overfitting using ensemble learning. |
| XGBoost              | Achieved the highest performance across all evaluation metrics and performed best overall. |

------------------------------------------------------------------------

## 🗂 Project Structure

    ML-Assignment-2/
    │-- app.py
    │-- train_models.py
    │-- requirements.txt
    │-- README.md
    │-- adult_clean.csv
    │-- model_results.csv
    │-- model/
    │     ├── Logistic_Regression.pkl
    │     ├── Decision_Tree.pkl
    │     ├── KNN.pkl
    │     ├── Naive_Bayes.pkl
    │     ├── Random_Forest.pkl
    │     └── XGBoost.pkl

------------------------------------------------------------------------

## 🚀 How to Run Locally

conda create -n mlassign python=3.10\
conda activate mlassign\
pip install -r requirements.txt\
python train_models.py\
streamlit run app.py

------------------------------------------------------------------------

## 📚 Reference

Becker, B. & Kohavi, R. (1996). Adult Dataset.\
UCI Machine Learning Repository.
