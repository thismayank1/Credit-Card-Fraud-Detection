# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. The dataset used contains anonymized transaction data, and the goal is to accurately classify fraudulent transactions.

## Features
- **Data Preprocessing:** Standardization of numerical features.
- **Handling Class Imbalance:** Using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Classification Model:** Logistic Regression model trained to detect fraudulent transactions.
- **Evaluation Metrics:** Classification report, confusion matrix, and ROC-AUC score.

## Dataset
The dataset used in this project is `creditcard.csv`, which contains anonymized transaction details along with a fraud label (`Class` column: 0 for legitimate, 1 for fraudulent).

## Model Performance
Below are the evaluation metrics obtained after training the model:

### Classification Report

![Screenshot 2025-04-03 132519](https://github.com/user-attachments/assets/ef91656a-7db9-4712-a4c9-7dedb26e7a27)


| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.93      | 0.98   | 0.95     | 56750   |
| 1     | 0.97      | 0.92   | 0.95     | 56976   |

**Accuracy:** 95%  
**Macro Avg:** Precision: 0.95, Recall: 0.95, F1-score: 0.95  
**Weighted Avg:** Precision: 0.95, Recall: 0.95, F1-score: 0.95  

### Confusion Matrix
```
[[55354  1396]
 [ 4284 52692]]
```

### ROC-AUC Score
```
0.9501
```

## How to Run
### Install Dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### Run the Script
```bash
python Credit_Card_Fraud_Detection.py
```
