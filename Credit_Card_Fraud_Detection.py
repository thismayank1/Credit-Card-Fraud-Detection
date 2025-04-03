import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

# Load dataset
df = pd.read_csv("creditcard.csv")

# Data preprocessing
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("### Classification Report")
print(classification_report(y_test, y_pred))
print("### Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("### ROC-AUC Score")
print(roc_auc_score(y_test, y_pred))

# Save the model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
