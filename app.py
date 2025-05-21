"""
------------------------------------------------------------------------
File : app.py
Description: Main project
Date creation: 21-05-2025
Project : soup-server
Author: Alessio Giacché, Matteo Brachetta, Lorenzo
Copyright: Copyright (c) 2024 Alessio Giacché <ale.giacc.dev@gmail.com>
License : MIT
------------------------------------------------------------------------
"""

# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML Import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. Load dataset
print("\n[1] Loading dataset...")
df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Data preprocessing
print("\n[2] Preprocessing data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 3. Drop irrelevant columns
print("\n[3] Dropping irrelevant columns...")
irrelevant_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
                   'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason']
df.drop(columns=irrelevant_cols, inplace=True, errors='ignore')

# 4. Use the target variable 'Churn Value' directly
print("\n[4] Using target variable 'Churn Value'...")
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Check class distribution
print("\nChurn class distribution:")
print(df['Churn'].value_counts(normalize=True))

# 5. Encode categorical features using one-hot encoding
print("\n[5] Encoding categorical features...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 6. Split features and target
print("\n[6] Splitting features and target...")
X = df.drop(['Churn', 'Churn Value'], axis=1, errors='ignore')
y = df['Churn']

# 7. Scale features
print("\n[7] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Train/test split
print("\n[8] Splitting into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 9. Train Random Forest with balanced class weights
print("\n[9] Training Random Forest model with class_weight='balanced'...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 10. Make predictions
print("\n[10] Making predictions...")
y_pred = model.predict(X_test)

# 11. Evaluate model performance
print("\n[11] Evaluating model performance...")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Try ROC AUC only if both classes are present in test set
if len(np.unique(y_test)) == 2:
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    print("\nROC AUC Score:", roc_auc)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("\n[WARNING] ROC AUC score and curve skipped: only one class present in y_test.")

# 12. Plot top 10 feature importances
print("\n[12] Plotting top 10 feature importances...")
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
features = np.array(X.columns)[indices]

plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.barh(features, importances[indices], color='skyblue')
plt.xlabel("Importance")
plt.tight_layout()
plt.grid()
plt.show()

print("\n[13] Generating churn probabilities for all customers...")
df['Churn_Probability'] = model.predict_proba(X_scaled)[:, 1]

# 14. Add predictioned
df['Predicted_Churn'] = model.predict(X_scaled)

# 15. Top 10 customers most at risk of churn
# TODO: add top 10 customers most at risk of churn


print("\n[FINISH] Process completed successfully.")

if __name__ == '__main__':
    print('Hello World')
