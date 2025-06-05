"""
------------------------------------------------------------------------
File : app.py
Description: Main project for Telco Customer Churn Prediction using Random Forest
Date creation: 21-05-2025 (Updated: 29-05-2025)
Project : soup-server
Author: Alessio Giacché, Matteo Brachetta, Lorenzo
Copyright: Copyright (c) 2024 Alessio Giacché <ale.giacc.dev@gmail.com>
License : MIT
------------------------------------------------------------------------
"""

# Core Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings('ignore')


def run_random_forest_churn_prediction(filepath="datasets/Telco-Customer-Churn.csv"):
    """
    Performs the entire Random Forest classification pipeline for customer churn prediction,
    including data loading, preprocessing, model training, evaluation and visualization.

    Args:
        filepath (str): the file path "Telco-Customer-Churn.csv".
    """

    print("\n--- Starting Churn Prediction with Random Forest ---")

    # 1. Dataset loading
    print("\n[1] Dataset Load...")
    try:
        df_original = pd.read_csv(filepath)
        print(f"Dataset loaded successfully: {df_original.shape[0]} rows and {df_original.shape[1]} columns.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at path '{filepath}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # Create a copy for preprocessing operations
    df_processed = df_original.copy()

    # Define columns to drop that are not typical for retail activities
    # These are telco-specific services or contract details.
    columns_to_drop_retail = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # Drop the specified columns if they exist in the DataFrame
    existing_columns_to_drop = [col for col in columns_to_drop_retail if col in df_processed.columns]
    if existing_columns_to_drop:
        df_processed.drop(columns=existing_columns_to_drop, inplace=True)
        print(f"Columns not relevant to retail removed: {', '.join(existing_columns_to_drop)}.")
    else:
        print("No Telco-specific service columns found to remove.")

    # 2. Initial Data Cleaning and Preparation
    print("\n[2] Initial Data Cleaning and Preparation...")
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed.dropna(inplace=True)
    print(f"Rows after handling missing values: {df_processed.shape[0]}.")

    if df_processed['SeniorCitizen'].dtype == 'object':
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({'No': 0, 'Yes': 1})

    df_processed.drop(columns=['customerID'], inplace=True)
    print("'customerID' column temporarily removed for training.")

    # 3. Target Variable Preparation
    if 'Churn' in df_processed.columns:
        df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})
        print("Target variable 'Churn' mapped to 0 (No) and 1 (Yes).")
    else:
        print("Error: 'Churn' column not found. Cannot proceed without a target variable.")
        return

    print("\n'Churn' class distribution:")
    print(df_processed['Churn'].value_counts(normalize=True))
    if df_processed['Churn'].nunique() < 2:
        print("[WARNING] Only one class present in the target variable. Cannot train a classifier.")
        return

    # 4. Feature Engineering and Encoding
    print("\n[4] Encoding categorical features with One-Hot Encoding...")
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        print(f"Encoded {len(categorical_cols)} categorical features.")
    else:
        print("No categorical features found for encoding.")

    # 5. Splitting Features and Target
    print("\n[5] Splitting features (X) and target (y)...")
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    print(f"Feature (X) dimensions: {X.shape}, Target (y) dimensions: {y.shape}")

    # 6. Splitting into Training and Test Sets with Stratification (80% Training, 20% Test)
    print("\n[6] Splitting into training and test sets with stratification (80% Training, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0] / X.shape[0]:.0%})")
    print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0] / X.shape[0]:.0%})")
    print(f"Target distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Target distribution in test set:\n{y_test.value_counts(normalize=True)}")

    # 7. Creating a Pipeline for Scaling and Model Training
    print("\n[7] Setting up a pipeline for StandardScaler and RandomForestClassifier...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # 8. Hyperparameter Optimization with GridSearchCV and StratifiedKFold
    print("\n[8] Executing hyperparameter optimization with GridSearchCV and StratifiedKFold...")
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_features': ['sqrt'],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = 'roc_auc'

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv_strategy, scoring=scorer, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found by GridSearchCV:")
    print(grid_search.best_params_)
    print(f"Best cross-validation ROC AUC score: {grid_search.best_score_:.4f}")

    model = grid_search.best_estimator_
    print("\n[9] Final model trained with the best parameters.")

    # 10. Making Predictions on the Test Set
    print("\n[10] Making predictions on the test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 11. Evaluating Model Performance
    print("\n[11] Evaluating model performance on the test set...")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Churn', 'Predicted Churn'],
                yticklabels=['Actual No Churn', 'Actual Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        print("\n[WARNING] ROC AUC score and curve skipped: not both classes are present in y_test or y_proba.")

    # 12. Plotting Top Feature Importances
    print("\n[12] Plotting top 15 feature ...")
    try:
        feature_importances = model.named_steps['classifier'].feature_importances_
        features = X.columns
        if len(feature_importances) == len(features):
            importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
            importance_df = importance_df.sort_values(by='importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importance_df.head(15), palette='viridis')
            plt.title("Top 15 Feature Importances (Random Forest)", fontsize=16)
            plt.xlabel("Importance", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.tight_layout()
            plt.show()
        else:
            print(
                "[WARNING] Cannot plot feature importances: misalignment between number of features and importances.")
    except Exception as e:
        print(f"[ERROR] during importance plot: {e}")

    # 13. Generating Churn Probabilities and Predictions for all Customers
    print("\n[13] Generating churn probabilities and predictions for all original customers...")

    df_for_full_prediction = df_original.copy()

    # Re-apply the same column dropping for retail services to the full prediction dataframe
    if existing_columns_to_drop:
        df_for_full_prediction.drop(columns=existing_columns_to_drop, inplace=True)

    df_for_full_prediction['TotalCharges'] = pd.to_numeric(df_for_full_prediction['TotalCharges'], errors='coerce')
    df_for_full_prediction.dropna(subset=['TotalCharges'], inplace=True)

    if 'SeniorCitizen' in df_for_full_prediction.columns and df_for_full_prediction['SeniorCitizen'].dtype == 'object':
        df_for_full_prediction['SeniorCitizen'] = df_for_full_prediction['SeniorCitizen'].map({'No': 0, 'Yes': 1})

    final_customer_ids = df_for_full_prediction['customerID']
    cols_to_drop_for_full_predict = ['customerID']
    if 'Churn' in df_for_full_prediction.columns:
        cols_to_drop_for_full_predict.append('Churn')

    df_for_full_prediction_features = df_for_full_prediction.drop(columns=cols_to_drop_for_full_predict)

    categorical_cols_full_predict = df_for_full_prediction_features.select_dtypes(include=['object']).columns.tolist()
    X_full_predict_encoded = pd.get_dummies(df_for_full_prediction_features, columns=categorical_cols_full_predict,
                                            drop_first=True)

    missing_cols_in_full = set(X.columns) - set(X_full_predict_encoded.columns)
    for c in missing_cols_in_full:
        X_full_predict_encoded[c] = 0

    extra_cols_in_full = set(X_full_predict_encoded.columns) - set(X.columns)
    X_full_predict_encoded.drop(columns=list(extra_cols_in_full), inplace=True)

    X_full_predict_aligned = X_full_predict_encoded[X.columns]

    churn_probabilities_full = model.predict_proba(X_full_predict_aligned)[:, 1]
    predicted_churn_full = model.predict(X_full_predict_aligned)

    results_df = pd.DataFrame({
        'customerID': final_customer_ids.values,
        'Churn_Probability': churn_probabilities_full,
        'Predicted_Churn': predicted_churn_full
    })

    results_df['Predicted_Churn'] = results_df['Predicted_Churn'].map({0: 'No', 1: 'Yes'})

    # 14. Top 10 Customers Most At Risk of Churn
    print("\n[14] Identifying top 10 customers most at risk of churn...")
    top_churn_risk_customers = results_df.sort_values(
        by='Churn_Probability', ascending=False
    ).head(10)
    print("\nTop 10 customers most at risk of churn:")
    print(top_churn_risk_customers)

    print("\n--- Process completed successfully! ---")


if __name__ == '__main__':
    run_random_forest_churn_prediction()
