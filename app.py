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


def run_random_forest_churn_prediction_adapted(filepath="datasets/Telco_customer_churn.csv"):
    """
    Performs the entire Random Forest classification pipeline for customer churn prediction,
    adapted for the new CSV structure, including data loading, preprocessing,
    model training, evaluation and visualization.

    Args:
        filepath (str): The file path to the new customer churn CSV.
    """

    print("\n--- Starting Churn Prediction ---")

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

    # Create a copy for preprocessing operations for training
    df_processed = df_original.copy()

    # Columns to remove for features
    columns_to_remove_from_features = [
        'Count', 'Churn Label', 'Churn Score', 'CLTV',
        'Churn Reason', 'Phone Service', 'Multiple Lines', 'Internet Service',
        'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
        'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method'
    ]

    customer_id_col = 'CustomerID'
    if customer_id_col in df_processed.columns:
        if customer_id_col in columns_to_remove_from_features:
            columns_to_remove_from_features.remove(customer_id_col)
    else:
        print(f"Warning: '{customer_id_col}' column not found. Results will not be mapped to customer IDs.")
        customer_id_col = None

    # Drop the specified columns from df_processed (for training)
    existing_columns_to_drop_for_training = [col for col in columns_to_remove_from_features if
                                             col in df_processed.columns]
    if customer_id_col and customer_id_col in df_processed.columns:
        existing_columns_to_drop_for_training.append(customer_id_col)

    if existing_columns_to_drop_for_training:
        df_processed.drop(columns=existing_columns_to_drop_for_training, inplace=True)
        print(f"Columns removed from training data: {', '.join(existing_columns_to_drop_for_training)}.")
    else:
        print("No specified columns found to remove from training data.")

    # 2. Initial Data Cleaning and Preparation (for df_processed)
    print("\n[2] Initial Data Cleaning and Preparation (for training data)...")
    if 'Total Charges' in df_processed.columns:
        df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
        df_processed.dropna(subset=['Total Charges'], inplace=True)
        print(f"Rows after handling missing 'Total Charges' in training data: {df_processed.shape[0]}.")
    else:
        print(
            "Warning: 'Total Charges' column not found in training data. Skipping numeric conversion and NaN handling.")

    if 'Senior Citizen' in df_processed.columns and df_processed['Senior Citizen'].dtype == 'object':
        df_processed['Senior Citizen'] = df_processed['Senior Citizen'].map({'No': 0, 'Yes': 1})
        print("'Senior Citizen' column mapped to 0 (No) and 1 (Yes) in training data.")
    elif 'Senior Citizen' not in df_processed.columns:
        print("Warning: 'Senior Citizen' column not found in training data.")

    # 3. Target Variable Preparation
    target_column = 'Churn Value'
    if target_column in df_processed.columns:
        if df_processed[target_column].dtype == 'object':
            df_processed[target_column] = df_processed[target_column].map({'No': 0, 'Yes': 1})
            print(f"Target variable '{target_column}' mapped to 0 (No) and 1 (Yes).")
        else:
            print(f"Target variable '{target_column}' is already numeric.")
    else:
        print(f"Error: '{target_column}' column not found. Cannot proceed without a target variable.")
        return

    print(f"\n'{target_column}' class distribution:")
    print(df_processed[target_column].value_counts(normalize=True))
    if df_processed[target_column].nunique() < 2:
        print(f"[WARNING] Only one class present in the target variable '{target_column}'. Cannot train a classifier.")
        return

    # 4. Feature Engineering and Encoding (for df_processed)
    print("\n[4] Encoding categorical features with One-Hot Encoding (for training data)...")
    categorical_cols_training = df_processed.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols_training:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols_training, drop_first=True)
        print(f"Encoded {len(categorical_cols_training)} categorical features for training.")
    else:
        print("No categorical features found for encoding in training data.")

    # 5. Splitting Features and Target
    print("\n[5] Splitting features (X) and target (y)...")
    x = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]

    print(f"Feature (X) dimensions: {x.shape}, Target (y) dimensions: {y.shape}")

    # 6. Splitting into Training and Test Sets with Stratification (80% Training, 20% Test)
    print("\n[6] Splitting into training and test sets with stratification (80% Training, 20% Test)...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {x_train.shape[0]} ({x_train.shape[0] / x.shape[0]:.0%})")
    print(f"Test set size: {x_test.shape[0]} ({x_test.shape[0] / x.shape[0]:.0%})")
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
    grid_search.fit(x_train, y_train)

    print("\nBest parameters found by GridSearchCV:")
    print(grid_search.best_params_)
    print(f"Best cross-validation ROC AUC score: {grid_search.best_score_:.4f}")

    model = grid_search.best_estimator_
    print("\n[9] Final model trained with the best parameters.")

    # 10. Making Predictions on the Test Set
    print("\n[10] Making predictions on the test set...")
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

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

    # 12. Plotting Top Feature Importance's
    print("\n[12] Plotting top 15 feature importances...")
    try:
        feature_importances = model.named_steps['classifier'].feature_importances_
        features = x.columns
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

    customer_ids_with_index = None
    if customer_id_col and customer_id_col in df_original.columns:
        customer_ids_with_index = df_original[[customer_id_col]].copy()

    df_for_full_prediction = df_original.copy()

    existing_cols_to_drop_for_full_predict = [col for col in columns_to_remove_from_features if
                                              col in df_for_full_prediction.columns]
    if customer_id_col and customer_id_col in df_for_full_prediction.columns:
        existing_cols_to_drop_for_full_predict.append(customer_id_col)

    if existing_cols_to_drop_for_full_predict:
        df_for_full_prediction_features_only = df_for_full_prediction.drop(
            columns=existing_cols_to_drop_for_full_predict, errors='ignore'
        )
    else:
        df_for_full_prediction_features_only = df_for_full_prediction.copy()

    # Clean 'Total Charges' from dataframe
    if 'Total Charges' in df_for_full_prediction_features_only.columns:
        df_for_full_prediction_features_only['Total Charges'] = pd.to_numeric(
            df_for_full_prediction_features_only['Total Charges'], errors='coerce')

        df_for_full_prediction_features_only.dropna(
            subset=['Total Charges'], inplace=True)

    # Map senior citizen
    if 'Senior Citizen' in df_for_full_prediction_features_only.columns and \
            df_for_full_prediction_features_only['Senior Citizen'].dtype == 'object':
        df_for_full_prediction_features_only['Senior Citizen'] = df_for_full_prediction_features_only[
            'Senior Citizen'].map({'No': 0, 'Yes': 1})

    categorical_cols_full_predict = df_for_full_prediction_features_only.select_dtypes(
        include=['object']).columns.tolist()
    x_full_predict_encoded = pd.get_dummies(
        df_for_full_prediction_features_only, columns=categorical_cols_full_predict, drop_first=True
    )

    missing_cols = set(x.columns) - set(x_full_predict_encoded.columns)
    for col in missing_cols:
        x_full_predict_encoded[col] = 0

    extra_cols = set(x_full_predict_encoded.columns) - set(x.columns)
    x_full_predict_encoded.drop(columns=list(extra_cols), inplace=True)

    x_full_predict_aligned = x_full_predict_encoded[x.columns]

    # Prevision
    churn_probabilities_full = model.predict_proba(x_full_predict_aligned)[:, 1]
    predicted_churn_full = model.predict(x_full_predict_aligned)

    # Create DataFrame
    results_prediction_data = pd.DataFrame({
        'Churn_Probability': churn_probabilities_full,
        'Predicted_Churn': predicted_churn_full
    }, index=x_full_predict_aligned.index)

    # Original Customer id
    if customer_ids_with_index is not None:
        results_df = customer_ids_with_index.merge(results_prediction_data, left_index=True, right_index=True,
                                                   how='inner')
    else:
        results_df = results_prediction_data

    results_df['Predicted_Churn'] = results_df['Predicted_Churn'].map({0: 'No', 1: 'Yes'})

    # 14. Top 10 Customers Most At Risk of Churn
    print("\n[14] Identifying top 10 customers most at risk of churn...")
    if customer_id_col and customer_id_col in results_df.columns:
        top_churn_risk_customers = results_df.sort_values(
            by='Churn_Probability', ascending=False
        ).head(10)
        print("\nTop 10 customers most at risk of churn:")
        print(top_churn_risk_customers)
    else:
        print("Skipping top 10 customers report as 'CustomerID' is not available in results_df.")

    print("\n--- Finish Churn Prediction ---")


if __name__ == '__main__':
    run_random_forest_churn_prediction_adapted()
