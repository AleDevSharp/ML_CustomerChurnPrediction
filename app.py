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
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def run_random_forest_churn_prediction(filepath="datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Esegue l'intera pipeline di classificazione Random Forest per la previsione del churn dei clienti,
    inclusi caricamento dati, pre-elaborazione, addestramento del modello, valutazione e visualizzazione.

    Args:
        filepath (str): Percorso del file CSV del dataset "WA_Fn-UseC_-Telco-Customer-Churn.csv".
    """

    print("\n--- Avvio della Previsione Churn con Random Forest ---")

    # 1. Caricamento del Dataset
    print("\n[1] Caricamento del dataset...")
    try:
        df_original = pd.read_csv(filepath)
        print(f"Dataset caricato con successo: {df_original.shape[0]} righe e {df_original.shape[1]} colonne.")
    except FileNotFoundError:
        print(f"Errore: Dataset non trovato al percorso '{filepath}'. Controlla il path.")
        return
    except Exception as e:
        print(f"Si è verificato un errore durante il caricamento del dataset: {e}")
        return

    # Crea una copia per le operazioni di pre-elaborazione
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
        print(f"Rimosse le colonne non pertinenti al retail: {', '.join(existing_columns_to_drop)}.")
    else:
        print("Nessuna colonna specifica del servizio Telco trovata da rimuovere.")

    # 2. Pulizia e Preparazione Iniziale dei Dati
    print("\n[2] Pulizia e preparazione iniziale dei dati...")
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed.dropna(inplace=True)
    print(f"Righe dopo la gestione dei valori mancanti: {df_processed.shape[0]}.")

    if df_processed['SeniorCitizen'].dtype == 'object':
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({'No': 0, 'Yes': 1})

    df_processed.drop(columns=['customerID'], inplace=True)
    print("Colonna 'customerID' temporaneamente rimossa per il training.")

    # 3. Preparazione della Variabile Target
    if 'Churn' in df_processed.columns:
        df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})
        print("Variabile target 'Churn' mappata a 0 (No) e 1 (Sì).")
    else:
        print("Errore: Colonna 'Churn' non trovata. Impossibile procedere senza una variabile target.")
        return

    print("\nDistribuzione delle classi 'Churn':")
    print(df_processed['Churn'].value_counts(normalize=True))
    if df_processed['Churn'].nunique() < 2:
        print("[AVVISO] È presente una sola classe nella variabile target. Impossibile addestrare un classificatore.")
        return

    # 4. Feature Engineering ed Encoding
    print("\n[4] Codifica delle feature categoriche con One-Hot Encoding...")
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        print(f"Codificate {len(categorical_cols)} feature categoriche.")
    else:
        print("Nessuna feature categorica trovata per la codifica.")

    # 5. Suddivisione di Feature e Target
    print("\n[5] Suddivisione delle feature (X) e del target (y)...")
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    print(f"Dimensioni delle Feature (X): {X.shape}, Dimensioni del Target (y): {y.shape}")

    # 6. Suddivisione in Training e Test Set con Stratificazione (80% Training, 20% Test)
    print("\n[6] Suddivisione in set di training e test con stratificazione (80% Training, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dimensione del set di training: {X_train.shape[0]} ({X_train.shape[0] / X.shape[0]:.0%})")
    print(f"Dimensione del set di test: {X_test.shape[0]} ({X_test.shape[0] / X.shape[0]:.0%})")
    print(f"Distribuzione del target nel training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Distribuzione del target nel test set:\n{y_test.value_counts(normalize=True)}")

    # 7. Creazione di una Pipeline per Scaling e Addestramento del Modello
    print("\n[7] Impostazione di una pipeline per StandardScaler e RandomForestClassifier...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # 8. Ottimizzazione degli Iperparametri con GridSearchCV e StratifiedKFold
    print("\n[8] Esecuzione dell'ottimizzazione degli iperparametri con GridSearchCV e StratifiedKFold...")
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

    print("\nI migliori parametri trovati da GridSearchCV:")
    print(grid_search.best_params_)
    print(f"Miglior punteggio ROC AUC di cross-validation: {grid_search.best_score_:.4f}")

    model = grid_search.best_estimator_
    print("\n[9] Modello finale addestrato con i migliori parametri.")

    # 10. Effettuazione delle Previsioni sul Test Set
    print("\n[10] Effettuazione delle previsioni sul test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 11. Valutazione delle Performance del Modello
    print("\n[11] Valutazione delle performance del modello sul test set...")
    print("\n--- Report di Classificazione ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Matrice di Confusione ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Previsto No Churn', 'Previsto Churn'],
                yticklabels=['Reale No Churn', 'Reale Churn'])
    plt.title('Matrice di Confusione')
    plt.ylabel('Etichetta Reale')
    plt.xlabel('Etichetta Prevista')
    plt.show()

    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classificatore Casuale')
        plt.xlabel("Tasso di Falsi Positivi (1 - Specificità)")
        plt.ylabel("Tasso di Veri Positivi (Sensibilità)")
        plt.title("Curva ROC")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        print("\n[AVVISO] Punteggio e curva ROC AUC saltati: non entrambe le classi sono presenti in y_test o y_proba.")

    # 12. Plot delle Importanze delle Feature Principali
    print("\n[12] Plot delle 15 importanze delle feature principali...")
    try:
        feature_importances = model.named_steps['classifier'].feature_importances_
        features = X.columns
        if len(feature_importances) == len(features):
            importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
            importance_df = importance_df.sort_values(by='importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importance_df.head(15), palette='viridis')
            plt.title("Top 15 Importanze delle Feature (Random Forest)", fontsize=16)
            plt.xlabel("Importanza", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.tight_layout()
            plt.show()
        else:
            print("[AVVISO] Impossibile plottare le importanze delle feature: disallineamento tra numero di feature e importanze.")
    except Exception as e:
        print(f"[ERRORE] durante il plot delle importanze: {e}")

    # 13. Generazione delle Probabilità di Churn e delle Previsioni per tutti i clienti
    print("\n[13] Generazione delle probabilità di churn e delle previsioni per tutti i clienti originali...")

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
    X_full_predict_encoded = pd.get_dummies(df_for_full_prediction_features, columns=categorical_cols_full_predict, drop_first=True)

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

    # 14. Top 10 clienti più a rischio di Churn
    print("\n[14] Identificazione dei 10 clienti più a rischio di churn...")
    top_churn_risk_customers = results_df.sort_values(
        by='Churn_Probability', ascending=False
    ).head(10)
    print("\nTop 10 clienti più a rischio di churn:")
    print(top_churn_risk_customers)

    print("\n--- Processo completato con successo! ---")

if __name__ == '__main__':
    run_random_forest_churn_prediction()