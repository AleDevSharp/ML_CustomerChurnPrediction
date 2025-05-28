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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer, \
    recall_score, precision_score, f1_score

# Per bilanciare il dataset con SMOTE (opzionale, ma consigliato per futuri miglioramenti)
# from imblearn.over_sampling import SMOTE
# from imbleblearn.pipeline import Pipeline as ImbPipeline # Usare Pipeline di imblearn con SMOTE

# 1. Load dataset
print("\n[1] Loading dataset...")
df_original = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Create a working copy of the dataframe
df = df_original.copy()

# 2. Data preprocessing
print("\n[2] Preprocessing data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Keep track of which customerIDs are retained after dropna for final output
customer_ids_aligned_with_X = df['customerID']  # This will be used at the end

# 3. Drop irrelevant columns (excluding customerID for now)
print("\n[3] Dropping irrelevant columns...")
irrelevant_cols = ['Count', 'Country', 'State', 'City', 'Zip Code',
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
# Ensure 'customerID' is not treated as a categorical feature for one-hot encoding
if 'customerID' in categorical_cols:
    categorical_cols.remove('customerID')
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 6. Split features and target
print("\n[6] Splitting features and target...")
# Separate customerID from features before scaling/training
X = df.drop(['Churn', 'Churn Value', 'customerID'], axis=1, errors='ignore')
y = df['Churn']

# 7. Scale features
print("\n[7] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Train/test split
print("\n[8] Splitting into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 9. Train Random Forest with hyperparameter tuning using GridSearchCV
print("\n[9] Training Random Forest model with GridSearchCV for hyperparameter tuning...")

# Definisci il range di iperparametri da testare
param_grid = {
    'n_estimators': [100, 200, 300],  # Numero di alberi nella foresta. Aumentare può migliorare ma rallentare.
    'max_features': ['sqrt', 'log2'],  # Numero di feature da considerare per il miglior split.
    'max_depth': [10, 20, 30, None],
    # Profondità massima dell'albero. None significa nodi espansi fino a quando non sono puri.
    'min_samples_split': [2, 5, 10],  # Numero minimo di campioni richiesti per dividere un nodo interno.
    'min_samples_leaf': [1, 2, 4]  # Numero minimo di campioni richiesti per essere a un nodo foglia.
}

# Inizializza il modello Random Forest con class_weight='balanced'
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Utilizza StratifiedKFold per garantire che la proporzione delle classi sia mantenuta in ogni fold
# Questo è importante per i dataset sbilanciati
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Crea un scorer personalizzato che tenga conto del recall per la classe 1 (churn)
# Puoi anche usare 'roc_auc' o 'f1' come scoring. 'recall' per la classe 1 è spesso cruciale per il churn.
scorer = make_scorer(recall_score, pos_label=1)  # Ottimizza per il recall della classe '1' (churn)

# Inizializza GridSearchCV
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           scoring=scorer,  # La metrica su cui ottimizzare
                           cv=cv,  # Cross-validation strategy
                           n_jobs=-1,  # Usa tutti i core del processore disponibili
                           verbose=2,  # Mostra i progressi
                           refit=True)  # Addestra il modello migliore su tutto il set di training

# Esegui la ricerca
grid_search.fit(X_train, y_train)

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score (Recall): {grid_search.best_score_:.4f}")

# Il modello migliore è ora disponibile
model = grid_search.best_estimator_

# 10. Make predictions
print("\n[10] Making predictions with the optimized model...")
y_pred = model.predict(X_test)

# 11. Evaluate model performance
print("\n[11] Evaluating optimized model performance...")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

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
print("\n[12] Plotting top 10 feature importances for the optimized model...")
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
features = np.array(X.columns)[indices]

plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances (Optimized Model)")
plt.barh(features, importances[indices], color='skyblue')
plt.xlabel("Importance")
plt.tight_layout()
plt.grid()
plt.show()

print("\n[13] Generating churn probabilities for all customers with optimized model...")

# Generate probabilities for the entire dataset (X_scaled)
df_results = pd.DataFrame({
    'customerID': customer_ids_aligned_with_X,
    'Churn_Probability': model.predict_proba(X_scaled)[:, 1],
    'Predicted_Churn': model.predict(X_scaled)
})

# 15. Top N customers most at risk of churn
print("\n[15] Identifying top N customers most at risk of churn with optimized model...")
n_top_customers = 10  # You can change this value

# Sort by Churn_Probability in descending order
top_n_at_risk = df_results.sort_values(by='Churn_Probability', ascending=False).head(n_top_customers)

print(f"\nTop {n_top_customers} Customers Most At Risk of Churn:")
print(top_n_at_risk)

print("\n[FINISH] Process completed successfully.")

if __name__ == '__main__':
    print('Hello World')
