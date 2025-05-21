# ML_CustomerChurnPrediction
This project focuses on predicting customer churn in the telecommunications industry using machine learning. Churn, or customer attrition, occurs when customers discontinue their subscription or service, representing a significant challenge and loss of revenue for telecom companies. Early identification of customers likely to churn enables businesses to implement targeted retention strategies, personalized offers, and improve overall customer satisfaction.

The main objective of this project is to build a robust predictive model that can accurately classify customers at risk of churn based on a variety of factors such as demographics, contract type, service usage, and billing information. The model serves as a decision support tool to help marketing and customer service teams prioritize their efforts on those customers who need the most attention.

Key project requirements include:
- Handling real-world data issues such as missing values and categorical variables
- Addressing class imbalance since churners typically represent a minority class
- Providing interpretable results to guide business actions
- Delivering actionable insights such as a ranked list of high-risk customers
- Evaluating model performance with appropriate metrics (accuracy, recall, ROC AUC)

This solution is designed to integrate seamlessly with existing business workflows, potentially allowing deployment in CRM systems or dashboards for continuous monitoring and intervention.

## Project Overview
The dataset used is a publicly available Telco Customer Churn dataset, containing customer demographic info, service details, contract and billing information, and whether the customer has churned. The project pipeline includes:

### Methodology
1. **Data Preprocessing**
    - Convert data types (e.g., TotalCharges to numeric)
    - Handle missing data by removing or imputing
    - Drop irrelevant or redundant columns such as customer IDs and location data

2. **Feature Engineering**
     - Encode categorical variables using one-hot encoding
     - Scale numerical features using StandardScaler to normalize feature ranges

3. **Model Training**
    - Split data into training and test sets with stratification to preserve class distribution
    - Use Random Forest classifier with class_weight='balanced' to handle class imbalance

4. **Model Evaluation**
      - Calculate classification metrics: precision, recall, F1-score, accuracy
      - Plot confusion matrix and ROC curve
      - Compute ROC AUC score as overall model discrimination metric

5. **Interpretation and Insights**
      - Extract feature importances from the Random Forest model
      - Identify the top features influencing churn prediction
      - Generate churn probabilities for each customer and rank them by risk

## Dataset
The dataset used in this project is the publicly available Telco Customer Churn dataset, containing information about customers of a telecommunications company. It includes:

- Customer demographic details (e.g., Senior Citizen status)
- Account information such as tenure, contract type, and payment method
- Service subscriptions (phone, internet, streaming, etc.)
- Billing details including monthly and total charges
- Target variable indicating if the customer churned (Yes/No)
- The dataset required preprocessing steps such as converting numerical columns, handling missing values, and encoding categorical variables.

## How To Run
1. Install required packages:
   ```python
   pip install -r requirements.txt
   ```

1. Run the main script:
   ```python
   python app.py
   ```

3. Output
   - Console logs showing preprocessing, training, and evaluation details
   - ROC curve and feature importance plots
   - CSV file output/top_10_churn_risk.csv with the top 10 customers at highest churn risk

## Results and Insights

- The model achieves a balanced performance with an ROC AUC score around 0.83, indicating good discrimination ability between churners and non-churners.
- Feature importance highlights factors like contract type, tenure, monthly charges, and total charges as key predictors of churn.
- The ranked list of customers at highest risk can directly support targeted retention campaigns.

## Next Steps and Improvements
- Improve recall to better capture at-risk customers
- Apply interpretability techniques like SHAP or LIME for local prediction explanations
- Experiment with other models and hyperparameter tuning
- Deploy model in real-time systems (CRM, dashboards) for operational use
- Update model periodically with new data to maintain performance
