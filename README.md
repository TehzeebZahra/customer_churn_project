# Customer Churn Prediction & Business Intelligence System

## Overview
An end-to-end **Data Science and Business Intelligence project** that analyzes customer behavior, predicts churn, identifies important churn factors, and provides business insights using **Python, Machine Learning, SQL, Streamlit, and Power BI**.

## Objectives

* Clean and preprocess customer data
* Perform EDA and feature engineering
* Train and compare ML models
* Predict customer churn probability
* Identify important churn factors
* Analyze churn using SQL
* Build Streamlit and Power BI dashboards

## Technologies

* Python
* Pandas & NumPy
* Matplotlib, Seaborn & Plotly
* Scikit-learn & Joblib
* MySQL/MariaDB & SQL
* Streamlit
* Power BI

## Dataset

The dataset contains customer demographics, services, contracts, payment methods, tenure, and charges.

Target: "Churn"

* "1" = Churned
* "0" = Not Churned

## Workflow

"""text
Raw Data
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Churn Prediction
   ↓
SQL Analysis
   ↓
Streamlit + Power BI
"""
# Machine Learning

The project compares:

* Logistic Regression
* Decision Tree
* Random Forest

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Model results and feature importance are stored in:

```text
models/model_results.csv
models/feature_importance.csv
```

The trained model is stored as:

```text
models/churn_model.pkl
```

## Dashboards

### Streamlit

Interactive dashboard showing:

* Customer KPIs
* Churn rate
* Churn distribution
* Contract analysis
* High-risk customers

Run with:

```bash
streamlit run dashboard/app.py
```

### Power BI

Business dashboard containing:

* Total Customers
* Churned Customers
* Churn Rate
* Customer Segments
* Churn Analysis
* Business KPIs

## SQL Database

Customer data is stored in ""MySQL/MariaDB"".

```text
Database: customer_churn_db
Port: 3307
```

Database connection:

```text
src/database_connection.py
```

##  Project Structure

```text
customer_churn_project/
│
├── data/
├── notebooks/
├── src/
├── models/
├── dashboard/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

##  Business Insights

The system helps businesses identify high-risk customers and supports retention strategies such as:

* Targeting high-risk customers
* Promoting long-term contracts
* Monitoring short-tenure customers
* Reviewing high monthly charges
* Providing proactive customer support

##  Conclusion

This project demonstrates a complete **end-to-end data science workflow**, combining machine learning, SQL, dashboards, and business analytics to support customer retention decisions.
