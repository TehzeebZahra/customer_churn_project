import pandas as pd
import joblib

print("PREDICT FILE STARTED")


# ==========================================
# 1. Load Trained Model
# ==========================================

model_path = (
    r"C:\Users\AA\Desktop\Internship"
    r"\customer_churn_project\models\churn_model.pkl"
)

model = joblib.load(model_path)

print("✅ Model loaded successfully!")


# ==========================================
# 2. Customer ID
# ==========================================

customer_id = "CUST-10294"


# ==========================================
# 3. Customer Data
# ==========================================

customer = pd.DataFrame({

    "gender": ["Female"],

    # IMPORTANT:
    # SeniorCitizen is numeric in the dataset
    "SeniorCitizen": [0],

    "Partner": ["No"],

    "Dependents": ["No"],

    "tenure": [5],

    "PhoneService": ["Yes"],

    "MultipleLines": ["No"],

    "InternetService": ["Fiber optic"],

    "OnlineSecurity": ["No"],

    "OnlineBackup": ["No"],

    "DeviceProtection": ["No"],

    "TechSupport": ["No"],

    "StreamingTV": ["Yes"],

    "StreamingMovies": ["Yes"],

    "Contract": ["Month-to-month"],

    "PaperlessBilling": ["Yes"],

    "PaymentMethod": [
        "Electronic check"
    ],

    "MonthlyCharges": [95.50],

    "TotalCharges": [477.50],

    "Average_Monthly_Spend": [95.50],

    "Service_Count": [3],

    "Customer_Lifetime_Months": [5],

    "High_Monthly_Charge": [1],

    "Short_Tenure": [1]
})


# ==========================================
# 4. Check Customer Data
# ==========================================

print("\nCustomer data:")
print(customer)

print("\nCustomer data types:")
print(customer.dtypes)


# ==========================================
# 5. Make Prediction
# ==========================================

churn_probability = model.predict_proba(
    customer
)[0][1]


# ==========================================
# 6. Risk Level
# ==========================================

if churn_probability >= 0.70:

    risk_level = "HIGH"

elif churn_probability >= 0.40:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ==========================================
# 7. Churn Prediction
# ==========================================

prediction = model.predict(customer)[0]

if prediction == 1:

    prediction_text = "Likely to Churn"

else:

    prediction_text = "Likely to Stay"


# ==========================================
# 8. Recommendation
# ==========================================

if risk_level == "HIGH":

    recommendation = (
        "Contact the customer proactively, "
        "offer a retention discount, and "
        "provide personalized support."
    )

elif risk_level == "MEDIUM":

    recommendation = (
        "Monitor the customer and provide "
        "a suitable loyalty offer."
    )

else:

    recommendation = (
        "Continue normal customer engagement "
        "and monitor future behavior."
    )


# ==========================================
# 9. Display Final Result
# ==========================================

print("\n")
print("======================================")
print("       CUSTOMER CHURN PREDICTION")
print("======================================")

print("Customer ID:", customer_id)

print(
    "Churn Probability:",
    round(churn_probability * 100, 2),
    "%"
)

print("Risk Level:", risk_level)

print("Prediction:", prediction_text)

print("Recommendation:")
print(recommendation)

print("======================================")