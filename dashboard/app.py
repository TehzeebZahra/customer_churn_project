import streamlit as st
import pandas as pd
import altair as alt
import joblib
import os
from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. COLOR THEME
# =========================================================
# One palette, reused everywhere so colors carry consistent meaning:
#   red    -> churned / high risk / factors that increase churn
#   green  -> stayed / low risk / factors that reduce churn
#   orange -> medium risk / warnings
#   blue   -> primary brand color / neutral metrics

COLORS = {
    "primary": "#4C6EF5",
    "primary_dark": "#364FC7",
    "danger": "#F03E3E",
    "warning": "#F59F00",
    "success": "#12B886",
    "neutral": "#495057",
    "bg": "#F8F9FA",
    "card": "#FFFFFF",
    "sidebar": "#1D3557",
    "sidebar_text": "#F1F3F5",
}

CHURN_COLORS = {"Stayed": COLORS["success"], "Churned": COLORS["danger"]}
RISK_COLORS = {"LOW": COLORS["success"], "MEDIUM": COLORS["warning"], "HIGH": COLORS["danger"]}

st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS['bg']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {COLORS['sidebar']};
    }}
    [data-testid="stSidebar"] * {{
        color: {COLORS['sidebar_text']} !important;
    }}
    [data-testid="stMetric"] {{
        background-color: {COLORS['card']};
        border: 1px solid #E9ECEF;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    [data-testid="stMetricValue"] {{
        color: {COLORS['primary_dark']};
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS['neutral']};
    }}
    h1, h2, h3 {{
        color: {COLORS['sidebar']};
    }}
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 8px;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['primary_dark']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CHART HELPERS (Altair, themed)
# =========================================================

def categorical_bar_chart(data: pd.Series, color_map: dict, y_title: str, sort_desc: bool = False):
    """Bar chart where each category gets a fixed, meaningful color."""
    chart_df = data.reset_index()
    chart_df.columns = ["category", "value"]

    order = list(color_map.keys())
    sort_order = sorted(order, key=lambda c: chart_df.loc[chart_df["category"] == c, "value"].sum(), reverse=True) if sort_desc else order

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=45)
        .encode(
            x=alt.X("category:N", title=None, sort=sort_order),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                legend=alt.Legend(title=None),
            ),
            tooltip=["category", "value"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def single_color_bar_chart(data: pd.Series, y_title: str, color: str = COLORS["primary"], sort_desc: bool = True):
    """Bar chart with one theme color, sorted so the worst/highest value stands out."""
    chart_df = data.reset_index()
    chart_df.columns = ["category", "value"]

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color, size=40)
        .encode(
            x=alt.X("category:N", title=None, sort="-y" if sort_desc else None),
            y=alt.Y("value:Q", title=y_title),
            tooltip=["category", "value"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def diverging_bar_chart(data: pd.DataFrame, label_col: str, value_col: str, y_title: str):
    """Bar chart for feature importance: red for positive (raises churn), green for negative (lowers churn)."""
    chart_df = data.copy()
    chart_df["direction"] = chart_df[value_col].apply(lambda v: "Raises churn risk" if v > 0 else "Lowers churn risk")

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{value_col}:Q", title=y_title),
            y=alt.Y(f"{label_col}:N", title=None, sort="-x"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Raises churn risk", "Lowers churn risk"], range=[COLORS["danger"], COLORS["success"]]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[label_col, value_col],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)


# =========================================================
# 4. PROJECT PATHS
# =========================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed", "cleaned_customer_churn.csv")
MODEL_PATH = os.path.join(PROJECT_PATH, "models", "churn_model.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(PROJECT_PATH, "reports", "feature_importance.csv")


# =========================================================
# 5. TITLE
# =========================================================

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Customer Churn Prediction & Business Intelligence System")


# =========================================================
# 6. LOAD DATASET
# =========================================================

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset file not found.")
    st.code(DATA_PATH)
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("❌ Could not load dataset.")
    st.exception(e)
    st.stop()


# =========================================================
# 7. LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.code(MODEL_PATH)
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Could not load model.")
    st.exception(e)
    st.stop()

st.success("✅ Dataset and model loaded successfully!")


# =========================================================
# 8. SIDEBAR
# =========================================================

st.sidebar.title("📌 Dashboard")
st.sidebar.write("Customer Churn Prediction System")
st.sidebar.markdown("---")
st.sidebar.write("Navigation")

page = st.sidebar.radio(
    "Select Section",
    ["Business Overview", "Churn Analysis", "Customer Prediction", "High-Risk Customers", "Model Explainability"]
)


# =========================================================
# 9. BUSINESS OVERVIEW
# =========================================================

if page == "Business Overview":
    st.header("📊 Business Overview")

    total_customers = len(df)
    churned_customers = (df["Churn"] == 1).sum() if "Churn" in df.columns else 0
    churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
    monthly_revenue = df["MonthlyCharges"].sum() if "MonthlyCharges" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Customers", total_customers)
    with col2:
        st.metric("🚨 Churned Customers", churned_customers)
    with col3:
        st.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
    with col4:
        st.metric("💰 Monthly Revenue", f"${monthly_revenue:,.2f}")

    st.subheader("📋 Customer Dataset")
    st.dataframe(df, use_container_width=True)


# =========================================================
# 10. CHURN ANALYSIS
# =========================================================

elif page == "Churn Analysis":
    st.header("📈 Churn Analysis")

    # Churn distribution
    st.subheader("Customer Churn Distribution")
    if "Churn" in df.columns:
        churn_distribution = df["Churn"].value_counts()
        churn_distribution.index = ["Stayed" if x == 0 else "Churned" for x in churn_distribution.index]
        categorical_bar_chart(churn_distribution, CHURN_COLORS, y_title="Customers")

    # Contract analysis
    st.subheader("📋 Churn Rate by Contract")
    if "Contract" in df.columns and "Churn" in df.columns:
        contract_churn = df.groupby("Contract")["Churn"].mean() * 100
        single_color_bar_chart(contract_churn, y_title="Churn Rate (%)", color=COLORS["primary"])

    # Tenure analysis
    st.subheader("📅 Churn Rate by Customer Tenure")
    if "tenure" in df.columns and "Churn" in df.columns:
        tenure_groups = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72, 100],
            labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months", "73+ Months"]
        )
        tenure_churn = df.groupby(tenure_groups, observed=True)["Churn"].mean() * 100
        single_color_bar_chart(tenure_churn, y_title="Churn Rate (%)", color=COLORS["primary_dark"], sort_desc=False)

    # Payment method
    st.subheader("💳 Churn Rate by Payment Method")
    if "PaymentMethod" in df.columns and "Churn" in df.columns:
        payment_churn = df.groupby("PaymentMethod")["Churn"].mean() * 100
        single_color_bar_chart(payment_churn, y_title="Churn Rate (%)", color=COLORS["warning"])

    # Monthly charges
    st.subheader("💰 Average Monthly Charges by Churn")
    if "MonthlyCharges" in df.columns and "Churn" in df.columns:
        charges_churn = df.groupby("Churn")["MonthlyCharges"].mean()
        charges_churn.index = ["Stayed" if x == 0 else "Churned" for x in charges_churn.index]
        categorical_bar_chart(charges_churn, CHURN_COLORS, y_title="Avg Monthly Charges ($)")


# =========================================================
# 11. CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":
    st.header("🔮 Individual Customer Churn Prediction")
    st.write("Enter customer information below.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=5)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=350.0)

    # Feature engineering
    average_monthly_spend = (total_charges / tenure) if tenure > 0 else monthly_charges
    customer_lifetime_months = tenure
    high_monthly_charge = 1 if monthly_charges > 70 else 0
    short_tenure = 1 if tenure < 12 else 0

    services = [phone_service, multiple_lines, online_security, online_backup,
                device_protection, tech_support, streaming_tv, streaming_movies]
    service_count = sum(1 for s in services if s not in ["No", "No internet service", "No phone service"])

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Average_Monthly_Spend": [average_monthly_spend],
        "Service_Count": [service_count],
        "Customer_Lifetime_Months": [customer_lifetime_months],
        "High_Monthly_Charge": [high_monthly_charge],
        "Short_Tenure": [short_tenure],
    })

    if st.button("🔮 Predict Churn", use_container_width=True):
        try:
            probability = model.predict_proba(customer)[0][1]
            prediction = model.predict(customer)[0]
            probability_percent = probability * 100

            if probability >= 0.70:
                risk = "HIGH"
            elif probability >= 0.40:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            result = "Likely to Churn" if prediction == 1 else "Likely to Stay"

            st.subheader("🎯 Prediction Result")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Probability", f"{probability_percent:.2f}%")
            with col2:
                st.metric("Risk Level", risk)
            with col3:
                st.metric("Prediction", result)

            st.subheader("💡 Recommended Action")
            if risk == "HIGH":
                st.error("HIGH RISK: Contact the customer proactively, offer a retention discount and provide personalized support.")
            elif risk == "MEDIUM":
                st.warning("MEDIUM RISK: Monitor this customer and consider offering a loyalty benefit.")
            else:
                st.success("LOW RISK: Continue normal customer engagement.")

            st.subheader("🔍 Why Is This Customer At Risk?")
            risk_factors = []
            if short_tenure == 1:
                risk_factors.append("🔴 Short customer tenure")
            if high_monthly_charge == 1:
                risk_factors.append("🔴 High monthly charges")
            if contract == "Month-to-month":
                risk_factors.append("🔴 Month-to-month contract")
            if tech_support == "No":
                risk_factors.append("🔴 No technical support")
            if online_security == "No":
                risk_factors.append("🔴 No online security")
            if payment_method == "Electronic check":
                risk_factors.append("🔴 Electronic check payment method")
            if service_count >= 5:
                risk_factors.append("🟢 Multiple services subscribed")
            if tenure >= 24:
                risk_factors.append("🟢 Long customer tenure")
            if contract == "Two year":
                risk_factors.append("🟢 Long-term contract")

            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("No major predefined risk factors detected.")

        except Exception as e:
            st.error("❌ Prediction Error")
            st.exception(e)


# =========================================================
# 12. HIGH-RISK CUSTOMERS
# =========================================================

elif page == "High-Risk Customers":
    st.header("🚨 High-Risk Customer Analysis")
    st.write("Customers with a high probability of churn.")

    try:
        prediction_data = df.copy()
        if "Churn" in prediction_data.columns:
            prediction_data = prediction_data.drop("Churn", axis=1)

        prediction_data["Average_Monthly_Spend"] = (
            prediction_data["TotalCharges"] / prediction_data["tenure"].replace(0, 1)
        )
        prediction_data["Customer_Lifetime_Months"] = prediction_data["tenure"]
        prediction_data["High_Monthly_Charge"] = (prediction_data["MonthlyCharges"] > 70).astype(int)
        prediction_data["Short_Tenure"] = (prediction_data["tenure"] < 12).astype(int)

        prediction_data["Service_Count"] = 0
        service_columns = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        for column in service_columns:
            if column in prediction_data.columns:
                prediction_data["Service_Count"] += (
                    ~prediction_data[column].isin(["No", "No internet service", "No phone service"])
                ).astype(int)

        required_columns = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
            "Average_Monthly_Spend", "Service_Count", "Customer_Lifetime_Months",
            "High_Monthly_Charge", "Short_Tenure"
        ]

        missing_columns = [c for c in required_columns if c not in prediction_data.columns]
        if missing_columns:
            st.error("❌ Required model columns are missing:")
            st.write(missing_columns)
            st.stop()

        prediction_data = prediction_data[required_columns]

        probabilities = model.predict_proba(prediction_data)[:, 1]

        risk_df = df.copy()
        risk_df["Churn_Probability"] = probabilities
        risk_df["Churn_Probability_Percent"] = probabilities * 100

        def get_risk(probability):
            if probability >= 0.70:
                return "HIGH"
            elif probability >= 0.40:
                return "MEDIUM"
            else:
                return "LOW"

        risk_df["Risk_Level"] = risk_df["Churn_Probability"].apply(get_risk)

        high_risk_df = risk_df[risk_df["Risk_Level"] == "HIGH"].copy()
        high_risk_count = len(high_risk_df)
        revenue_at_risk = high_risk_df["MonthlyCharges"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🚨 High-Risk Customers", high_risk_count)
        with col2:
            st.metric("💰 Monthly Revenue at Risk", f"${revenue_at_risk:,.2f}")

        st.subheader("Customers Requiring Immediate Attention")
        if high_risk_count > 0:
            display_columns = [c for c in [
                "Contract", "tenure", "MonthlyCharges", "TotalCharges",
                "Churn_Probability_Percent", "Risk_Level"
            ] if c in high_risk_df.columns]

            high_risk_display = high_risk_df[display_columns].sort_values(
                "Churn_Probability_Percent", ascending=False
            )
            st.dataframe(high_risk_display, use_container_width=True)
        else:
            st.success("🎉 No high-risk customers found.")

        st.subheader("📊 Customer Risk Distribution")
        risk_distribution = risk_df["Risk_Level"].value_counts()
        categorical_bar_chart(risk_distribution, RISK_COLORS, y_title="Customers", sort_desc=False)

    except Exception as e:
        st.error("❌ Could not generate high-risk customer analysis.")
        st.exception(e)


# =========================================================
# 13. MODEL EXPLAINABILITY
# =========================================================

elif page == "Model Explainability":
    st.header("🔍 Model Explainability")
    st.write("Understanding which features influence customer churn predictions.")

    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        st.warning("⚠️ Feature importance file not found.")
        st.write("Please run the `05_model_explainability.ipynb` notebook first.")
        st.stop()

    try:
        importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        top_features = importance_df.sort_values(by="Absolute_Importance", ascending=False).head(10)

        st.subheader("📊 Top 10 Churn Drivers")
        diverging_bar_chart(top_features, label_col="Feature", value_col="Importance", y_title="Importance")

        st.subheader("📋 Feature Importance Details")
        st.dataframe(top_features[["Feature", "Importance", "Absolute_Importance"]], use_container_width=True)

        st.subheader("🔴 Factors Associated With Higher Churn")
        positive = importance_df[importance_df["Importance"] > 0].sort_values(by="Importance", ascending=False).head(5)
        if len(positive) > 0:
            for _, row in positive.iterrows():
                st.write(f"🔴 **{row['Feature']}** — Importance: {row['Importance']:.4f}")
        else:
            st.info("No positive churn factors found.")

        st.subheader("🟢 Factors Associated With Lower Churn")
        negative = importance_df[importance_df["Importance"] < 0].sort_values(by="Importance", ascending=True).head(5)
        if len(negative) > 0:
            for _, row in negative.iterrows():
                st.write(f"🟢 **{row['Feature']}** — Importance: {row['Importance']:.4f}")
        else:
            st.info("No negative churn factors found.")

        st.subheader("💡 Business Recommendation")
        if len(positive) > 0:
            top_factor = positive.iloc[0]["Feature"]
            st.info(
                f"The model indicates that **{top_factor}** is one of the strongest factors "
                f"associated with higher churn. The business should monitor customers affected "
                f"by this factor and consider targeted retention strategies."
            )
        else:
            st.info("Review the feature importance results to identify suitable retention strategies.")

    except Exception as e:
        st.error("❌ Could not load feature importance.")
        st.exception(e)
import streamlit as st
import pandas as pd
import altair as alt
import joblib
import os
from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. COLOR THEME
# =========================================================
# One palette, reused everywhere so colors carry consistent meaning:
#   red    -> churned / high risk / factors that increase churn
#   green  -> stayed / low risk / factors that reduce churn
#   orange -> medium risk / warnings
#   blue   -> primary brand color / neutral metrics

# Viridis stops (dark purple -> blue -> teal -> green -> yellow)
VIRIDIS = {
    "v0": "#440154",  # deep purple  (darkest / calmest)
    "v1": "#472D7B",
    "v2": "#3B528B",
    "v3": "#2C728E",
    "v4": "#21908C",
    "v5": "#27AD81",
    "v6": "#5DC863",
    "v7": "#AADC32",
    "v8": "#FDE725",  # bright yellow (brightest / highest-alert)
}

COLORS = {
    "primary": VIRIDIS["v3"],
    "primary_dark": VIRIDIS["v0"],
    "danger": VIRIDIS["v8"],     # yellow reads as "hottest" on viridis, used for churn/high-risk
    "warning": VIRIDIS["v6"],
    "success": VIRIDIS["v1"],    # dark purple reads as "coolest"/safest on viridis
    "neutral": "#495057",
    "bg": "#F8F9FA",
    "card": "#FFFFFF",
    "sidebar": VIRIDIS["v0"],
    "sidebar_text": "#F5F3FA",
}

CHURN_COLORS = {"Stayed": COLORS["success"], "Churned": COLORS["danger"]}
RISK_COLORS = {"LOW": VIRIDIS["v1"], "MEDIUM": VIRIDIS["v4"], "HIGH": VIRIDIS["v8"]}
VIRIDIS_SCHEME = "viridis"  # Altair's built-in scheme name, used for continuous/quantitative bars

st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS['bg']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {COLORS['sidebar']};
    }}
    [data-testid="stSidebar"] * {{
        color: {COLORS['sidebar_text']} !important;
    }}
    [data-testid="stMetric"] {{
        background-color: {COLORS['card']};
        border: 1px solid #E9ECEF;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    [data-testid="stMetricValue"] {{
        color: {COLORS['primary_dark']};
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS['neutral']};
    }}
    h1, h2, h3 {{
        color: {COLORS['sidebar']};
    }}
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 8px;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['primary_dark']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CHART HELPERS (Altair, themed)
# =========================================================

def categorical_bar_chart(data: pd.Series, color_map: dict, y_title: str, sort_desc: bool = False):
    """Bar chart where each category gets a fixed, meaningful color."""
    chart_df = data.reset_index()
    chart_df.columns = ["category", "value"]

    order = list(color_map.keys())
    sort_order = sorted(order, key=lambda c: chart_df.loc[chart_df["category"] == c, "value"].sum(), reverse=True) if sort_desc else order

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=45)
        .encode(
            x=alt.X("category:N", title=None, sort=sort_order),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                legend=alt.Legend(title=None),
            ),
            tooltip=["category", "value"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def single_color_bar_chart(data: pd.Series, y_title: str, sort_desc: bool = True):
    """Bar chart colored on a viridis gradient by value, so higher bars read as 'hotter'."""
    chart_df = data.reset_index()
    chart_df.columns = ["category", "value"]

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=40)
        .encode(
            x=alt.X("category:N", title=None, sort="-y" if sort_desc else None),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("value:Q", scale=alt.Scale(scheme=VIRIDIS_SCHEME), legend=None),
            tooltip=["category", "value"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def diverging_bar_chart(data: pd.DataFrame, label_col: str, value_col: str, y_title: str):
    """Bar chart for feature importance, colored on a viridis gradient by importance value."""
    chart_df = data.copy()

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{value_col}:Q", title=y_title),
            y=alt.Y(f"{label_col}:N", title=None, sort="-x"),
            color=alt.Color(f"{value_col}:Q", scale=alt.Scale(scheme=VIRIDIS_SCHEME), legend=None),
            tooltip=[label_col, value_col],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)


# =========================================================
# 4. PROJECT PATHS
# =========================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed", "cleaned_customer_churn.csv")
MODEL_PATH = os.path.join(PROJECT_PATH, "models", "churn_model.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(PROJECT_PATH, "reports", "feature_importance.csv")


# =========================================================
# 5. TITLE
# =========================================================

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Customer Churn Prediction & Business Intelligence System")


# =========================================================
# 6. LOAD DATASET
# =========================================================

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset file not found.")
    st.code(DATA_PATH)
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("❌ Could not load dataset.")
    st.exception(e)
    st.stop()


# =========================================================
# 7. LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.code(MODEL_PATH)
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Could not load model.")
    st.exception(e)
    st.stop()

st.success("✅ Dataset and model loaded successfully!")


# =========================================================
# 8. SIDEBAR
# =========================================================

st.sidebar.title("📌 Dashboard")
st.sidebar.write("Customer Churn Prediction System")
st.sidebar.markdown("---")
st.sidebar.write("Navigation")

page = st.sidebar.radio(
    "Select Section",
    ["Business Overview", "Churn Analysis", "Customer Prediction", "High-Risk Customers", "Model Explainability"]
)


# =========================================================
# 9. BUSINESS OVERVIEW
# =========================================================

if page == "Business Overview":
    st.header("📊 Business Overview")

    total_customers = len(df)
    churned_customers = (df["Churn"] == 1).sum() if "Churn" in df.columns else 0
    churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
    monthly_revenue = df["MonthlyCharges"].sum() if "MonthlyCharges" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Customers", total_customers)
    with col2:
        st.metric("🚨 Churned Customers", churned_customers)
    with col3:
        st.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
    with col4:
        st.metric("💰 Monthly Revenue", f"${monthly_revenue:,.2f}")

    st.subheader("📋 Customer Dataset")
    st.dataframe(df, use_container_width=True)


# =========================================================
# 10. CHURN ANALYSIS
# =========================================================

elif page == "Churn Analysis":
    st.header("📈 Churn Analysis")

    # Churn distribution
    st.subheader("Customer Churn Distribution")
    if "Churn" in df.columns:
        churn_distribution = df["Churn"].value_counts()
        churn_distribution.index = ["Stayed" if x == 0 else "Churned" for x in churn_distribution.index]
        categorical_bar_chart(churn_distribution, CHURN_COLORS, y_title="Customers")

    # Contract analysis
    st.subheader("📋 Churn Rate by Contract")
    if "Contract" in df.columns and "Churn" in df.columns:
        contract_churn = df.groupby("Contract")["Churn"].mean() * 100
        single_color_bar_chart(contract_churn, y_title="Churn Rate (%)")

    # Tenure analysis
    st.subheader("📅 Churn Rate by Customer Tenure")
    if "tenure" in df.columns and "Churn" in df.columns:
        tenure_groups = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72, 100],
            labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months", "73+ Months"]
        )
        tenure_churn = df.groupby(tenure_groups, observed=True)["Churn"].mean() * 100
        single_color_bar_chart(tenure_churn, y_title="Churn Rate (%)", sort_desc=False)

    # Payment method
    st.subheader("💳 Churn Rate by Payment Method")
    if "PaymentMethod" in df.columns and "Churn" in df.columns:
        payment_churn = df.groupby("PaymentMethod")["Churn"].mean() * 100
        single_color_bar_chart(payment_churn, y_title="Churn Rate (%)")

    # Monthly charges
    st.subheader("💰 Average Monthly Charges by Churn")
    if "MonthlyCharges" in df.columns and "Churn" in df.columns:
        charges_churn = df.groupby("Churn")["MonthlyCharges"].mean()
        charges_churn.index = ["Stayed" if x == 0 else "Churned" for x in charges_churn.index]
        categorical_bar_chart(charges_churn, CHURN_COLORS, y_title="Avg Monthly Charges ($)")


# =========================================================
# 11. CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":
    st.header("🔮 Individual Customer Churn Prediction")
    st.write("Enter customer information below.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=5)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=350.0)

    # Feature engineering
    average_monthly_spend = (total_charges / tenure) if tenure > 0 else monthly_charges
    customer_lifetime_months = tenure
    high_monthly_charge = 1 if monthly_charges > 70 else 0
    short_tenure = 1 if tenure < 12 else 0

    services = [phone_service, multiple_lines, online_security, online_backup,
                device_protection, tech_support, streaming_tv, streaming_movies]
    service_count = sum(1 for s in services if s not in ["No", "No internet service", "No phone service"])

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Average_Monthly_Spend": [average_monthly_spend],
        "Service_Count": [service_count],
        "Customer_Lifetime_Months": [customer_lifetime_months],
        "High_Monthly_Charge": [high_monthly_charge],
        "Short_Tenure": [short_tenure],
    })

    if st.button("🔮 Predict Churn", use_container_width=True):
        try:
            probability = model.predict_proba(customer)[0][1]
            prediction = model.predict(customer)[0]
            probability_percent = probability * 100

            if probability >= 0.70:
                risk = "HIGH"
            elif probability >= 0.40:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            result = "Likely to Churn" if prediction == 1 else "Likely to Stay"

            st.subheader("🎯 Prediction Result")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Probability", f"{probability_percent:.2f}%")
            with col2:
                st.metric("Risk Level", risk)
            with col3:
                st.metric("Prediction", result)

            st.subheader("💡 Recommended Action")
            if risk == "HIGH":
                st.error("HIGH RISK: Contact the customer proactively, offer a retention discount and provide personalized support.")
            elif risk == "MEDIUM":
                st.warning("MEDIUM RISK: Monitor this customer and consider offering a loyalty benefit.")
            else:
                st.success("LOW RISK: Continue normal customer engagement.")

            st.subheader("🔍 Why Is This Customer At Risk?")
            risk_factors = []
            if short_tenure == 1:
                risk_factors.append("🔴 Short customer tenure")
            if high_monthly_charge == 1:
                risk_factors.append("🔴 High monthly charges")
            if contract == "Month-to-month":
                risk_factors.append("🔴 Month-to-month contract")
            if tech_support == "No":
                risk_factors.append("🔴 No technical support")
            if online_security == "No":
                risk_factors.append("🔴 No online security")
            if payment_method == "Electronic check":
                risk_factors.append("🔴 Electronic check payment method")
            if service_count >= 5:
                risk_factors.append("🟢 Multiple services subscribed")
            if tenure >= 24:
                risk_factors.append("🟢 Long customer tenure")
            if contract == "Two year":
                risk_factors.append("🟢 Long-term contract")

            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("No major predefined risk factors detected.")

        except Exception as e:
            st.error("❌ Prediction Error")
            st.exception(e)


# =========================================================
# 12. HIGH-RISK CUSTOMERS
# =========================================================

elif page == "High-Risk Customers":
    st.header("🚨 High-Risk Customer Analysis")
    st.write("Customers with a high probability of churn.")

    try:
        prediction_data = df.copy()
        if "Churn" in prediction_data.columns:
            prediction_data = prediction_data.drop("Churn", axis=1)

        prediction_data["Average_Monthly_Spend"] = (
            prediction_data["TotalCharges"] / prediction_data["tenure"].replace(0, 1)
        )
        prediction_data["Customer_Lifetime_Months"] = prediction_data["tenure"]
        prediction_data["High_Monthly_Charge"] = (prediction_data["MonthlyCharges"] > 70).astype(int)
        prediction_data["Short_Tenure"] = (prediction_data["tenure"] < 12).astype(int)

        prediction_data["Service_Count"] = 0
        service_columns = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        for column in service_columns:
            if column in prediction_data.columns:
                prediction_data["Service_Count"] += (
                    ~prediction_data[column].isin(["No", "No internet service", "No phone service"])
                ).astype(int)

        required_columns = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
            "Average_Monthly_Spend", "Service_Count", "Customer_Lifetime_Months",
            "High_Monthly_Charge", "Short_Tenure"
        ]

        missing_columns = [c for c in required_columns if c not in prediction_data.columns]
        if missing_columns:
            st.error("❌ Required model columns are missing:")
            st.write(missing_columns)
            st.stop()

        prediction_data = prediction_data[required_columns]

        probabilities = model.predict_proba(prediction_data)[:, 1]

        risk_df = df.copy()
        risk_df["Churn_Probability"] = probabilities
        risk_df["Churn_Probability_Percent"] = probabilities * 100

        def get_risk(probability):
            if probability >= 0.70:
                return "HIGH"
            elif probability >= 0.40:
                return "MEDIUM"
            else:
                return "LOW"

        risk_df["Risk_Level"] = risk_df["Churn_Probability"].apply(get_risk)

        high_risk_df = risk_df[risk_df["Risk_Level"] == "HIGH"].copy()
        high_risk_count = len(high_risk_df)
        revenue_at_risk = high_risk_df["MonthlyCharges"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🚨 High-Risk Customers", high_risk_count)
        with col2:
            st.metric("💰 Monthly Revenue at Risk", f"${revenue_at_risk:,.2f}")

        st.subheader("Customers Requiring Immediate Attention")
        if high_risk_count > 0:
            display_columns = [c for c in [
                "Contract", "tenure", "MonthlyCharges", "TotalCharges",
                "Churn_Probability_Percent", "Risk_Level"
            ] if c in high_risk_df.columns]

            high_risk_display = high_risk_df[display_columns].sort_values(
                "Churn_Probability_Percent", ascending=False
            )
            st.dataframe(high_risk_display, use_container_width=True)
        else:
            st.success("🎉 No high-risk customers found.")

        st.subheader("📊 Customer Risk Distribution")
        risk_distribution = risk_df["Risk_Level"].value_counts()
        categorical_bar_chart(risk_distribution, RISK_COLORS, y_title="Customers", sort_desc=False)

    except Exception as e:
        st.error("❌ Could not generate high-risk customer analysis.")
        st.exception(e)


# =========================================================
# 13. MODEL EXPLAINABILITY
# =========================================================

elif page == "Model Explainability":
    st.header("🔍 Model Explainability")
    st.write("Understanding which features influence customer churn predictions.")

    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        st.warning("⚠️ Feature importance file not found.")
        st.write("Please run the `05_model_explainability.ipynb` notebook first.")
        st.stop()

    try:
        importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        top_features = importance_df.sort_values(by="Absolute_Importance", ascending=False).head(10)

        st.subheader("📊 Top 10 Churn Drivers")
        diverging_bar_chart(top_features, label_col="Feature", value_col="Importance", y_title="Importance")

        st.subheader("📋 Feature Importance Details")
        st.dataframe(top_features[["Feature", "Importance", "Absolute_Importance"]], use_container_width=True)

        st.subheader("🔴 Factors Associated With Higher Churn")
        positive = importance_df[importance_df["Importance"] > 0].sort_values(by="Importance", ascending=False).head(5)
        if len(positive) > 0:
            for _, row in positive.iterrows():
                st.write(f"🔴 **{row['Feature']}** — Importance: {row['Importance']:.4f}")
        else:
            st.info("No positive churn factors found.")

        st.subheader("🟢 Factors Associated With Lower Churn")
        negative = importance_df[importance_df["Importance"] < 0].sort_values(by="Importance", ascending=True).head(5)
        if len(negative) > 0:
            for _, row in negative.iterrows():
                st.write(f"🟢 **{row['Feature']}** — Importance: {row['Importance']:.4f}")
        else:
            st.info("No negative churn factors found.")

        st.subheader("💡 Business Recommendation")
        if len(positive) > 0:
            top_factor = positive.iloc[0]["Feature"]
            st.info(
                f"The model indicates that **{top_factor}** is one of the strongest factors "
                f"associated with higher churn. The business should monitor customers affected "
                f"by this factor and consider targeted retention strategies."
            )
        else:
            st.info("Review the feature importance results to identify suitable retention strategies.")

    except Exception as e:
        st.error("❌ Could not load feature importance.")
        st.exception(e)




# =========================================================
# 14. FOOTER
# =========================================================

st.markdown("---")
st.write("Customer Churn Prediction & Business Intelligence System")
st.write("Built using Python, Pandas, Scikit-learn and Streamlit")