import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. PROJECT PATH
# =========================================================

PROJECT_PATH = r"C:\Users\AA\Desktop\Internship\customer_churn_project"

DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "processed",
    "cleaned_customer_churn.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "churn_model.pkl"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    PROJECT_PATH,
    "reports",
    "feature_importance.csv"
)


# =========================================================
# 3. TITLE
# =========================================================

st.title("📊 Customer Churn Prediction Dashboard")

st.write(
    "Customer Churn Prediction & Business Intelligence System"
)


# =========================================================
# 4. LOAD DATASET
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
# 5. LOAD MODEL
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


st.success(
    "✅ Dataset and model loaded successfully!"
)


# =========================================================
# 6. SIDEBAR
# =========================================================

st.sidebar.title("📌 Dashboard")

st.sidebar.write(
    "Customer Churn Prediction System"
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Navigation"
)

page = st.sidebar.radio(
    "Select Section",
    [
        "Business Overview",
        "Churn Analysis",
        "Customer Prediction",
        "High-Risk Customers",
        "Model Explainability"
    ]
)


# =========================================================
# 7. BUSINESS OVERVIEW
# =========================================================

if page == "Business Overview":

    st.header("📊 Business Overview")


    total_customers = len(df)


    if "Churn" in df.columns:

        churned_customers = (
            df["Churn"] == 1
        ).sum()

    else:

        churned_customers = 0


    if total_customers > 0:

        churn_rate = (
            churned_customers
            / total_customers
            * 100
        )

    else:

        churn_rate = 0


    if "MonthlyCharges" in df.columns:

        monthly_revenue = (
            df["MonthlyCharges"].sum()
        )

    else:

        monthly_revenue = 0


    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "👥 Total Customers",
            total_customers
        )


    with col2:

        st.metric(
            "🚨 Churned Customers",
            churned_customers
        )


    with col3:

        st.metric(
            "📉 Churn Rate",
            f"{churn_rate:.2f}%"
        )


    with col4:

        st.metric(
            "💰 Monthly Revenue",
            f"${monthly_revenue:,.2f}"
        )


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader(
        "📋 Customer Dataset"
    )


    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# 8. CHURN ANALYSIS
# =========================================================

elif page == "Churn Analysis":

    st.header("📈 Churn Analysis")


    # =====================================================
    # CHURN DISTRIBUTION
    # =====================================================

    st.subheader(
        "Customer Churn Distribution"
    )


    if "Churn" in df.columns:

        churn_distribution = (
            df["Churn"].value_counts()
        )

        churn_distribution.index = [
            "Stayed" if x == 0 else "Churned"
            for x in churn_distribution.index
        ]

        st.bar_chart(
            churn_distribution
        )


    # =====================================================
    # CONTRACT ANALYSIS
    # =====================================================

    st.subheader(
        "📋 Churn Rate by Contract"
    )


    if (
        "Contract" in df.columns
        and "Churn" in df.columns
    ):

        contract_churn = (
            df.groupby(
                "Contract"
            )["Churn"]
            .mean()
            * 100
        )

        st.bar_chart(
            contract_churn
        )


    # =====================================================
    # TENURE ANALYSIS
    # =====================================================

    st.subheader(
        "📅 Churn Rate by Customer Tenure"
    )


    if (
        "tenure" in df.columns
        and "Churn" in df.columns
    ):

        tenure_groups = pd.cut(
            df["tenure"],
            bins=[
                -1,
                12,
                24,
                48,
                72,
                100
            ],
            labels=[
                "0-12 Months",
                "13-24 Months",
                "25-48 Months",
                "49-72 Months",
                "73+ Months"
            ]
        )


        tenure_churn = (
            df.groupby(
                tenure_groups,
                observed=True
            )["Churn"]
            .mean()
            * 100
        )


        st.bar_chart(
            tenure_churn
        )


    # =====================================================
    # PAYMENT METHOD
    # =====================================================

    st.subheader(
        "💳 Churn Rate by Payment Method"
    )


    if (
        "PaymentMethod" in df.columns
        and "Churn" in df.columns
    ):

        payment_churn = (
            df.groupby(
                "PaymentMethod"
            )["Churn"]
            .mean()
            * 100
        )

        st.bar_chart(
            payment_churn
        )


    # =====================================================
    # MONTHLY CHARGES
    # =====================================================

    st.subheader(
        "💰 Average Monthly Charges by Churn"
    )


    if (
        "MonthlyCharges" in df.columns
        and "Churn" in df.columns
    ):

        charges_churn = (
            df.groupby(
                "Churn"
            )["MonthlyCharges"]
            .mean()
        )


        charges_churn.index = [
            "Stayed" if x == 0 else "Churned"
            for x in charges_churn.index
        ]


        st.bar_chart(
            charges_churn
        )


# =========================================================
# 9. CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":

    st.header(
        "🔮 Individual Customer Churn Prediction"
    )


    st.write(
        "Enter customer information below."
    )


    # =====================================================
    # INPUTS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    # =====================================================
    # COLUMN 1
    # =====================================================

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )


        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )


        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )


        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )


        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=5
        )


        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )


        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )


    # =====================================================
    # COLUMN 2
    # =====================================================

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )


        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    # =====================================================
    # COLUMN 3
    # =====================================================

    with col3:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )


        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )


        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )


        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=350.0
        )


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================


    # Average Monthly Spend

    if tenure > 0:

        average_monthly_spend = (
            total_charges / tenure
        )

    else:

        average_monthly_spend = (
            monthly_charges
        )


    # Customer Lifetime

    customer_lifetime_months = tenure


    # High Monthly Charge

    if monthly_charges > 70:

        high_monthly_charge = 1

    else:

        high_monthly_charge = 0


    # Short Tenure

    if tenure < 12:

        short_tenure = 1

    else:

        short_tenure = 0


    # Service Count

    service_count = 0


    services = [

        phone_service,

        multiple_lines,

        online_security,

        online_backup,

        device_protection,

        tech_support,

        streaming_tv,

        streaming_movies
    ]


    for service in services:

        if service not in [
            "No",
            "No internet service",
            "No phone service"
        ]:

            service_count += 1


    # =====================================================
    # CUSTOMER DATAFRAME
    # =====================================================

    customer = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [
            senior_citizen
        ],

        "Partner": [
            partner
        ],

        "Dependents": [
            dependents
        ],

        "tenure": [
            tenure
        ],

        "PhoneService": [
            phone_service
        ],

        "MultipleLines": [
            multiple_lines
        ],

        "InternetService": [
            internet_service
        ],

        "OnlineSecurity": [
            online_security
        ],

        "OnlineBackup": [
            online_backup
        ],

        "DeviceProtection": [
            device_protection
        ],

        "TechSupport": [
            tech_support
        ],

        "StreamingTV": [
            streaming_tv
        ],

        "StreamingMovies": [
            streaming_movies
        ],

        "Contract": [
            contract
        ],

        "PaperlessBilling": [
            paperless_billing
        ],

        "PaymentMethod": [
            payment_method
        ],

        "MonthlyCharges": [
            monthly_charges
        ],

        "TotalCharges": [
            total_charges
        ],

        "Average_Monthly_Spend": [
            average_monthly_spend
        ],

        "Service_Count": [
            service_count
        ],

        "Customer_Lifetime_Months": [
            customer_lifetime_months
        ],

        "High_Monthly_Charge": [
            high_monthly_charge
        ],

        "Short_Tenure": [
            short_tenure
        ]

    })


    # =====================================================
    # PREDICT
    # =====================================================

    if st.button(
        "🔮 Predict Churn",
        use_container_width=True
    ):

        try:

            probability = (
                model.predict_proba(
                    customer
                )[0][1]
            )


            prediction = (
                model.predict(
                    customer
                )[0]
            )


            probability_percent = (
                probability * 100
            )


            # =================================================
            # RISK LEVEL
            # =================================================

            if probability >= 0.70:

                risk = "HIGH"

            elif probability >= 0.40:

                risk = "MEDIUM"

            else:

                risk = "LOW"


            # =================================================
            # PREDICTION
            # =================================================

            if prediction == 1:

                result = "Likely to Churn"

            else:

                result = "Likely to Stay"


            # =================================================
            # RESULT
            # =================================================

            st.subheader(
                "🎯 Prediction Result"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Churn Probability",
                    f"{probability_percent:.2f}%"
                )


            with col2:

                st.metric(
                    "Risk Level",
                    risk
                )


            with col3:

                st.metric(
                    "Prediction",
                    result
                )


            # =================================================
            # RECOMMENDATION
            # =================================================

            st.subheader(
                "💡 Recommended Action"
            )


            if risk == "HIGH":

                st.error(
                    "HIGH RISK: Contact the customer "
                    "proactively, offer a retention "
                    "discount and provide personalized "
                    "support."
                )


            elif risk == "MEDIUM":

                st.warning(
                    "MEDIUM RISK: Monitor this customer "
                    "and consider offering a loyalty "
                    "benefit."
                )


            else:

                st.success(
                    "LOW RISK: Continue normal customer "
                    "engagement."
                )


            # =================================================
            # CUSTOMER RISK FACTORS
            # =================================================

            st.subheader(
                "🔍 Why Is This Customer At Risk?"
            )


            risk_factors = []


            if short_tenure == 1:

                risk_factors.append(
                    "🔴 Short customer tenure"
                )


            if high_monthly_charge == 1:

                risk_factors.append(
                    "🔴 High monthly charges"
                )


            if contract == "Month-to-month":

                risk_factors.append(
                    "🔴 Month-to-month contract"
                )


            if tech_support == "No":

                risk_factors.append(
                    "🔴 No technical support"
                )


            if online_security == "No":

                risk_factors.append(
                    "🔴 No online security"
                )


            if payment_method == "Electronic check":

                risk_factors.append(
                    "🔴 Electronic check payment method"
                )


            if service_count >= 5:

                risk_factors.append(
                    "🟢 Multiple services subscribed"
                )


            if tenure >= 24:

                risk_factors.append(
                    "🟢 Long customer tenure"
                )


            if contract == "Two year":

                risk_factors.append(
                    "🟢 Long-term contract"
                )


            if len(risk_factors) > 0:

                for factor in risk_factors:

                    st.write(factor)

            else:

                st.success(
                    "No major predefined risk factors detected."
                )


        except Exception as e:

            st.error(
                "❌ Prediction Error"
            )

            st.exception(e)


# =========================================================
# 10. HIGH-RISK CUSTOMERS
# =========================================================

elif page == "High-Risk Customers":

    st.header(
        "🚨 High-Risk Customer Analysis"
    )


    st.write(
        "Customers with a high probability of churn."
    )


    try:

        # =================================================
        # COPY DATA
        # =================================================

        prediction_data = df.copy()


        # Remove target

        if "Churn" in prediction_data.columns:

            prediction_data = prediction_data.drop(
                "Churn",
                axis=1
            )


        # =================================================
        # FEATURE ENGINEERING
        # =================================================


        # Average Monthly Spend

        prediction_data[
            "Average_Monthly_Spend"
        ] = (

            prediction_data[
                "TotalCharges"
            ]

            /

            prediction_data[
                "tenure"
            ].replace(
                0,
                1
            )
        )


        # Customer Lifetime

        prediction_data[
            "Customer_Lifetime_Months"
        ] = prediction_data[
            "tenure"
        ]


        # High Monthly Charge

        prediction_data[
            "High_Monthly_Charge"
        ] = (

            prediction_data[
                "MonthlyCharges"
            ] > 70

        ).astype(int)


        # Short Tenure

        prediction_data[
            "Short_Tenure"
        ] = (

            prediction_data[
                "tenure"
            ] < 12

        ).astype(int)


        # Service Count

        prediction_data[
            "Service_Count"
        ] = 0


        service_columns = [

            "PhoneService",

            "MultipleLines",

            "OnlineSecurity",

            "OnlineBackup",

            "DeviceProtection",

            "TechSupport",

            "StreamingTV",

            "StreamingMovies"
        ]


        for column in service_columns:

            if column in prediction_data.columns:

                prediction_data[
                    "Service_Count"
                ] += (

                    ~prediction_data[
                        column
                    ].isin([
                        "No",
                        "No internet service",
                        "No phone service"
                    ])

                ).astype(int)


        # =================================================
        # REQUIRED MODEL FEATURES
        # =================================================

        required_columns = [

            "gender",

            "SeniorCitizen",

            "Partner",

            "Dependents",

            "tenure",

            "PhoneService",

            "MultipleLines",

            "InternetService",

            "OnlineSecurity",

            "OnlineBackup",

            "DeviceProtection",

            "TechSupport",

            "StreamingTV",

            "StreamingMovies",

            "Contract",

            "PaperlessBilling",

            "PaymentMethod",

            "MonthlyCharges",

            "TotalCharges",

            "Average_Monthly_Spend",

            "Service_Count",

            "Customer_Lifetime_Months",

            "High_Monthly_Charge",

            "Short_Tenure"
        ]


        # =================================================
        # CHECK COLUMNS
        # =================================================

        missing_columns = [

            column

            for column in required_columns

            if column not in prediction_data.columns
        ]


        if len(missing_columns) > 0:

            st.error(
                "❌ Required model columns are missing:"
            )

            st.write(
                missing_columns
            )

            st.stop()


        # Keep exact columns

        prediction_data = prediction_data[
            required_columns
        ]


        # =================================================
        # PREDICTION
        # =================================================

        probabilities = (
            model.predict_proba(
                prediction_data
            )[:, 1]
        )


        # =================================================
        # CREATE RISK DATAFRAME
        # =================================================

        risk_df = df.copy()


        risk_df[
            "Churn_Probability"
        ] = probabilities


        risk_df[
            "Churn_Probability_Percent"
        ] = (
            probabilities * 100
        )


        # =================================================
        # RISK LEVEL
        # =================================================

        def get_risk(probability):

            if probability >= 0.70:

                return "HIGH"

            elif probability >= 0.40:

                return "MEDIUM"

            else:

                return "LOW"


        risk_df[
            "Risk_Level"
        ] = risk_df[
            "Churn_Probability"
        ].apply(
            get_risk
        )


        # =================================================
        # HIGH-RISK CUSTOMERS
        # =================================================

        high_risk_df = risk_df[
            risk_df[
                "Risk_Level"
            ] == "HIGH"
        ].copy()


        high_risk_count = len(
            high_risk_df
        )


        # =================================================
        # REVENUE AT RISK
        # =================================================

        revenue_at_risk = (

            high_risk_df[
                "MonthlyCharges"
            ].sum()

        )


        # =================================================
        # KPI
        # =================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "🚨 High-Risk Customers",
                high_risk_count
            )


        with col2:

            st.metric(
                "💰 Monthly Revenue at Risk",
                f"${revenue_at_risk:,.2f}"
            )


        # =================================================
        # HIGH RISK TABLE
        # =================================================

        st.subheader(
            "Customers Requiring Immediate Attention"
        )


        if high_risk_count > 0:

            display_columns = []


            for column in [

                "Contract",

                "tenure",

                "MonthlyCharges",

                "TotalCharges",

                "Churn_Probability_Percent",

                "Risk_Level"

            ]:

                if column in high_risk_df.columns:

                    display_columns.append(
                        column
                    )


            high_risk_display = (

                high_risk_df[
                    display_columns
                ]

                .sort_values(
                    "Churn_Probability_Percent",
                    ascending=False
                )

            )


            st.dataframe(
                high_risk_display,
                use_container_width=True
            )


        else:

            st.success(
                "🎉 No high-risk customers found."
            )


        # =================================================
        # RISK DISTRIBUTION
        # =================================================

        st.subheader(
            "📊 Customer Risk Distribution"
        )


        risk_distribution = (
            risk_df[
                "Risk_Level"
            ].value_counts()
        )


        st.bar_chart(
            risk_distribution
        )


    except Exception as e:

        st.error(
            "❌ Could not generate high-risk customer analysis."
        )

        st.exception(e)


# =========================================================
# 11. MODEL EXPLAINABILITY
# =========================================================

elif page == "Model Explainability":

    st.header(
        "🔍 Model Explainability"
    )


    st.write(
        "Understanding which features influence "
        "customer churn predictions."
    )


    # =====================================================
    # CHECK FEATURE IMPORTANCE FILE
    # =====================================================

    if not os.path.exists(
        FEATURE_IMPORTANCE_PATH
    ):

        st.warning(
            "⚠️ Feature importance file not found."
        )

        st.write(
            "Please run the "
            "`05_model_explainability.ipynb` notebook first."
        )

        st.stop()


    try:

        importance_df = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )


        # =================================================
        # TOP 10 FEATURES
        # =================================================

        top_features = (

            importance_df

            .sort_values(
                by="Absolute_Importance",
                ascending=False
            )

            .head(10)

        )


        # =================================================
        # CHART
        # =================================================

        st.subheader(
            "📊 Top 10 Churn Drivers"
        )


        chart_data = (

            top_features[
                [
                    "Feature",
                    "Importance"
                ]
            ]

            .set_index(
                "Feature"
            )

        )


        st.bar_chart(
            chart_data
        )


        # =================================================
        # TABLE
        # =================================================

        st.subheader(
            "📋 Feature Importance Details"
        )


        st.dataframe(
            top_features[
                [
                    "Feature",
                    "Importance",
                    "Absolute_Importance"
                ]
            ],
            use_container_width=True
        )


        # =================================================
        # POSITIVE FACTORS
        # =================================================

        st.subheader(
            "🔴 Factors Associated With Higher Churn"
        )


        positive = (

            importance_df[
                importance_df[
                    "Importance"
                ] > 0
            ]

            .sort_values(
                by="Importance",
                ascending=False
            )

            .head(5)

        )


        if len(positive) > 0:

            for _, row in positive.iterrows():

                st.write(
                    f"🔴 **{row['Feature']}** — "
                    f"Importance: "
                    f"{row['Importance']:.4f}"
                )

        else:

            st.info(
                "No positive churn factors found."
            )


        # =================================================
        # NEGATIVE FACTORS
        # =================================================

        st.subheader(
            "🟢 Factors Associated With Lower Churn"
        )


        negative = (

            importance_df[
                importance_df[
                    "Importance"
                ] < 0
            ]

            .sort_values(
                by="Importance",
                ascending=True
            )

            .head(5)

        )


        if len(negative) > 0:

            for _, row in negative.iterrows():

                st.write(
                    f"🟢 **{row['Feature']}** — "
                    f"Importance: "
                    f"{row['Importance']:.4f}"
                )

        else:

            st.info(
                "No negative churn factors found."
            )


        # =================================================
        # BUSINESS RECOMMENDATION
        # =================================================

        st.subheader(
            "💡 Business Recommendation"
        )


        if len(positive) > 0:

            top_factor = positive.iloc[0][
                "Feature"
            ]


            st.info(
                f"The model indicates that "
                f"**{top_factor}** is one of the "
                f"strongest factors associated with "
                f"higher churn. The business should "
                f"monitor customers affected by this "
                f"factor and consider targeted "
                f"retention strategies."
            )

        else:

            st.info(
                "Review the feature importance results "
                "to identify suitable retention strategies."
            )


    except Exception as e:

        st.error(
            "❌ Could not load feature importance."
        )

        st.exception(e)


# =========================================================
# 12. FOOTER
# =========================================================

st.markdown("---")

st.write(
    "Customer Churn Prediction & Business Intelligence System"
)

st.write(
    "Built using Python, Pandas, Scikit-learn and Streamlit"
)