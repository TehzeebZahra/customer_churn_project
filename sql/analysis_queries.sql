---1.  Total Customers ---
SELECT COUNT(*) AS Total_Customers
FROM customers;

---2.  Churned Customers ---
SELECT COUNT(*) AS Churned_Customers
FROM customers
WHERE Churn = 1;

---3.  Customers Who Stayed --- 
SELECT COUNT(*) AS Stayed_Customers
FROM customers
WHERE Churn = 0;

---4.  Churn Rate ---
SELECT 
    ROUND(
        SUM(Churn) * 100.0 / COUNT(*),
        2
    ) AS Churn_Rate_Percentage
FROM customers;

---5.  Churn by Contract ---
SELECT 
    Contract,
    COUNT(*) AS Total_Customers,
    SUM(Churn) AS Churned_Customers,
    ROUND(
        SUM(Churn) * 100.0 / COUNT(*),
        2
    ) AS Churn_Rate
FROM customers
GROUP BY Contract
ORDER BY Churn_Rate DESC; 

---6.  Churn by Internet Service ---
SELECT 
    InternetService,
    COUNT(*) AS Total_Customers,
    SUM(Churn) AS Churned_Customers,
    ROUND(
        SUM(Churn) * 100.0 / COUNT(*),
        2
    ) AS Churn_Rate
FROM customers
GROUP BY InternetService
ORDER BY Churn_Rate DESC;

---7.  Average Monthly Charges: Churn vs No Churn ---
SELECT 
    Churn,
    ROUND(AVG(MonthlyCharges), 2) AS Average_Monthly_Charges
FROM customers
GROUP BY Churn;

---8.  High Monthly Charges Customers ---
SELECT 
    gender,
    tenure,
    Contract,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM customers
WHERE MonthlyCharges > 80
ORDER BY MonthlyCharges DESC
LIMIT 20;

---9.  Short Tenure + Churn --- 
SELECT 
    tenure,
    Contract,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM customers
WHERE tenure <= 6
AND Churn = 1
ORDER BY tenure ASC;

---10.  Payment Method ke according Churn --- 
SELECT 
    PaymentMethod,
    COUNT(*) AS Total_Customers,
    SUM(Churn) AS Churned_Customers,
    ROUND(
        SUM(Churn) * 100.0 / COUNT(*),
        2
    ) AS Churn_Rate
FROM customers
GROUP BY PaymentMethod
ORDER BY Churn_Rate DESC;