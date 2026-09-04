import mysql.connector

try:
    connection = mysql.connector.connect(
        host="localhost",
        port=3307,
        user="churn_user",
        password="Churn@123",
        database="customer_churn_db"
    )

    if connection.is_connected():
        print("MySQL connected successfully!")

except mysql.connector.Error as error:
    print("Error while connecting to MySQL:", error)

finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("MySQL connection closed.")