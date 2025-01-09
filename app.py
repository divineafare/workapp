import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Streamlit App
st.title("Product Pricing Analysis and Prediction")

# Load the dataset
@st.cache_data  # Updated to use st.cache_data
def load_data():
    df = pd.read_csv('product_pricing_dataset.csv')
    return df

df = load_data()

# Display dataset overview
st.header("Dataset Overview")
st.write("First few rows of the dataset:")
st.write(df.head())

st.write("Dataset shape:", df.shape)

# Check for missing values
st.write("Missing values in the dataset:")
st.write(df.isnull().sum())

# Visualization
st.header("Visualization")
st.write("Scatterplot: Price vs Units Sold (colored by Product Category)")
fig, ax = plt.subplots()
sns.scatterplot(x="Price", y="Units_Sold", hue="Product_Category", data=df, ax=ax)
st.pyplot(fig)

# Feature Selection
X = df[["Price", "Competitor_Price", "Customer_Rating", "Demand_Elasticity"]]
y = df["Units_Sold"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the trained model (optional)
joblib.dump(model, "pricing_model.pkl")
st.write("Trained model saved as 'pricing_model.pkl'.")

# Prediction
st.header("Make Predictions")
st.write("Enter product details to predict units sold:")

# User input for prediction
price = st.number_input("Price", min_value=0.0, value=50.0, step=1.0)
competitor_price = st.number_input("Competitor Price", min_value=0.0, value=50.0, step=1.0)
customer_rating = st.number_input("Customer Rating (1-5)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
demand_elasticity = st.number_input("Demand Elasticity", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)

# Predict button
if st.button("Predict Units Sold"):
    input_data = pd.DataFrame([[price, competitor_price, customer_rating, demand_elasticity]],
                              columns=["Price", "Competitor_Price", "Customer_Rating", "Demand_Elasticity"])
    prediction = model.predict(input_data)
    st.write(f"Predicted Units Sold: {prediction[0]:.2f}")
