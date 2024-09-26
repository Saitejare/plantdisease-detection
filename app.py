import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the model
model = joblib.load('credit_card_fraud_detection.pkl')

# Function to predict fraud
def predict_fraud(features):
    prediction = model.predict([features])
    return prediction[0]

# UI components
st.title("Credit Card Fraud Detection")

# Input fields for transaction data
trans_num = st.text_input("Transaction Number:")
cc_num = st.text_input("Credit Card Number:")  # This should ideally be a string
amt = st.number_input("Transaction Amount:", min_value=0.0)
merchant = st.text_input("Merchant:")
city = st.text_input("City:")
trans_date_trans_time = st.date_input("Transaction Date:", value=datetime.today())

# Prepare features for prediction
features = [
    float(amt),          # Amount as float
    merchant,           # Categorical
    city,               # Categorical
    trans_date_trans_time.strftime("%Y-%m-%d")  # Date as string
]

# Predict button
if st.button("Predict"):
    # One-hot encode categorical features
    df_features = pd.DataFrame([features], columns=['amt', 'merchant', 'city', 'trans_date_trans_time'])
    df_encoded = pd.get_dummies(df_features, drop_first=True)
    
    # Align columns with the model
    model_columns = joblib.load('x_train_columns.pkl')  # Load the model columns if saved
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Make the prediction
    result = model.predict(df_encoded)
    
    # Display the result
    if result[0] == 0:
        st.success("Fraud Transaction Detected!")
    else:
        st.success("Transaction is Not Fraud details.")
