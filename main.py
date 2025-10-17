import streamlit as st
import pandas as pd
import pickle
import os
import re

# --- 1) CONFIGURATION ---
PIPELINE_PATH = r"C:\Users\User\Desktop\PY DS\Dixson\ML\ML PROJECTS\house prices\house_prices_model.pkl"

# --- 2) LOAD YOUR PICKLED PIPELINE ---
try:
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    required_keys = {"scaler", "model", "columns"}
    missing = required_keys - set(pipeline.keys())
    if missing:
        st.error(f"Loaded pipeline missing keys: {', '.join(missing)}")
        st.stop()
except FileNotFoundError:
    st.error(f"Model file not found at: {PIPELINE_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3) APP TITLE ---
st.title("Bengaluru House Price Predictor")

# --- 4) USER INPUT WIDGETS ---
loc=pd.read_csv("bengaluru_house_prices.csv")['location'].dropna().astype(str).unique()
loc=[re.sub(r"[^a-zA-Z0-9 ]","",str(x)).strip().title() for x in loc]
location = st.selectbox("Enter the location:",loc)
sqft     = st.number_input("Enter the total square feet:", min_value=300.0, max_value=16000.0, value=1000.0)
bath     = st.selectbox("Number of bathrooms:", range(1, 6))
bhk      = st.selectbox("Size (in BHK):", range(1, 10))

# --- 5) PREDICT BUTTON ---
if st.button("Predict Price"):
    # Validate inputs
    if not location.strip():
        st.warning("Please enter a valid location before predicting.")
    else:
        # Prepare input DataFrame
        df_input = pd.DataFrame([{  
            "location": location.lower().strip(),
            "total_sqft": sqft,
            "bath":       bath,
            "size":       bhk
        }])

        # One-hot encode and align to training columns
        try:
            df_encoded = pd.get_dummies(df_input)
            df_encoded = df_encoded.reindex(columns=pipeline["columns"], fill_value=0)
        except Exception as e:
            st.error(f"Error encoding input data: {e}")
            st.stop()

        # Scale and predict
        try:
            X_scaled = pipeline["scaler"].transform(df_encoded)
            prediction = pipeline["model"].predict(X_scaled)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Display result
        price = max(0.0, prediction)
        st.success(f"Predicted Price: ₹{price:,.2f} lakhs")