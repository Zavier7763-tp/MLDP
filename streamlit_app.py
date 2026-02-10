import joblib
import streamlit as st
import numpy as np
import pandas as pd
import base64

## Load trained model
model = joblib.load("logistic_regression_model.pkl")

## Streamlit app
st.title("Smartphone Price Range Prediction")

## User inputs
ram_selected = st.number_input("Enter RAM (MB)", 
                               min_value=256, 
                               max_value=4000, 
                               value=2048)

px_width_selected = st.number_input("Enter Pixel Width", 
                                    min_value=500, 
                                    max_value=2000, 
                                    value=720)

px_height_selected = st.number_input("Enter Pixel Height", 
                                     min_value=1, 
                                     max_value=2000, 
                                     value=1280)

battery_power_selected = st.number_input("Enter Battery Power (mAh)", 
                                         min_value=500, 
                                         max_value=5000, 
                                         value=3000)

mobile_wt_selected = st.number_input("Enter Mobile Weight (grams)", 
                                     min_value=80, 
                                     max_value=200, 
                                     value=150)

## Predict button
if st.button("Predict Price Range"):
    
    # Calculate battery_per_weight
    battery_per_weight = battery_power_selected / mobile_wt_selected

    ## Create DataFrame for input features
    df_input = pd.DataFrame({
        'ram': [ram_selected],
        'battery_per_weight': [battery_per_weight],
        'px_width': [px_width_selected],
        'px_height': [px_height_selected],
        'battery_power': [battery_power_selected]
    })

    df_input = df_input.reindex(columns=model.feature_names_in_,
                                fill_value=0)

    ## Predict
    y_pred = model.predict(df_input)[0]
    
    price_ranges = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }
    
    st.success(f"Predicted Price Range: {price_ranges[y_pred]}")

## Page design
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("smartphone.webp")

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
            url("data:image/webp;base64,{bg}");
        background-size: cover;
        background-position: center;
    }}
    .stAlert {{
        background-color: rgba(0, 0, 0, 0.6) !important;
    }}
    .stAlert > div {{
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)