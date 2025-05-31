import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import base64
import os
from fpdf import FPDF
from PIL import Image

# 🟢 إعداد الخلفية
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background(image_file):
    if not os.path.exists(image_file):
        st.error(f"❌ Background image not found: {image_file}")
        return
    
    base64_str = get_base64_of_image(image_file)
    css_code = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* تكبير وتلوين اسم الفيتشر (label) فوق number_input */
    div[data-testid="stNumberInput"] > label > div {
        font-size: 20px !important;
        color: white !important;
        font-weight: bold !important;
    }

    /* لو حابب كمان تنسق الليبل بتاع selectbox */
    div[data-testid="stSelectbox"] > label > div {
        font-size: 20px !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)



set_background("photo1.jpg")




# Load the trained model and preprocessing files
model = joblib.load("catboost_model.pkl")
standard_scaler = joblib.load("standard_scaler.pkl")

with open("power_transformers.pkl", "rb") as f:
    power_transformers = pickle.load(f)

with open("winsor_bounds.pkl", "rb") as f:
    winsor_bounds = pickle.load(f)

label_encoders = joblib.load("label_encoders.pkl")

st.title("🔍 Loan Approval Prediction App")
st.markdown("Kindly Fill The Requirements")

# Input features
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Monthly Income", min_value=0, value=5000)
person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
person_emp_length = st.selectbox("Employment Length", ['<1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10-20yrs', '20+yrs'])
loan_intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.number_input("Loan Amount", min_value=1000, value=10000)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=12.5)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.2)
cb_person_default_on_file = st.selectbox("Has Default on File?", ['Y', 'N'])
cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, value=5)

if st.button("🔮 Predict Loan Approval"):

    # Create DataFrame
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    # Label encoding
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Standard scaling
    input_data[['person_age', 'loan_int_rate']] = standard_scaler.transform(
        input_data[['person_age', 'loan_int_rate']]
    )

    # Apply PowerTransformers individually
    pt_cols = ['loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length']
    for col in pt_cols:
        input_data[col] = power_transformers[col].transform(input_data[[col]])

    # Apply PowerTransformer + Winsorizing for person_income
    input_data['person_income'] = power_transformers['person_income'].transform(
        input_data[['person_income']]
    )
    lower = winsor_bounds['person_income']['lower']
    upper = winsor_bounds['person_income']['upper']
    input_data['person_income'] = input_data['person_income'].clip(lower=lower, upper=upper)

    # Predict
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan Approved with probability: {proba:.2%}")
    else:
        st.error(f"❌ Loan NOT Approved – probability of approval: {proba:.2%}")
