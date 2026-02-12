import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Set page configuration
st.set_page_config(page_title="ChurnGuard AI", page_icon="📊", layout="wide")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache model loading for better performance
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    return model, le, ohe, sc

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()

# --- Header Section ---
st.title(' Customer Churn Analytics')
st.markdown("Predict the likelihood of a customer leaving the bank based on their profile.")
st.divider()

# --- Sidebar / Input Section ---
st.subheader(" Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 30)

with col2:
    credit_score = st.number_input('Credit Score', 300, 850, 600)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)
    num_of_products = st.number_input('Number of Products', 1, 4, 1)

with col3:
    balance = st.number_input('Account Balance ($)', value=0.0, format="%.2f")
    estimated_salary = st.number_input('Estimated Salary ($)', value=50000.0, format="%.2f")

st.markdown("###  Account Settings")
c1, c2 = st.columns(2)
with c1:
    has_cr_card = st.toggle('Has Credit Card')
with c2:
    is_active_member = st.toggle('Is Active Member')

# --- Prediction Logic ---
if st.button('Analyze Customer Risk'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card else 0],
        'IsActiveMember': [1 if is_active_member else 0],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale and Predict
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prob = float(prediction[0][0])

    # --- Results Display ---
    st.divider()
    st.subheader("Analysis Result")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric(label="Churn Probability", value=f"{prob:.1%}")
    
    with col_res2:
        if prob > 0.5:
            st.error(" **High Risk:** This customer is likely to churn.")
            st.progress(prob)
        else:
            st.success(" **Low Risk:** This customer is likely to stay.")
            st.progress(prob)