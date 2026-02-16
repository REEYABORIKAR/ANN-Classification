import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

## Load the trained model
model=tf.keras.models.load_model('regression_model.h5')

## Load the encoders and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb')as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)

## Streamlit app
st.title('Estimated Salary Prediction')

## User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 30)
balance = st.number_input('Account Balance ($)', value=0.0, format="%.2f")
credit_score = st.number_input('Credit Score', 300, 850, 600)
exited=st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure (Years)', 0, 10, 5)
num_of_products = st.number_input('Number of Products', 1, 4, 1)
has_cr_card = st.toggle('Has Credit Card')
is_active_member = st.toggle('Is Active Member')

## Prepare the input data
input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card else 0],
        'IsActiveMember': [1 if is_active_member else 0],
        'Exited':[exited]
    })

## Onehot encode 'Geograohy'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combine onehot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df])

## Scale the input data
input_data_scaled=scaler.transform(input_data)

## Predict estimated salary
prediction=model.predict(input_data_scaled)
predicted_salary=prediction[0][0]

st.write(f'Predicted Estimated Salary: ${predicted_salary:.2f}')