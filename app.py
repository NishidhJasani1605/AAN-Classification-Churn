import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

# loading the train model
model = tf.keras.models.load_model('model.h5')

# load the encoder an scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Customer Churn Prediction')

# user input
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age' , 18 , 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_Salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure' , 0 , 10)
num_Of_Products = st.number_input('Number of Products')
has_Cr_Card = st.number_input('Has Credit Card')
is_Active_Member = st.number_input('Is Active Member')

# prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_Of_Products],
    'HasCrCard': [has_Cr_Card],
    'IsActiveMember': [is_Active_Member],
    'EstimatedSalary': [estimated_Salary],
})

# one hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# combine one-hot encoded columns with input data
input_df = pd.concat([input_data.reset_index(drop=True) , geo_encoded_df], axis=1)

# scale the input data
scaled_input = scaler.transform(input_df)

# make the prediction
prediction = model.predict(scaled_input)
prediction_prob = prediction[0][0]

st.write(f"Churm Probability: {prediction_prob}")

# display the prediction
if prediction_prob > 0.5:
    st.write("The customer is likely to leave the bank.")
else:
    st.write("The customer is likely to stay with the bank.")