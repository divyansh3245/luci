import streamlit as st
import numpy as np 
import joblib
import warnings
warnings.filterwarnings("ignore")

# load model and scaler
model = joblib.load('wine_model.pkl')
scaler = joblib.load('scaler.pkl')

# titling the web app (gotta change this later)
st.title('wine quality prediction')

# input all features of the dataset
features = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

user_input = []

for feature in features:
    value = st.number_input(f'Enter {feature}:', step=0.1)
    user_input.append(value)

# predict button implementation

if st.button('Predict'):
    input_scaled = scaler.transform([user_input])
    prediction = model.predict(input_scaled)
    st.success(f'predicted wine quality: {round(prediction[0], 2)}')

    # human readable prediction
    quality = prediction[0]
    if quality < 4:
        label = 'poor quality'
    elif quality < 5.6:
        label = 'average quality'
    elif quality < 6.6:
        label = 'good quality'
    else: 
        label = 'excellent quality'

    # write it off on the ST interface
    st.write(f'**wine quality label:** {label}')