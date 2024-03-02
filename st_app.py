import streamlit as st
from joblib import load
import numpy as np

# Load your trained CatBoost model
model = load('catboost_rainfall_model.joblib')

# Function to make predictions
def predict_rainfall(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
def main():
    st.title('Rainfall Prediction App')

    # Input fields
    year = st.number_input('Year', min_value=2000, max_value=2050, value=2021)
    month = st.number_input('Month', min_value=1, max_value=12, value=1)
    day = st.number_input('Day', min_value=1, max_value=31, value=1)
    max_temp = st.number_input('Max Temperature')
    min_temp = st.number_input('Min Temperature')
    wind_speed = st.number_input('Wind Speed')
    humidity = st.number_input('Humidity')

    # Button to make prediction
    if st.button('Predict Rainfall'):
        # Creating a numpy array from the input data
        input_data = np.array([[year, month, day, max_temp, min_temp, wind_speed, humidity]])

        # Making prediction
        prediction = predict_rainfall(input_data)
        
        print(prediction)  # Add this to debug

        # Mapping prediction to label
        prediction_label = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Extreme Rain'][prediction[0][0]]
        
        st.success(f'The predicted rainfall category is: {prediction_label}')

if __name__ == '__main__':
    main()
