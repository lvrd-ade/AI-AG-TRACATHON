import streamlit as st
import matplotlib.pyplot as plt
import base64
import numpy as np
import pandas as pd 
from joblib import load


#------------------------------------------ CSS Stuff
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
img = get_img_as_base64("img.jpg")
page_bg_image = f"""
  <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    background-size: 100%;
    background-position: top left;
    background-repeat: no-repeat;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
  </style>
"""
st.markdown(page_bg_image, unsafe_allow_html=True)



st.title('Rainfall Predictions Utilizing Machine Learning')


data = {
    "humidity": [85, 90, 69.10, 92, 86, 87, 88],
    "max_temperature": [28, 30, 27.40, 37, 35, 30, 33],
    "min_temperature": [25, 20, 25.70, 27, 25, 20, 28],
    "avg_wind_speed": [5, 7, 4.6, 1.8, 2.3, 6.8, 5.6],
    "month": [4, 4, 4, 4, 1, 1, 4],  
    "day": [15, 16, 17, 18, 19, 2, 21],  
    "year": [2024, 2024, 2024, 2024, 2024, 2024, 2024], 
    #"avg_temperature": [73, 70, 77, 72, 65, 75, 70], 
}
dfDisplayData = pd.DataFrame(data)

# Load your trained CatBoost model
model = load('catboost_rainfall_model.joblib')

# Function to make predictions
def predict_rainfall(input_data):
    prediction = model.predict(input_data)
    return prediction
    
#css for making the horizontal line custom
st.markdown("""
  <style>
      .day-divider {
          border-bottom: 10px solid black; /* Changed color to black and made thicker*/
          margin-bottom: 10px; 
      }
      .prediction-label {  /* Style for the prediction label */
            font-size: 40px;
            color: white;  /* White text on black background */
            background-color: black;
            padding: 5px;  /* Add some padding for visual comfort */
      }
  </style>
  """, unsafe_allow_html=True)
    

#third iteration
if st.button('Predict Rainfall'):
    
  
  

  for i in range(7):
      
      st.markdown('<div class="day-divider"></div>', unsafe_allow_html=True)
      
      # st.markdown('---')  # Add a horizontal line 
      
      # Input data (get this from user input later)
      input_data = np.array([[dfDisplayData["year"][i], dfDisplayData["month"][i], dfDisplayData["day"][i], dfDisplayData["max_temperature"][i], dfDisplayData["min_temperature"][i], dfDisplayData["avg_wind_speed"][i], dfDisplayData["humidity"][i]]])

      # Prediction
      prediction = predict_rainfall(input_data)
      prediction_label = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Extreme Rain'][prediction[0][0]]

      # Create two columns
      col1, col2, = st.columns(2)

      # Display weather data in the first column
      with col1:
          st.write("**Day:**", i + 1)  
          st.write("**Year:**", dfDisplayData["year"][i])
          st.write("**Month:**", dfDisplayData["month"][i])
          st.write("**Day:**", dfDisplayData["day"][i])
          st.write("**Max Temperature:**", dfDisplayData["max_temperature"][i])
          st.write("**Min Temperature:**", dfDisplayData["min_temperature"][i])
          st.write("**Avg. Wind Speed:**", dfDisplayData["avg_wind_speed"][i])
          st.write("**Humidity:**", dfDisplayData["humidity"][i]) 

      # Display prediction in the second column
      with col2:
        if prediction_label == 'No Rain':
          st.write(f'<span class="prediction-label">**Predicted Rainfall:** {prediction_label}🌵</span>', unsafe_allow_html=True, icon="🌵")
        elif prediction_label == 'Light Rain':
          st.write(f'<span class="prediction-label">**Predicted Rainfall:** {prediction_label}💧</span>', unsafe_allow_html=True, icon="💧")
        elif prediction_label == 'Moderate Rain':
          st.write(f'<span class="prediction-label">**Predicted Rainfall:** {prediction_label}💦</span>', unsafe_allow_html=True, icon="💦")
        elif prediction_label == 'Heavy Rain':
          st.write(f'<span class="prediction-label">**Predicted Rainfall:** {prediction_label}🌧️</span>', unsafe_allow_html=True, icon="🌧️")
        elif prediction_label == 'Extreme Rain':
          st.write(f'<span class="prediction-label">**Predicted Rainfall:** {prediction_label}⚠️⛈️</span>', unsafe_allow_html=True, icon="⚠️⛈️")

                
