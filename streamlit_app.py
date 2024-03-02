import datetime
import streamlit as st
import matplotlib.pyplot as plt
import base64
import numpy as np
import cohere # cohere AI import
from dotenv import load_dotenv
import os
import pandas as pd 
import pickle
from joblib import load
import plotly.express as px

load_dotenv()
COHERE_API_KEY = os.getenv("API_KEY")


#loading our rain prediction model
#model = pickle.load(open('rainfall_prediction_model.pkl', 'rb')) 



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

#------------------------------------------ Cohort Stuff
co = cohere.Client(COHERE_API_KEY)

# from cohere.responses.classify import Example

# examples=[
#   Example("How do I find my insurance policy?", "Finding policy details"),
#   Example("How do I download a copy of my insurance policy?", "Finding policy details"),
#   Example("How do I find my policy effective date?", "Finding policy details"),
#   Example("When does my insurance policy end?", "Finding policy details"),
#   Example("Could you please tell me the date my policy becomes effective?", "Finding policy details"),
#   Example("How do I sign up for electronic filing?", "Change account settings"),
#   Example("How do I change my policy?", "Change account settings"),
#   Example("How do I sign up for direct deposit?", "Change account settings"),
#   Example("I want direct deposit. Can you help with that?", "Change account settings"),
#   Example("Could you deposit money into my account rather than mailing me a physical cheque?", "Change account settings"),
#   Example("How do I file an insurance claim?", "Filing a claim and viewing status"),
#   Example("How do I file a reimbursement claim?", "Filing a claim and viewing status"),
#   Example("How do I check my claim status?", "Filing a claim and viewing status"),
#   Example("When will my claim be reimbursed?", "Filing a claim and viewing status"),
#   Example("I filed my claim 2 weeks ago but I still haven’t received a deposit for it.", "Filing a claim and viewing status"),
#   Example("I want to cancel my policy immediately! This is nonsense.", "Cancelling coverage"),
#   Example("Could you please help my end my insurance coverage? Thank you.",
#   "Cancelling coverage"),
#   Example("Your service sucks. I’m switching providers. Cancel my coverage.", "Cancelling coverage"),
#   Example("Hello there! How do I cancel my coverage?", "Cancelling coverage"),
#   Example("How do I delete my account?", "Cancelling coverage")
# ]

# inputs = ["I want to change my password", 
#           "Does my policy cover prescription medication?" 
#          ]

# response = co.classify(  
#     model='large',  
#     inputs=inputs,  
#     examples=examples)

# print(response.classifications)

# {
#   "results": [
#     {
#       "text": "I want to change my password",
#       "prediction": "Change account settings",
#       "confidence": 0.82,
#       "confidences": [
#         {
#           "option": "Finding policy details",
#           "confidence": 0.05
#         },
#         {
#           "option": "Change account settings ",
#           "confidence": 0.82
#         },
#         {
#           "option": "Filing a claim and viewing status",
#           "confidence": 0.05
#         },
#        {
#           "option":  "Cancelling coverage",
#           "confidence": 0.08
#         }
#       ],
      
#       "labels": {
#        "Finding policy details": {
#           "confidence": 0.05
#         },
#         "Change account settings": {
#           "confidence": 0.82
#         },
#         "Filing a claim and viewing status": {
#           "confidence": 0.05
#         },
#          "Cancelling coverage": {
#           "confidence": 0.08
#         }
#       }
#     },
#     {
#       "text":  "Does my policy cover prescription medication?",
#       "prediction": "Finding policy details",
#       "confidence": 0.75,
#       "confidences": [
#         {
#           "option": "Finding policy details",
#           "confidence": 0.75
#         },
#         {
#           "option": "Change account settings",
#           "confidence": 0.15
#         },
#         {
#           "option": "Filing a claim and viewing status",
#           "confidence": 0.05
#         },
#        {
#           "option":  "Cancelling coverage",
#           "confidence": 0.05
#         }
#       ],
#       "labels": {
#          "Finding policy details": {
#           "confidence": 0.75
#         },
#         "Change account settings": {
#           "confidence": 0.15
#         },
#         "Filing a claim and viewing status": {
#           "confidence": 0.05
#         },
#          "Cancelling coverage": {
#           "confidence": 0.05
#         }
#       }
#     }
#   ]
# }





#------------------------------------ streamlit stuff


st.title('Rainfall Predictions Utilizing Machine Learning')
# st.write("Hello, world!")

#----------Cohere UI Stuff----------
# User input area
# user_input = st.text_input("Enter your query:", "")
# Button to trigger classification
# if st.button('Submit'):
#     if user_input:  # Check if there's input
#         response = co.classify(
#             model='large',  
#             inputs=[user_input],  
#             examples=examples
#         )
#         classification = response.classifications[0]

#         # Display the AI's response
#         st.write("Predicted Category: ", classification.prediction)
#         st.write("Confidence:", classification.confidence)
#     else:
#         st.write("Please enter some text.")
#--------------------
        


#--------Placeholder Graph Stuff-------
# Placeholder data
# x = np.arange(10)
# y = np.random.randn(10)

# # Create three columns for the graphs
# col1, col2, col3 = st.columns(3)

# # Graph in the first column
# with col1:
#     fig, ax = plt.subplots()
#     ax.plot(x, y)
#     st.pyplot(fig)

# # Graph in the second column
# with col2:
#     fig, ax = plt.subplots()
#     ax.plot(x, y) 
#     st.pyplot(fig)

# # Graph in the third column
# with col3:
#     fig, ax = plt.subplots()
#     ax.plot(x, y) 
#     st.pyplot(fig)
#------------------------------------



#-----Inputting user data stuff-----
# User date input
# user_date = st.date_input("Select a date:", datetime.date.today())

# # are we doing the computation for this prediction in-app, or sending this date to a backend?
# st.write("You selected:", user_date)
#-----------------------------------



#Rain Prediction stuff------------------

data = {
    "humidity": [75, 68, 80, 72, 60, 78, 65],
    "max_temperature": [82, 78, 85, 80, 72, 83, 79],
    "min_temperature": [65, 61, 68, 63, 58, 66, 62],
    "avg_wind_speed": [12, 8, 15, 10, 9, 11, 7],
    "month": [8, 8, 8, 8, 8, 8, 8],  
    "day": [15, 16, 17, 18, 19, 20, 21],  
    "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023], 
    "avg_temperature": [73, 70, 77, 72, 65, 75, 70], 
}
dfDisplayData = pd.DataFrame(data)

# Load your trained CatBoost model
model = load('catboost_rainfall_model.joblib')

# Function to make predictions
def predict_rainfall(input_data):
    prediction = model.predict(input_data)
    return prediction


#first iteration
  # for i in range(7):
  #   # Creating a numpy array from the input data
  #   # input_data = [dfDisplayData["humidity"][i], dfDisplayData["max_temperature"][i], dfDisplayData["min_temperature"][i], dfDisplayData["avg_wind_speed"][i],dfDisplayData["month"][i],dfDisplayData["day"][i],dfDisplayData["year"][i],dfDisplayData["avg_temperature"][i]] # Construct from user input
  #   input_data = np.array([[dfDisplayData["year"][i], dfDisplayData["month"][i], dfDisplayData["day"][i], dfDisplayData["max_temperature"][i], dfDisplayData["min_temperature"][i], dfDisplayData["avg_wind_speed"][i], dfDisplayData["humidity"][i]]])
  #   # Making prediction
  #   prediction = predict_rainfall(input_data)
    
  #   print(prediction)  # Add this to debug

  #   # Mapping prediction to label
  #   prediction_label = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Extreme Rain'][prediction[0][0]]

#second iteration
# # Button to make prediction
# if st.button('Predict Rainfall'):
#     for i in range(7):
#         # Input data (get this from user input later)
#         input_data = np.array([[dfDisplayData["year"][i], dfDisplayData["month"][i], dfDisplayData["day"][i], dfDisplayData["max_temperature"][i], dfDisplayData["min_temperature"][i], dfDisplayData["avg_wind_speed"][i], dfDisplayData["humidity"][i]]])

#         # Prediction
#         prediction = predict_rainfall(input_data)
#         prediction_label = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Extreme Rain'][prediction[0][0]]

#         # Display weather data
#         st.write("**Day:**", i + 1)  # Adjust if you need zero-based indexing
#         st.write("**Year:**", dfDisplayData["year"][i])
#         st.write("**Month:**", dfDisplayData["month"][i])
#         st.write("**Day:**", dfDisplayData["day"][i])
#         st.write("**Max Temperature:**", dfDisplayData["max_temperature"][i])
#         st.write("**Min Temperature:**", dfDisplayData["min_temperature"][i])
#         st.write("**Avg. Wind Speed:**", dfDisplayData["avg_wind_speed"][i])
#         st.write("**Humidity:**", dfDisplayData["humidity"][i]) 

#     # print(prediction_label)
#     if prediction_label == 'No Rain':
#       st.success(f'The predicted rainfall category is: {prediction_label}', icon="🌵")
#       # print("goo goo doo doo")
#     elif prediction_label == 'Light Rain':
#       st.success(f'The predicted rainfall category is: {prediction_label}', icon="💧")
#     elif prediction_label == 'Moderate Rain':
#       st.success(f'The predicted rainfall category is: {prediction_label}', icon="💦")
#     elif prediction_label == 'Heavy Rain':
#       st.success(f'The predicted rainfall category is: {prediction_label}', icon="🌧️")
#     elif prediction_label == 'Extreme Rain':
#       st.success(f'The predicted rainfall category is: {prediction_label}', icon="⚠️⛈️")
    

#third iteration
if st.button('Predict Rainfall'):

    for i in range(7):
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
                st.write('**Predicted Rainfall:**', prediction_label, icon="🌵")
            elif prediction_label == 'Light Rain':
                st.write('**Predicted Rainfall:**', prediction_label, icon="💧")
            elif prediction_label == 'Light Rain':
                st.write('**Predicted Rainfall:**', prediction_label, icon="💦")
            elif prediction_label == 'Light Rain':
                st.write('**Predicted Rainfall:**', prediction_label, icon="🌧️")
            elif prediction_label == 'Light Rain':
                st.write('**Predicted Rainfall:**', prediction_label, icon="⚠️⛈️")
            




#pandas dataframe to hold display



# Prediction
# if st.button("Predict"):
    
  
    
#     input_data = [dfDisplayData["humidity"][i], dfDisplayData["max_temperature"][i], dfDisplayData["min_temperature"][i], dfDisplayData["avg_wind_speed"][i],dfDisplayData["month"][i],dfDisplayData["day"][i],dfDisplayData["year"][i],dfDisplayData["avg_temperature"][i]] # Construct from user input
#     prediction = model.predict(input_data)

#     if prediction == 1:
#         st.write("Prediction: It's likely to rain.")
#     else:
#         st.write("Prediction: It's unlikely to rain.")