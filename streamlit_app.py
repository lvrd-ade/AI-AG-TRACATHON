import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


#------------------------------------------ Cohort Stuff
import cohere
co = cohere.Client('i8LhL790rANZGVLff63yPnvuykoeu1SnkRdi9QKz')

from cohere.responses.classify import Example


examples=[
  Example("How do I find my insurance policy?", "Finding policy details"),
  Example("How do I download a copy of my insurance policy?", "Finding policy details"),
  Example("How do I find my policy effective date?", "Finding policy details"),
  Example("When does my insurance policy end?", "Finding policy details"),
  Example("Could you please tell me the date my policy becomes effective?", "Finding policy details"),
  Example("How do I sign up for electronic filing?", "Change account settings"),
  Example("How do I change my policy?", "Change account settings"),
  Example("How do I sign up for direct deposit?", "Change account settings"),
  Example("I want direct deposit. Can you help with that?", "Change account settings"),
  Example("Could you deposit money into my account rather than mailing me a physical cheque?", "Change account settings"),
  Example("How do I file an insurance claim?", "Filing a claim and viewing status"),
  Example("How do I file a reimbursement claim?", "Filing a claim and viewing status"),
  Example("How do I check my claim status?", "Filing a claim and viewing status"),
  Example("When will my claim be reimbursed?", "Filing a claim and viewing status"),
  Example("I filed my claim 2 weeks ago but I still haven’t received a deposit for it.", "Filing a claim and viewing status"),
  Example("I want to cancel my policy immediately! This is nonsense.", "Cancelling coverage"),
  Example("Could you please help my end my insurance coverage? Thank you.",
  "Cancelling coverage"),
  Example("Your service sucks. I’m switching providers. Cancel my coverage.", "Cancelling coverage"),
  Example("Hello there! How do I cancel my coverage?", "Cancelling coverage"),
  Example("How do I delete my account?", "Cancelling coverage")
]


inputs = ["I want to change my password", 
          "Does my policy cover prescription medication?" 
         ]

response = co.classify(  
    model='large',  
    inputs=inputs,  
    examples=examples)

print(response.classifications)


{
  "results": [
    {
      "text": "I want to change my password",
      "prediction": "Change account settings",
      "confidence": 0.82,
      "confidences": [
        {
          "option": "Finding policy details",
          "confidence": 0.05
        },
        {
          "option": "Change account settings ",
          "confidence": 0.82
        },
        {
          "option": "Filing a claim and viewing status",
          "confidence": 0.05
        },
       {
          "option":  "Cancelling coverage",
          "confidence": 0.08
        }
      ],
      
      "labels": {
       "Finding policy details": {
          "confidence": 0.05
        },
        "Change account settings": {
          "confidence": 0.82
        },
        "Filing a claim and viewing status": {
          "confidence": 0.05
        },
         "Cancelling coverage": {
          "confidence": 0.08
        }
      }
    },
    {
      "text":  "Does my policy cover prescription medication?",
      "prediction": "Finding policy details",
      "confidence": 0.75,
      "confidences": [
        {
          "option": "Finding policy details",
          "confidence": 0.75
        },
        {
          "option": "Change account settings",
          "confidence": 0.15
        },
        {
          "option": "Filing a claim and viewing status",
          "confidence": 0.05
        },
       {
          "option":  "Cancelling coverage",
          "confidence": 0.05
        }
      ],
      "labels": {
         "Finding policy details": {
          "confidence": 0.75
        },
        "Change account settings": {
          "confidence": 0.15
        },
        "Filing a claim and viewing status": {
          "confidence": 0.05
        },
         "Cancelling coverage": {
          "confidence": 0.05
        }
      }
    }
  ]
}





#------------------------------------ streamlit stuff
st.title('My First Streamlit App')
st.write("Hello, world!")

if st.button('Click me'):
    st.write("You clicked the button!")

# User input area
user_input = st.text_input("Enter your query:", "")
# Button to trigger classification
if st.button('Submit'):
    if user_input:  # Check if there's input
        response = co.classify(
            model='large',  
            inputs=[user_input],  
            examples=examples
        )
        classification = response.classifications[0]

        # Display the AI's response
        st.write("Predicted Category: ", classification.prediction)
        st.write("Confidence:", classification.confidence)
    else:
        st.write("Please enter some text.")

# Placeholder data
x = np.arange(10)
y = np.random.randn(10)

# Create three columns for the graphs
col1, col2, col3 = st.columns(3)

# Graph in the first column
with col1:
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

# Graph in the second column
with col2:
    fig, ax = plt.subplots()
    ax.plot(x, y) 
    st.pyplot(fig)

# Graph in the third column
with col3:
    fig, ax = plt.subplots()
    ax.plot(x, y) 
    st.pyplot(fig)