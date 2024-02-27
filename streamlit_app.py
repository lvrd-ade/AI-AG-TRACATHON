import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


#------------------------------------------ Cohort Stuff
import cohere
co = cohere.Client('<<i8LhL790rANZGVLff63yPnvuykoeu1SnkRdi9QKz>>')

from cohere.responses.classify import Example


examples=[
  Example("How do I ride a tractor", "riding fukin tractors boa"),
  Example("How do I sign up for electronic filing?", "Change account settings"),
  Example("How do I change my policy?", "Change account settings"),
  Example("How do I sign up for direct deposit?", "Change account settings"),
  Example("I want direct deposit. Can you help with that?", "Change account settings"),
  Example("Could you deposit money into my account rather than mailing me a physical cheque?", "Change account settings")
]

inputs = [" I want to change my password", 
          "Does my policy cover prescription medication?",
          "I want to ride a fukin tractor boa"
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
          "option": "Change account settings",
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
    },
    {
      "text":  "I want to ride a fukin tractor boa",
      "prediction": "riding fukin tractors boa",
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
         "riding fukin tractors boa": {
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