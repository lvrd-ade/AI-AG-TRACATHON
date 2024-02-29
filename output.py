import json
import pandas as pd

from sklearn.model_selection import train_test_split
from globals import *

def main():
    # Load the data from the CSV file into a Pandas DataFrame.
    df = pd.read_csv(FILE_ROOT + "cleaned_data.csv")

    # TODO: split cleaned data into training data and validation data

    # Separate features (X) and target variables (y) from the dataset
    X = df[['Year', 'Month', 'Day', 'MaxT', 'MinT', 'AvgT', 'WindSpeed', 'Humidity', 'Precipitation']]
    y_rainfall = df['Rainfall']
    y_extreme_rainfall = df['ExtremeRainfall']

    # Split the dataset into training and validation sets for rainfall prediction
    X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(X, y_rainfall, test_size=0.2, random_state=42)

    # Split the dataset into training and validation sets for extreme rainfall prediction
    X_train_erf, X_val_erf, y_train_erf, y_val_erf = train_test_split(X, y_extreme_rainfall, test_size=0.2, random_state=42)

    # Print the shapes of the training and validation sets for rainfall prediction
    print("Rainfall Prediction:")
    print("X_train_rf shape:", X_train_rf.shape)
    print("X_val_rf shape:", X_val_rf.shape)
    print("y_train_rf shape:", y_train_rf.shape)
    print("y_val_rf shape:", y_val_rf.shape)

    # Print the shapes of the training and validation sets for extreme rainfall prediction
    print("\nExtreme Rainfall Prediction:")
    print("X_train_erf shape:", X_train_erf.shape)
    print("X_val_erf shape:", X_val_erf.shape)
    print("y_train_erf shape:", y_train_erf.shape)
    print("y_val_erf shape:", y_val_erf.shape)



main()