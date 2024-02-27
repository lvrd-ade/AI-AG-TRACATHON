import json
import pandas as pd

from globals import *

def main():
    # Load the data from the CSV file into a Pandas DataFrame.
    df = pd.read_csv(FILE_ROOT + "cleaned_data.csv")

    # TODO: split cleaned data into training data and validation data, create a XGBoost model for prediction

main()