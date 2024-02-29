import json
import pandas as pd

from globals import *

def main():
    # Load the data from the CSV file into a Pandas DataFrame.
    df = pd.read_csv(FILE_ROOT + "Farm_Weather_Data.csv")
    
    # Clean data
    df = df[df['MaxT'] >= df['MinT']]   # Remove rows where MaxT is less than MinT
    df['Date'] = df['Date'].str.split().str[0]   # Extract date part of Date column
    #df = df[(df['Humidity'] >= 60) & (df['Precipitation'] > 0.5)]

    # extra columns for analysis
    df['Year'] = df['Date'].apply(lambda x: x.split('-')[0])
    df['Month'] = df['Date'].apply(lambda x: x.split('-')[1])
    df['Day'] = df['Date'].apply(lambda x: x.split('-')[2])
    df['AvgT'] = (df['MaxT'] + df['MinT']) / 2  # in Celsius

    # Thresholds for precipitation and humidity
    precipitation_threshold = 0.1  # mm
    humidity_threshold = 60  # percentage

    # Extreme rainfall
    threshold = df['Precipitation'].quantile(0.95)  # extreme rainfall should be in the 95th percentile of rainfall
    df['ExtremeRainfall'] = df['Precipitation'] > threshold 

    # Target variable indicating rainfall based on both precipitation and humidity
    df['Rainfall'] = ((df['Precipitation'] > precipitation_threshold) | (df['Humidity'] > humidity_threshold)).astype(int)

    # Specify order
    df = df[['Year', 'Month', 'Day', 'MaxT', 'MinT', 'AvgT', 'WindSpeed', 'Humidity', 'Precipitation', 'Rainfall', 'ExtremeRainfall']]

    # Save cleaned dataframe to csv
    df.to_csv(FILE_ROOT + "cleaned_data.csv")
    print(df.info())

    # checking for days there are extreme rainfall
    print(df[df['ExtremeRainfall'] == True])
    df[df['ExtremeRainfall'] == True].to_csv(FILE_ROOT + "extreme_days.csv")


main()