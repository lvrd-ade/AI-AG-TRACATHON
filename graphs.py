import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from globals import *

df = pd.read_csv(FILE_ROOT + "cleaned_data.csv")

def main():
    x1, x2, x3, x4, x5 = 'Year', 'Month', 'Day', 'Rainfall', 'ExtremeRainfall'
    chosen_x = x1

    y1, y2, y3, y4 = 'AvgT', 'WindSpeed', 'Humidity', 'Precipitation'
    chosen_y  = y1

    #df['Date'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)

    # Create the heatmap
    #sns.heatmap(data=df, x='Date', y='Rainfall', cmap='coolwarm')
    #plt.show()


    #bar_graph(chosen_x, chosen_y)
    #bar_graph(y4, x2)
    pivot_df = df.pivot_table(index='Month', columns=['Day'], values='Precipitation', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, cmap='YlGnBu', cbar_kws={'label': 'Average Precipitation (mm)'})
    plt.title('Average Precipitation by Day and Month (2006-2023)')
    plt.xlabel('Month - Day')
    plt.ylabel('Year')
    plt.show()


def bar_graph(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y], color='blue', alpha=0.5)
    plt.title('Relationship between ' + x + ' and ' + y)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()

def heat_map(x, y):
    pivot_df = df.pivot_table(index='Year', columns=['Month', 'Day'], values='Precipitation', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, cmap='YlGnBu', cbar_kws={'label': 'Average Precipitation (mm)'})
    plt.title('Average Precipitation by Day and Month (2006-2023)')
    plt.xlabel('Month - Day')
    plt.ylabel('Year')
    plt.show()


main()