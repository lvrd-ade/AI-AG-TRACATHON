import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from globals import *

df = pd.read_csv(FILE_ROOT + "cleaned_data.csv")

def main():
    x1, x2, x3, x4, x5 = 'Year', 'Month', 'Day', 'Rainfall', 'ExtremeRainfall'
    chosen_x = x1

    y1, y2, y3, y4 = 'AvgT', 'WindSpeed', 'Humidity', 'Precipitation'
    chosen_y  = y1

    #bar_graph(chosen_x, chosen_y)
    bar_graph(x5, y2)


def bar_graph(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y], color='blue', alpha=0.5)
    plt.title('Relationship between ' + x + ' and ' + y)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()



main()