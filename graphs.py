import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import calendar

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
    heat_map(df)



def bar_graph(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y], color='blue', alpha=0.5)
    plt.title('Relationship between ' + x + ' and ' + y)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()

def heat_map(df):
    # Daily Average Data
    df = pd.read_excel('Farm_Weather_Data.xlsx', index_col=None, header=0) 
    df.to_excel("Weather_Data.xlsx") 
    df['month'] = df.Date.dt.month
    df['Month']=df.Date.dt.month_name()
    df['year'] = df.Date.dt.year
    df['day'] = df.Date.dt.day
    df['weekdayName'] = df.Date.dt.day_name # df.date.dt.day_name() on Pandas 1.0
    df['weekday'] = df.Date.dt.weekday
    df['week'] = df.Date.dt.isocalendar().week
    df['weekend'] = df.Date.dt.weekday // 5 == 1
    all_days_avg_df = df.groupby([df.Date.dt.month, df.Date.dt.day])['Precipitation'].mean()
    all_days_avg_df = all_days_avg_df.unstack()
    all_days_avg_df = all_days_avg_df.set_index([[calendar.month_abbr[i] for i in list(all_days_avg_df.index)]])

    gyr = LinearSegmentedColormap.from_list(
    name='Divergent', 
    colors=['moccasin','papayawhip','khaki','yellow','greenyellow','lawngreen','mediumseagreen','forestgreen',\
            'darkgreen', 'salmon', 'orangered','firebrick','crimson']
    )


    plt.figure(figsize = (40, 14))
    ax = sns.heatmap(all_days_avg_df, cmap = gyr, annot=True, fmt='.0f',
                vmin=0, linewidths=.1,
                annot_kws={"size": 12}, square=True,  # <-- square cell
                cbar_kws={"shrink": .9, 'label': 'Rain (mm)'})
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 14)
    ax.tick_params(rotation = 0)
    _ = plt.title('Average Daily Precipitation (mm) 2006-2022', fontdict={'fontsize':18}, pad=14)
    plt.show()

main()