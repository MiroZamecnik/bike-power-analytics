import pandas as pd
import streamlit as st
#import numpy as np
from gpxcsv import gpxtolist
#import gpxpy
import matplotlib.pyplot as plt

bg_color = 'black'

def track_plot(x, y, start, end, bg_color=bg_color, place=st):
    p1, p2 = plt.subplots(facecolor = bg_color)
    p2.set_facecolor(bg_color)

    plt.grid(which='major', axis='both' ,linestyle = 'dashed', linewidth = 1)

    plt.xlabel(' km', fontweight='bold')
    plt.ylabel(' km', fontweight='bold')
    
    plt.scatter(df[x], df[y], marker = '.', color='dodgerblue')
    plt.scatter(df[x][start:end], df[y][start:end], marker = '.', color='red')
   
    plt.title(x+'   '+y, fontweight='bold', color = 'dodgerblue')
    return place.pyplot(p1,p2)

def profile_plot(y, start, end, bg_color=bg_color, place=st):
    p1, p2 = plt.subplots(facecolor = bg_color)
    p2.set_facecolor(bg_color)

    plt.grid(which='major', axis='both' ,linestyle = 'dashed', linewidth = 1)
    plt.ylabel(' m', fontweight='bold')
    
    plt.plot(df.index, df[y], linestyle='-', color='dodgerblue')
    plt.plot(df.index[start:end], df[y][start:end], linestyle='-', color='red')
   
    plt.title(y, fontweight='bold', color = 'dodgerblue')
    return place.pyplot(p1,p2)

st.title('Bike and power calculator')

uploaded_file = st.file_uploader("Upload GPX file", type=["gpx"], accept_multiple_files=False)
#uploaded_file = 1
if uploaded_file is not None:
    df = pd.DataFrame(gpxtolist(uploaded_file.name))
    d = df.describe()
    st.write(d.lon['min'], d.lat['min'])
    st.write(d)
    st.write(df)
    
   
    len_df = len(df) 
    track_slider = st.slider('**Seconds to consider?**  ', 0, len_df, (round(len_df/3), round(2*len_df/3)))
    profile_plot('ele', track_slider[0], track_slider[1],  bg_color=bg_color, place=st)
    #st.write(df)
    track_plot('lon', 'lat', track_slider[0], track_slider[1],  bg_color=bg_color, place=st)

