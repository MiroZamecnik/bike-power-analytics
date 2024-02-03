import gpxpy
import gpxpy.gpx
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

bg_color = 'black'
earth_radius = 6378 # in km
weight0 = 102    # rider's weight in kilograms
weight_bike0 = 11  # in kilogramg

weight_bike = st.sidebar.text_input('Weight of your bike? (kg)', weight_bike0)
weight = st.sidebar.text_input('Your weight? (kg)', weight0)



def miros_filter(df, col, new_col):
    df['delta'] = 0
    df['delta'] = round(df[col].shift(-1) - ((df[col]+df[col].shift(-2))/2), 3)
    df[new_col] = df[col] + df['delta']/6-df['delta'].shift()/3+df['delta'].shift(2)/6
    return df


def profile_plot(y, start, end, bg_color=bg_color, place=st):
    p1, p2 = plt.subplots(facecolor = bg_color)
    p2.set_facecolor(bg_color)

    plt.grid(which='major', axis='both' ,linestyle = 'dashed', linewidth = 1)
    plt.ylabel(' m', fontweight='bold')
    
    plt.plot(df.index, df[y], linestyle='-', color='dodgerblue')
    plt.plot(df.index[start:end], df[y][start:end], linestyle='-', color='red')
   
    plt.title(y, fontweight='bold', color = 'dodgerblue')
    return place.pyplot(p1,p2)

# user can upload a gpx file which is then stored as a df
uploaded_file = st.sidebar.file_uploader("Upload your GPX file", type=["gpx"], accept_multiple_files=False)
if uploaded_file is not None:
    gpx = gpxpy.parse(uploaded_file)

    route = []
    for track in gpx.tracks:
        for segment in track.segments:
            st.write(segment)
            for point in segment.points:
                #st.write(point)
                route.append({
                    'time': point.time,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'power':point.extensions[0].text
                })
    df = pd.DataFrame(route)
    #st.write(df)
    len_df = len(df)
    track_slider = st.slider('**Seconds to consider?**  ', 0, len_df, (round(len_df/7), round(6*len_df/7)))
 
    #st.map(df, latitude='latitude', longitude='longitude', size=0.1, color=[0,0,255])
    st.map(df[track_slider[0]:track_slider[1]], latitude='latitude', longitude='longitude', size=0.1, color=[250,0,0])
    profile_plot('elevation', track_slider[0], track_slider[1],  bg_color=bg_color, place=st)
    
    #st.area_chart(df, y="elevation")

    df['d_lat'] = df.latitude - df.latitude.shift()
    df['d_lon'] = df.longitude - df.longitude.shift()
    
    
    
    df = miros_filter(df, 'elevation', 'ele1')
    df = miros_filter(df, 'ele1', 'ele2')
    df = miros_filter(df, 'ele2', 'ele3')
    
    
    radius = earth_radius * 1000  # in metres
    # zvislo   df.d_lat/180*np.pi*radius
    # vodorovne   np.cos(df.lat)*df.d_lon/180*np.pi*radius
    df['d'] = np.pi*radius/180*(df.d_lat**2 + (np.cos(df.latitude)*df.d_lon)**2)**0.5
    df['dist'] = (df['d'].shift(1)+df['d']+df['d'].shift(-1)+df['d'].shift(-2))/4
    
    df = miros_filter(df, 'dist', 'dist1')
    df = miros_filter(df, 'dist1', 'dist2')
    df = miros_filter(df, 'dist2', 'dist3')
    df = miros_filter(df, 'dist3', 'dist4')
    df = miros_filter(df, 'dist4', 'dist5')
    df = miros_filter(df, 'dist5', 'dist6')
    df = miros_filter(df, 'dist6', 'dist7')
    
    
    df['d_ele'] = df['ele3'] - df['ele3'].shift()
    df['d_Ep'] = df['d_ele']*(float(weight) + float(weight_bike))*9.81
    df['Ek'] = (float(weight) + float(weight_bike))/2*df['dist7']**2
    
    a = df.loc[track_slider[0]:track_slider[1], ]#[(df.index>1000) & (df.index<2000)]#[(df.index > (3760-30)) & (df.index < (3760+20)) ]#& (df.power == 0)][['power', 'dist', 'ele', 'd_ele', 'E', 'Ek']]
    a['Ep'] = a['d_Ep'].cumsum()
    a['total_dist'] = a['dist7'].cumsum()
    a['dist_2'] = a['dist7'].shift(1) * a['dist7'] #a['#a['dist7']**2
    a['total_dist_2'] = a['dist_2'].cumsum()
    a['total_power'] = a['power'].cumsum()
    #a = a[['total_dist', 'total_dist2', 'E_cons', 'total_power']]
    #a.describe()
    #a = a.dropna()
    
    
    st.write(a)
    
    #a['Y'] = a['Ek']+a['Ep']+4.5*a['total_dist']-0.93*a['total_power']#+1*a['total_dist_2']#+a['total_power']  #- 5*a['total_dist']
    #Y = a[['Y']]
    #X = a[['total_dist_2']]#, 'total_dist_2', 'total_power']]
    #X = a[['total_dist_2']]
    
    
    #lin_regressor = LinearRegression()
    #lin_regressor = lin_regressor.fit(X, Y)
    #print(lin_regressor.intercept_, lin_regressor.coef_)
    








