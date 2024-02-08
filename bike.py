import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.DataFrame()
if 'df' in st.session_state:
    df = st.session_state.df

#track_slider = {0: 100, 1: 200}#
slider_start, slider_end = 1000, 2000
if 'slider_start' in st.session_state:
    slider_start = st.session_state.slider_start
if 'slider_end' in st.session_state:  
    slider_end = st.session_state.slider_end

whole_file = str()
if 'whole_file' in st.session_state:
    whole_file = st.session_state.whole_file
    st.write('AHA')
    st.write(len(whole_file) + len(st.session_state.whole_file))

bg_color = 'white'
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


def profile_plot(df, y, start, end, bg_color=bg_color, place=st):
    fig, ax1 = plt.subplots(facecolor = 'grey')
    fig.set_facecolor('grey')

    plt.grid(which='major', axis='both' ,linestyle = 'dashed', linewidth = 1)
    lab = 'watts'
    if y[:3] == 'ele':
        lab = 'metres'
    plt.ylabel(lab, fontweight='bold', color='white')
    plt.plot(df.index, df[y], linestyle='-', color="#ddbb88")
    plt.plot()
   
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel(lab, color=color)
    ax1.plot(df.index[start:end], df[y][start:end], linestyle='-', color="#ffaa00")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('', color=color)  # we already handled the x-label with ax1
    ax2.plot(df.index[start:end], df['latitude'][start:end], linestyle='-', color="#ffaa00")
    ax2.tick_params(axis='y', labelcolor=color)
   
    
    plt.title(y, fontweight='bold', color = 'dodgerblue')
    fig.set_size_inches(6, 2)
    return place.pyplot(fig, ax1, ax2)

def gpx_message(item, label, where=st.sidebar):
    if d[item]['mean']==0:
        where.write(f':red[No {label} data] observed in GPX file :(') 
    else:
        where.write(f':green[{label} data] observed in GPX file :)')
        return
    
    
    
# user can upload a gpx file which is then stored as a df

#uploaded_file = st.sidebar.file_uploader("Upload your GPX file", type=["gpx"], accept_multiple_files=False)


button1 = st.sidebar.button('*Load r1*')
#on pressing left "Add" button, the text from the text_input will be included into the STOCKS list 
if button1:
    with open('r1.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read()) #toto dole
        

      
button2 = st.sidebar.button('*Load pekna8*')
if button2:
    with open('pekna8.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read())
             
        
button3 = st.sidebar.button('*Load OJ 2018*')
if button3:
    with open('oj.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read()) 
     
st.write(len(whole_file))                     
if len(whole_file)>100:
    st.session_state.whole_file = whole_file
    
    #whole_file = str(uploaded_file.read())
    times = whole_file.split('<time>')
    items = ['ele','power', 'hr', 'atemp', 'cad']
    string = dict()
    for i in items:
        st.write(f'{i} : {whole_file.count(i)}')

    counter = 0
    records = []
    t_string, lat_string, lon_string = '0', '0', '0'
    for part in times:
        #st.write(f'time : {part[11:20]}')
        t_string = part[11:20]
        for i in items:
            string[i]='0'
            if part.count('<'+i+'>')>0:
                string[i] = part[part.index('<'+i+'>')+len(i)+2:part.index('</'+i+'>')]
                #st.write(f'{i} : {string[i]}')
            elif part.count('gpxtpx:'+i)>0:
                start = part.index('gpxtpx:'+i)+len('gpxtpx:'+i)
                string[i] = part[start+1:start + part[start:].index('</')]
                #st.write(f'{i} : {string[i]}')
                
        if part.count('lat=\"')>0:
            lat_string = part[part.index('lat=\"')+5:part.index('lat=\"')+15]
            #st.write(f'LAT : {lat_string}')
        if part.count('lon=\"')>0:
            lon_string = part[part.index('lon=\"')+5:part.index('lon=\"')+15]
            #st.write(f'LON : {lon_string}')
        #st.write('----------------------------------------------')
        records.append({
            'time': t_string,
            'latitude': float(lat_string),
            'longitude': float(lon_string),
            'elevation': float(string['ele']),
            'power': int(string['power']),
            'hr': int(string['hr']),
            'temp': int(string['atemp']),
            'cad': int(string['cad'])
        })

                   
    df = pd.DataFrame(records)
    df = df[df.latitude!=0]
    df = df[df.elevation!=0]
    df = df.reset_index()
    len_df = len(df)
    st.write(df)
    d = df.describe()
    power_meter = d['power']['mean']!=0
    st.write(d)
    
    label = {'latitude':'latitude',
            'longitude':'longitude', 'elevation':'elevation', 'power':'power',
            'hr':'heart rate', 'temp':'temperature','cad':'cadence'}
    for item in ('latitude',
            'longitude', 'elevation','power',
            'hr', 'temp','cad'): 
        gpx_message(item, label[item], where=st.sidebar)

        
    if 'slider_start' not in st.session_state:
        slider_start = round(len_df/3)  
    if 'slider_end' not in st.session_state:
        slider_end = round(2*len_df/3) 
    track_slider = st.slider('**Seconds to consider?**  ', 0, len_df, (slider_start, slider_end))# (round(len_df/7), round(6*len_df/7)))
    st.session_state.slider_start = slider_start = track_slider[0]
    st.session_state.slider_end = slider_end = track_slider[1]
    
    #st.map(df, latitude='latitude', longitude='longitude', size=0.1, color=[0,0,255])
   
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
    
    
    #df['Ep'] = df['d_Ep'].cumsum()
    df['one'] = 1
    df['ones'] = df['one'].cumsum()
    #df['total_dist'] = df['dist7'].cumsum()
    df['dist_2'] = df['dist7'].shift(1) * df['dist7'] #a['#a['dist7']**2
    df['Ek'] = (float(weight) + float(weight_bike))/2*df['dist_2']
    #df['total_dist_2'] = df['dist_2'].cumsum()
    
    #a = df.loc[track_slider[0]:track_slider[1], ]#[(df.index>1000) & (df.index<2000)]#[(df.index > (3760-30)) & (df.index < (3760+20)) ]#& (df.power == 0)][['power', 'dist', 'ele', 'd_ele', 'E', 'Ek']]
  
    #if power_meter: a['total_power'] = a['power'].cumsum()
    #st.write(a[['power', 'dist_2']])
    #a = a[['total_dist', 'total_dist2', 'E_cons', 'total_power']]
    #a.describe()
    #a = a.dropna()
    
    #a['Y'] = a['Ek']+a['Ep']#+4.5*a['total_dist']-0.93*a['total_power']#+1*a['total_dist_2']#+a['total_power']  #- 5*a['total_dist']
    #Y = a[['Y']]
    #if power_meter:
    #    X = a[['total_dist', 'total_dist_2', 'total_power']]
    #else:
    #    X = a[['total_dist', 'ones']]# a[['total_dist', 'total_dist_2', 'ones']]
    #X = a[['total_dist_2']]

    profile_plot(df, 'elevation', max(0, round(0.8*slider_start)), min(len(df), round(1.2*slider_end)),  bg_color=bg_color, place=st)
    df['color'] = "#ddbb88" 
    #st.write(df)
    df['color'][slider_start:slider_end] = "#ffaa00"
    st.map(df, latitude='latitude', longitude='longitude', size = 0.2, color='color')
   
    if not power_meter:
        st.write('Without powermeter data you can estimate your average power or resistance coefficients:')
        c1, c2, c3 = st.columns((2,3,3))
        no_pm = c1.selectbox('I want to estimate:', ('none', 'power', 'resistance coefficients - no pedaling'), 0)
        if no_pm == 'power':
            tr_str = c2.text_input('Enter tire resistance coeff (in promiles):', '4')
            ar_str = c3.text_input('Enter air resistance coeff (in BLABLA):', '2')
            ar = float(ar_str)
            tr = float(tr_str)
            df['d_Ek'] = df['Ek'] - df['Ek'].shift(1)
            df['power estimate0'] = df['d_Ek'] + df['d_Ep'] + (df['dist7'] * tr /1000 * (float(weight_bike) + float(weight))) + (df['dist_2'] * ar)
            
            df['delta_E0'] = np.where(df['power estimate0'] < 0, -df['power estimate0'], 0)
            df['power estimate1'] = df['d_Ek'] + df['d_Ep'] + (df['dist7'] * tr /1000 * (float(weight_bike) + float(weight))) + (df['dist_2'] * ar)-df['delta_E0'].shift(-1)
            df['delta_E1'] = np.where(df['power estimate1'] < 0, -df['power estimate1'], 0)
            df['power estimate2'] = df['d_Ek'] + df['d_Ep'] + (df['dist7'] * tr /1000 * (float(weight_bike) + float(weight))) + (df['dist_2'] * ar)-df['delta_E1'].shift(-1)
            
            df = miros_filter(df, 'power estimate2', 'power estimate3')
            df = miros_filter(df, 'power estimate3', 'power estimate4')
            df = miros_filter(df, 'power estimate4', 'power estimate5')
            df['power estimate6'] = (df['power estimate5'].shift(-2)+df['power estimate5'].shift(-1)+df['power estimate5']+df['power estimate5'].shift(1)+df['power estimate5'].shift(2))/5
            df['power estimate7'] = (df['power estimate6'].shift(-2)+df['power estimate6'].shift(-1)+df['power estimate6']+df['power estimate6'].shift(1)+df['power estimate6'].shift(2))/5
            
            df['power estimate'] = np.where(df['power estimate7'] < 0, 0, df['power estimate7'])
            
            
            profile_plot(df[track_slider[0]:track_slider[1]],'power estimate',0,len(df)-1, bg_color=bg_color, place=st)
            #st.write(df[track_slider[0]:track_slider[1]][['power estimate0', 'power estimate']].describe())
        if no_pm == 'resistance coefficients - no pedaling':
            c2.write('Please select part of your route without pedaling (climb descend or so). ')
            c3.write(' We will try tom estimate both resistance coefficients - linear/tire one and qvadratic/air one.')
            lin_regressor = LinearRegression()
            a = df[track_slider[0]:track_slider[1]]
            a['Ep'] = a['d_Ep'].cumsum()
            Y = -a['Ep'] - a['Ek']
            a['total_dist'] = a['dist7'].cumsum()
            a['tire'] = a['total_dist'] * (float(weight) + float(weight_bike))* 9.81
            a['total_dist_2'] = a['dist_2'].cumsum()
            X = a[['tire', 'total_dist_2']]
            lin_regressor = lin_regressor.fit(X, Y)
    #st.write(X.columns)
            c1.write('intercept:')
            c1.write(lin_regressor.intercept_)
            c3.write(f'linear/tire coefficient is {-1000 * lin_regressor.coef_[0]}')
            c3.write(f'quadratic/air coefficient is {-lin_regressor.coef_[1]}')
            #st.write(a[['Ep','ele3','Ek']])
            st.write(Y, lin_regressor.predict(X))
            st.write((Y-lin_regressor.predict(X)).describe())
            lin_str = st.text_input('Enter tire resistance coeff (in promiles):')
            
            #if lin_str != '':
            #    Y = -a['Ep'] - a['Ek'] - float(lin_str) * (a['tire'])
            #    X = a[['total_dist_2']]
            #    st.write(f'quadratic/air coefficient is {-lin_regressor.coef_[0]}')
    
    #profile_plot('elevation', track_slider[0], track_slider[1],  bg_color=bg_color, place=st)
            
            if df not in st.session_state:
                st.session_state.df = df

            








