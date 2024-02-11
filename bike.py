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
    #st.write('AHA')
    #st.write(len(whole_file) + len(st.session_state.whole_file))

bg_color = 'white'
earth_radius = 6378 # in km
weight0 = 102    # rider's weight in kilograms
weight_bike0 = 11  # in kilogramg

left, right = st.sidebar.columns(2)
weight_bike = left.text_input('Weight of your bike? (kg)', weight_bike0)
bike_type = right.selectbox('Type of your bike:', ('road', 'trekking/gravel', 'MTB'), 0)
tire_resistance = (bike_type=='road')*5 + (bike_type=='trekking/gravel')*6 + (bike_type=='MTB')*8
weight = st.sidebar.text_input('Your weight? (kg)', weight0)

#@st.cache_data
def miros_filter(df, col, new_col):
    df['delta'] = 0
    df['delta'] = round(df[col].shift(-1) - ((df[col]+df[col].shift(-2))/2), 3)
    df[new_col] = df[col] + df['delta']/6-df['delta'].shift()/3+df['delta'].shift(2)/6
    return df


def profile_plot(df, start, end, left, right, left_color, right_color, bg_color=bg_color, place=st):
    fig, ax1 = plt.subplots(facecolor = bg_color)
    fig.set_facecolor(bg_color)

    plt.grid(which='major', axis='both' ,linestyle = 'dashed', linewidth = 1)

    plt.plot(df.index[start:end], df[left][start:end], linestyle='-', color=left_color)
    
    plt.plot()
   
    
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel(left, color=left_color)
    ax1.plot(df.index[slider_start:slider_end], df[left][slider_start:slider_end], linestyle='-', color=left_color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = right_color
    ax2.set_ylabel(right, color=right_color)  # we already handled the x-label with ax1
    plt.plot(df.index[start:end], df[right][start:end], linestyle='-', color=right_color)
    ax2.fill_between(df.index[slider_start:slider_end], df[right][slider_start:slider_end]-df[right][slider_start:slider_end], df[right][slider_start:slider_end], alpha = 0.7, color='grey')
    ax2.tick_params(axis='y', labelcolor=right_color)
    ax1.set_ylabel(left, color=left_color)
   
    
    plt.title(right +' vs. '+left, fontweight='bold', color = 'black')
    fig.set_size_inches(6, 2)
    return place.pyplot(fig, ax1, ax2)

def gpx_message(item, label, where=st.sidebar):
    if d[item]['mean']==0:
        where.write(f':red[No {label} data] observed in GPX file :(') 
    else:
        where.write(f':green[{label} data] observed in GPX file :)')
        return
    
def power_estimate(df, tr, ar):
    '''
    Parameters
    ----------
    df : input datafraame
    tr : tire resistance - in promiles, 5 means rolling power coefficient of 0.005
    ar : air resistance - nominal, is equal to F/velocity-quared, cca 0.2 for a orad cyclist

    Returns
    -------
    dataframe df with the 'power estimate' column

    '''
    df['d_Ek'] = df['Ek'] - df['Ek'].shift(1)
    df['friction'] = (9.81 * df['dist7'] * tr /1000 * (float(weight_bike) + float(weight))) + (df['dist_2'] * ar * df['dist7'])
    df['power estimate0'] = df['d_Ek'] + df['d_Ep'] + df['friction']
    
    df['delta_E0'] = np.where(df['power estimate0'] < 0, -df['power estimate0'], 0)
    df['power estimate1'] = df['d_Ek'] + df['d_Ep'] + df['friction']-df['delta_E0'].shift(-1)
    df['delta_E1'] = np.where(df['power estimate1'] < 0, -df['power estimate1'], 0)
    df['power estimate2'] = df['d_Ek'] + df['d_Ep'] + df['friction']-df['delta_E1'].shift(-1)
    
    df = miros_filter(df, 'power estimate2', 'power estimate3')
    df = miros_filter(df, 'power estimate3', 'power estimate4')
    df = miros_filter(df, 'power estimate4', 'power estimate5')
    df['power estimate6'] = (df['power estimate5'].shift(-2)+df['power estimate5'].shift(-1)+df['power estimate5']+df['power estimate5'].shift(1)+df['power estimate5'].shift(2))/5
    df['power estimate7'] = (df['power estimate6'].shift(-2)+df['power estimate6'].shift(-1)+df['power estimate6']+df['power estimate6'].shift(1)+df['power estimate6'].shift(2))/5 
    df['power estimate'] = np.where(df['power estimate7'] < 0, 0, df['power estimate7'])
    return df

    
# user can upload a gpx file which is then stored as a df

uploaded_file = st.sidebar.file_uploader("Upload your GPX file", type=["gpx"], accept_multiple_files=False)
if uploaded_file is not None:
    whole_file = st.session_state.whole_file = str(uploaded_file.read())

button1 = st.sidebar.button('*Load JuRaVa*')
#on pressing left "Add" button, the text from the text_input will be included into the STOCKS list 
if button1:
    with open('JuRaVa.gpx', 'r') as uploaded_file:   #toto prec
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
    date = times[3][0:10]
    for part in times:
        #st.write(f'time : {part[11:20]}')
        t_string = part[11:19]
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
    #st.write(df)
    d = df.describe()
    power_meter = d['power']['mean']!=0
    #st.write(d)
    
    label = {'latitude':'latitude',
            'longitude':'longitude', 'elevation':'elevation', 'power':'power',
            'hr':'heart rate', 'temp':'temperature','cad':'cadence'}
    for item in ('latitude',
            'longitude', 'elevation','power',
            'hr', 'temp','cad'): 
        gpx_message(item, label[item], where=st.sidebar)

# SUMMARY
    df['d_lat'] = df.latitude - df.latitude.shift()
    df['d_lon'] = df.longitude - df.longitude.shift() 
    df = miros_filter(df, 'elevation', 'ele1')
    df = miros_filter(df, 'ele1', 'ele2')
    df = miros_filter(df, 'ele2', 'ele3')
    radius = earth_radius * 1000  # in metres
    # verticaly   df.d_lat/180*np.pi*radius
    # horizontaly   np.cos(df.lat)*df.d_lon/180*np.pi*radius
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
    #df = power_estimate(df, tire_resistance, 0.2)
    df['dist5sec'] = df['dist7'].shift(-2) + df['dist7'].shift(-1) + df['dist7'] + df['dist7'].shift(1) + df['dist7'].shift(2)
    df['slope'] = (df['dist5sec']>8)*(df['d_ele'].shift(-2) + df['d_ele'].shift(-1) + df['d_ele'] + df['d_ele'].shift(1) + df['d_ele'].shift(2) )/df['dist5sec']  
    
    
    d = df.describe(percentiles = [.1,.25,.5])
    st.write(f'Your route from **{date}** started at **{df["time"][df.index.min()]}** and ended at **{df["time"][df.index.max()]}**.')
    st.write()
    st.write(f"Your elevation ranged between {round(d['elevation']['min'])}m and {round(d['elevation']['max'])}m above see level and you overall gained **{round(df[df['d_ele']>0]['d_ele'].sum())}m**.")
    st.write(f"Steepest ascend was **{round(100*d['slope']['max'],1)}%** and steepest descend was **{-round(100*d['slope']['min'],1)}%**.")
    st.write()
    if 0 != d['temp']['max'] or d['temp']['min'] !=0:
        st.write(f'Temperature ranged between **{round(d["temp"]["10%"])}** and **{round(d["temp"]["max"])}** degrees with median value of **{round(d["temp"]["50%"])}** degrees.')
    if 0 != d['hr']['max']:
        st.write(f"Your heart rate ranged between {d['hr']['min']}bpm and {d['hr']['max']}bpm with median value of {d['hr']['50%']}bpm.")
    if 0 != d['cad']['max']:
        st.write(f"Your cadence ranged between **{round(d['cad']['10%'])}rpm** and **{round(d['cad']['max'])}rpm** with median value of **{round(d['cad']['50%'])}rpm**.")

    st.write(f"Your speed ranged between **{round(3.6*d['dist7']['10%'],1)}km/h** and **{round(3.6*d['dist7']['max'],1)}km/h** with median value of **{round(3.6*d['dist7']['50%'],1)}km/h** (average **{round(3.6*d['dist7']['mean'],1)}km/h**).")
    st.write()
    if power_meter:
        st.write(f"Your average power was **{round(d['power']['mean'])} watts**, when pedalling even **{round(df[df['power']>0]['power'].mean())} watts**.")
    else:
        df = power_estimate(df, tire_resistance, 0.2)
        #st.write(df[200:550])
        st.write(f"Your average power is estimated to **{round(df['power estimate'].mean())} watts**, when pedalling even **{round(df[df['power estimate']>0]['power estimate0'].mean())} watts**.")
    #st.write(d)
    
    st.write('-----------------------------------------------------')
    
    
        
    if 'slider_start' not in st.session_state:
        slider_start = round(len_df/3)  
    if 'slider_end' not in st.session_state:
        slider_end = round(2*len_df/3)
    slider_start, slider_end = 356, 539 #1110, 1220
    track_slider = st.slider('**Seconds to consider?**  ', 0, len_df, (slider_start, slider_end))# (round(len_df/7), round(6*len_df/7)))
    st.session_state.slider_start = slider_start = track_slider[0]
    st.session_state.slider_end = slider_end = track_slider[1]
    
    #st.map(df, latitude='latitude', longitude='longitude', size=0.1, color=[0,0,255])
   
    #st.area_chart(df, y="elevation")


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
    
    # PLOTS
    dd = df.describe(percentiles = [.1, .9])
    variable = st.selectbox('What to color on a map?', ('speed', 'elevation', 'slope', 'cadence'),0)
    min_value = dd['dist7']['10%']
    max_value = dd['dist7']['90%']
    var_range = (max_value - min_value)
    if variable == 'speed':
        df['dist7'] = df['dist7'].fillna(0)
        df['rel'] = round((df['dist7']-min_value)/var_range,1)
        df['red'] = (df['rel']-0.5)/0.25
        df['red'] = round(255*(df['red']))
        df['red'] = np.where(df['red'] < 0, 0, df['red'])
        df['red'] = np.where(df['red'] > 255, 255, df['red'])
        #df['red'] = 255
        
        df['green'] = np.where(df['rel']<0.25, df['rel']/0.25,1)
        df['green'] = np.where(df['rel']>0.75, (1-df['rel'])/0.25,  df['green'])
        df['green'] = round(255*(df['green']))
        df['green'] = np.where(df['green'] < 0, 0, df['green'])
        df['green'] = np.where(df['green'] > 255, 255, df['green'])
        #df['green'] = 0
        
        df['blue'] = (0.5-df['rel'])/0.25
        df['blue'] = np.where(df['rel']<0.25, 1, df['blue'])
        df['blue'] = np.where(df['rel']>0.5, 0, df['blue'])
        df['blue'] = round(255*(df['blue']))
        df['blue'] = np.where(df['blue'] < 0, 0, df['blue'])
        df['blue'] = np.where(df['blue'] > 255, 255, df['blue'])
        #df['blue'] = 0
        df['color speed'] = list(zip(df.red/255, df.green/255, df.blue/255))#(df['dist7']-min_value)*/var_range
    
    st.write(df[['red','green','blue','rel']])
    dff = df[100:150]
    dff = dff.reset_index()
    st.map(dff, latitude='latitude', longitude='longitude', size = 0.2, color='color speed')
    
    profile_plot(df, max(0, round(0.8*slider_start)), min(len(df), round(1.2*slider_end)), 'elevation', 'dist7', 'green', 'red',  bg_color=bg_color, place=st)
    df['color'] = "#ddbb88" 
    #st.write(df)
    df['color'][slider_start:slider_end] = "#ffaa00"
    
   
    if not power_meter:
        st.write('Without powermeter data you can estimate your average power or resistance coefficients:')
        c1, c2, c3 = st.columns((2,3,3))
        no_pm = c1.selectbox('I want to estimate:', ('none', 'power', 'resistance coefficients - no pedaling'), 0)
        if no_pm == 'power':
            tr_str = c2.text_input('Enter tire resistance coeff (in promiles):', '4')
            ar_str = c3.text_input('Enter air resistance coeff (in BLABLA):', '0.2')
            ar = float(ar_str)
            tr = float(tr_str)
            df = power_estimate(df, tr, ar)
            #######
            
            
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
            a['air'] = a['total_dist_2'] * a['dist7'] 
            X = a[['tire', 'air']]
            lin_regressor = lin_regressor.fit(X, Y)
    #st.write(X.columns)
            c1.write('intercept:')
            c1.write(lin_regressor.intercept_)
            c3.write(f'linear/tire coefficient is {round(-1000 * lin_regressor.coef_[0], 2)}')
            c3.write(f'quadratic/air coefficient is {-round(lin_regressor.coef_[1], 2)}')
            #st.write(a[['Ep','ele3','Ek']])
            #st.write(Y, lin_regressor.predict(X))
            #st.write((Y-lin_regressor.predict(X)).describe())
            lin_str = st.text_input('Enter tire resistance coeff (in promiles):','-0.00659')
            
            if lin_str != '':
                Y = -a['Ep'] - a['Ek'] + float(lin_str) * (a['tire'])
                X = a[['air']]
                lin_regressor = lin_regressor.fit(X, Y)
                st.write(f'quadratic/air coefficient is {-round(lin_regressor.coef_[0], 2)}')
    
    #profile_plot('elevation', track_slider[0], track_slider[1],  bg_color=bg_color, place=st)
            
            if df not in st.session_state:
                st.session_state.df = df

            








