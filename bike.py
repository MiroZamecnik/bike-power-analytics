import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(layout="wide")
primaryColor="#f3ee3b"
bg_color = backgroundColor="#372f33"
secondaryBackgroundColor="#8FCB42"
textColor="#FFFDFD"

df = pd.DataFrame()
if 'df' in st.session_state:
    df = st.session_state.df

slider_start, slider_end = 1000, 2000
if 'slider_start' in st.session_state:
    slider_start = st.session_state.slider_start
if 'slider_end' in st.session_state:  
    slider_end = st.session_state.slider_end

whole_file = str()
if 'whole_file' in st.session_state:
    whole_file = st.session_state.whole_file

earth_radius = 6378 # radius of the Earth in km used for the calculation of distances
weight0 = 100    # rider's weight in kilograms - initial value for text input field
weight_bike0 = 11  # bike's weight in kilograms - initial value for text input field

left, right = st.sidebar.columns((3,2))
weight_bike = left.text_input(':black[Weight of your bike? (kg)]', weight_bike0)
bike_type = right.selectbox('Type of your bike:', ('road', 'trekking/gravel', 'MTB'), 0)
tyre_resistance = (bike_type=='road')*4.2 + (bike_type=='trekking/gravel')*7 + (bike_type=='MTB')*12
air_resistance = (bike_type=='road')*.185 + (bike_type=='trekking/gravel')*.23 + (bike_type=='MTB')*.25
weight = st.sidebar.text_input('Your weight? (kg)', weight0)

@st.cache_data
def enrich_gpx(df):
    #new_df=pd.DataFrame(columns = df.columns)
    for index, row in df.iterrows():
        if index==0: 
            prev_row = row
            prev_index = index
            
        elif prev_row['dt']>1 and prev_row['dt']<100:
            dt = prev_row['dt']
            for i in range(1, int(dt)):
                add_row = prev_row.copy()
                if dt<20: 
                    for col in cols:
                        if col in ('time_no','latitude','longitude','elevation','power','heart rate','temperature','cadence','time_no'):  
                            add_row[col]= prev_row[col] + (i*(row[col]-prev_row[col])/dt)
                else:
                    add_row['time_no']= prev_row['time_no'] + (i*(row['time_no']-prev_row['time_no'])/dt)           
                df.loc[prev_index+(i/dt)]=add_row
        prev_row = row
        prev_index = index
    #df = df.reset_index('aha')
    #df['new_index'] = df.index
    df = df.sort_values(by='time_no')
    return df

def power_curve(pt, where_table, where_plot, show_yn=True):
    '''

    Parameters
    ----------
    pt : 'power' if real powermeter data are available in GPX file
        'power estimate' if the power is only estimated with power_estimate function

    Returns
    -------
    power curve - dataframe object with corresponding HR if available
    will print POWER (POWER ESTIMATE) CURVE in the form of dataframe to ST.

    '''
    
    df[pt+' 5sec'] = (df[pt] + df[pt].shift(-1) + df[pt].shift(-2) + df[pt].shift(-3) + df[pt].shift(-4))/5
    df[pt+' 20sec'] = (df[pt+' 5sec'] + df[pt+' 5sec'].shift(-5) + df[pt+' 5sec'].shift(-10) + df[pt+' 5sec'].shift(-15))/4
    df[pt+' 1min'] = (df[pt+' 20sec'] + df[pt+' 20sec'].shift(-20) + df[pt+' 20sec'].shift(-40))/3
    counter = 3
    if len(df)> 320:
        counter += 1
        df[pt+' 5min'] = (df[pt+' 1min'] + df[pt+' 1min'].shift(-60)+ df[pt+' 1min'].shift(-120)+ df[pt+' 1min'].shift(-180)+ df[pt+' 1min'].shift(-240))/5
    if len(df) > 1205:
        counter += 1
        df[pt+' 20min'] = (df[pt+' 5min'] + df[pt+' 5min'].shift(-300) + df[pt+' 5min'].shift(-600) + df[pt+' 5min'].shift(-900))/4
    if len(df)> 3700:
        counter += 1
        df[pt+' 1hour'] = (df[pt+' 20min'] + df[pt+' 20min'].shift(-1200) + df[pt+' 20min'].shift(-2400))/3
    
    list_power = df.columns[df.columns.get_loc(pt+' 5sec'):df.columns.get_loc(pt+' 5sec')+counter].to_list()
    power_desc = df[list_power].describe()
    pom = power_desc.transpose()[['max', 'min']]
    pom['POWER CURVE'] = 'max ' + pom.index
    pom['HR avg'] = 0
    pom['HR max'] = 0
    dict_inc = {pt+' 5sec':5, pt+' 20sec':20, pt+' 1min':60, pt+' 5min':300, pt+' 20min':1200, pt+' 1hour':3600}

    for col in df[list_power]:
        i = df[df[col]==power_desc[col]['max']].index[0]
        
        if heart_rate:
            pom['HR avg'][col]=round(df[i:i+dict_inc[col]]['heart rate'].mean())
            pom['HR max'][col]=df[i:i+dict_inc[col]]['heart rate'].max()
        df['interval'+col[-5:]] = 0
        df.loc[(df.index > i) & (df.index < i+dict_inc[col]), 'interval'+col[-5:]] = 1
        
    pom['watts'] = round(pom['max'])
    if show_yn:
        if pom['HR max'].max()>0:
            where_table.dataframe(pom[['POWER CURVE', 'watts', 'HR avg','HR max']], hide_index=True)
        else: where_table.dataframe(pom[['POWER CURVE', 'watts']], hide_index=True)
    return pom

def show_power_curve(power_curve, where):
    fig, ax1 = plt.subplots(facecolor = bg_color)
    ax1.set_facecolor(bg_color)

    list_times = ['5 seconds', '20 seconds', '1 minute', '5 minutes', '20 minutes', '1 hour']
    list_times = list_times[:len(power_curve)]
    
    plt.scatter(list_times, power_curve.watts,  marker='.', color=primaryColor, s=30)
    ax1.spines['bottom'].set_color(textColor)
    ax1.spines['top'].set_color(textColor)
    ax1.spines['right'].set_color(textColor)
    ax1.spines['left'].set_color(primaryColor)      
    
    ax1.tick_params(axis='y', labelcolor=primaryColor)
    ax1.tick_params(axis='x', labelcolor=textColor, labelrotation = 90)
    ax1.set_ylabel('watts', color=primaryColor) 
    ax1.xaxis.label.set_color(primaryColor)
    plt.plot([str(i) for i in list_times], power_curve.watts, linestyle='dashed', color = primaryColor)

    if power_curve['HR max'].max()>0: 
        ax2 = ax1.twinx()  # initiate second axes that shares the same x-axis
        plt.scatter(list_times, power_curve['HR avg'],  marker='.', color=secondaryBackgroundColor, s=30)
        ax2.set_ylabel('heart rate avg\n (bpm)', color=secondaryBackgroundColor)  # we already handled the x-label with ax1)
    #ax2.fill_between(df.index[slider_start:slider_end], df[right][slider_start:slider_end]-df[right][slider_start:slider_end], df[right][slider_start:slider_end], alpha = 0.4, color='white')
        ax2.tick_params(axis='y', labelcolor=secondaryBackgroundColor)
        ax2.spines['bottom'].set_color(textColor)
        ax2.spines['top'].set_color(textColor)
        ax2.spines['right'].set_color(secondaryBackgroundColor)
        ax2.spines['left'].set_color(primaryColor)
        ax2.xaxis.label.set_color(secondaryBackgroundColor)
        plt.plot([str(i) for i in list_times], power_curve['HR avg'], linestyle='dashed', marker='s', color = secondaryBackgroundColor)
    plt.grid(which='major', axis='both' ,linestyle = 'dashed', linewidth = 1, alpha = 0.4)
    fig.set_size_inches(4, 3)
    return where.pyplot(fig, ax1)
    
    
    return 

def miros_filter(df, col, new_col):
    '''
    A basic filter serving for smoothing of valuesc - from gpx files.
    Parameters
    ----------
    df : input pandas dataframe
    col : column of 'df' dataframe on which the filter is to be applied
    new_col : new column - filtered values of the 'col' column

    Returns
    -------
    df : original dataframe with the new column 

    '''
    df['delta'] = 0
    df['delta'] = round(df[col].shift(-1) - (df[col]+df[col].shift(-2))/2, 3)
    df[new_col] = df[col] + df['delta']/6-df['delta'].shift()/3+df['delta'].shift(2)/6
    return df

def profile_plot(df, start, end, left, right, left_color, right_color, bg_color=bg_color, place=st):
    fig, ax1 = plt.subplots(facecolor = bg_color)
    ax1.set_facecolor(bg_color)

    plt.grid(which='major', axis='both' ,linestyle = 'dashed', linewidth = 1, alpha = 0.4)

    plt.plot(df.index[start:end], df[left][start:end], linestyle='-', color=left_color)
      
    ax1.set_xlabel('time (s)', color=textColor)
    ax1.set_ylabel(left+'\n('+dict_units[left]+')', color=left_color)
    ax1.plot(df.index[slider_start:slider_end], df[left][slider_start:slider_end], linestyle='-', color=left_color)
    ax1.tick_params(axis='y', labelcolor=left_color)
    ax1.tick_params(axis='x', labelcolor=textColor)
    if left == 'slope' : fig.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax2 = ax1.twinx()  # initiate second axes that shares the same x-axis

    color = right_color
    ax2.set_ylabel(right+'\n('+dict_units[right]+')', color=right_color)  # we already handled the x-label with ax1)
    plt.plot(df.index[start:end], df[right][start:end], linestyle='-', color=right_color)
    ax2.fill_between(df.index[slider_start:slider_end], df[right][slider_start:slider_end]-df[right][slider_start:slider_end], df[right][slider_start:slider_end], alpha = 0.4, color='white')
    ax2.tick_params(axis='y', labelcolor=right_color)
    ax2.set_facecolor(bg_color)
    if right == 'slope' : fig.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax1.spines['bottom'].set_color(textColor)
    ax1.spines['top'].set_color(textColor)
    ax1.spines['right'].set_color(textColor)
    ax1.spines['left'].set_color(textColor)
    ax2.spines['bottom'].set_color(textColor)
    ax2.spines['top'].set_color(textColor)
    ax2.spines['right'].set_color(textColor)
    ax2.spines['left'].set_color(textColor)
    fig.set_size_inches(6, 3)
    return place.pyplot(fig, ax1, ax2)

def heatmap(variable, min_value, max_value, units, place=st):
    fig, ax = plt.subplots(facecolor = bg_color)
    fig.set_facecolor(bg_color)
    plt.grid(which='major', axis='x' ,linestyle = '-', linewidth = 0.1)
    for i in range(101):
        rel = i/100
        if i<25:
            r,g,b=0,i/25,1
        elif i<=50:
            r,g,b  =0,1,(50-i)/25
        elif i<=75:
            r,g,b=(i-50)/25,1,0
        else:
            r,g,b=1,(100-i)/25,0
        
        plt.scatter(10*[min_value+i*(max_value-min_value)/100],[0,1,2,3,4,5,6,7,8,9],marker='.', color=(r, g, b))

    plt.yticks([])
    if variable == 'slope' : fig.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis='x', colors=textColor)
    ax.set_xlabel(units, color=primaryColor, weight='bold', size = 15)
    ax.xaxis.set_label_position('top')
    plt.xlim(min_value, max_value)
    fig.set_size_inches(6, 0.3)
    return place.pyplot(fig, ax)
    
def gpx_message(item, label, where=st.sidebar):
    if d[item]['mean']==0:
        where.write(f'No {label} data observed in GPX file :neutral_face:') 
    else:
        where.write(f'**{label}** observed in GPX file :ok_hand:')
        return
    
def power_estimate(df, tr, ar):
    '''
    Parameters
    ----------
    df : input datafraame
    tr : tyre resistance - in promiles, 5 means rolling power coefficient of 0.005
    ar : air resistance - nominal, is equal to F/velocity-quared, cca 0.2 for a road cyclist

    Returns
    -------
    dataframe 'df' with the 'power estimate' column containing 
    the estimated (non-negative) value of cyclists' power in watts

    '''
    df['d_Ek'] = df['Ek'] - df['Ek'].shift(1)
    df['friction'] = (9.81 * df['dist7'] * tr /1000 * (float(weight_bike) + float(weight))) + (df['dist_2'] * ar * df['dist7'])
    df['power estimate0'] = df['d_Ek'] + df['d_Ep'] + df['friction']
    df['power estimate'] = np.where(df['power estimate0'] < 0, 0, df['power estimate0'])
    df['power estimate'] = df['power estimate'].fillna(0)
    df['power estimate'] = np.where(df['power estimate'] > 1000, 1000, df['power estimate'])
    return df
    
# user can upload a gpx file which is then stored as a df
st.sidebar.write('**--------------------------------**')

uploaded_file = st.sidebar.file_uploader("Upload your GPX file", type=["gpx"], accept_multiple_files=False)
if uploaded_file is not None:
    whole_file = st.session_state.whole_file = str(uploaded_file.read())

st.sidebar.subheader("Or pick up one of Miro's GPX records:")
left_b,right_b = st.sidebar.columns(2)
button1 = left_b.button('**OJ 2018**')
#on pressing left "Add" button, the text from the text_input will be included into the STOCKS list 
if button1:
    with open('OJ 2018.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read()) #toto dole
              
button2 = right_b.button('**OJ 2020**')
if button2:
    with open('OJ 2020.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read())
             
        
button3 = left_b.button('**OJ 2022**')
if button3:
    with open('OJ 2022.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read()) 

button4 = right_b.button('**Slnava**')
if button4:
    with open('Slnava.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read()) 

button5 = left_b.button('**NM PK**')
if button5:
    with open('NM PK.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read())

button6 = right_b.button('**20 min FTP test**')
if button6:
    with open('20minFTP.gpx', 'r') as uploaded_file:   #toto prec
        whole_file = st.session_state.whole_file = str(uploaded_file.read())
                   
if len(whole_file)>100:
    st.session_state.whole_file = whole_file
    times = whole_file.split('<time>')
    items = ['ele','power', 'hr', 'atemp', 'cad']
    string = dict()
    counter = 0
    records = []
    t_string, lat_string, lon_string = '0', '0', '0'
    date = times[3][0:10]
    for part in times:
        t_string = part[11:19]
        for i in items:
            string[i]='0'
            if part.count('<'+i+'>')>0:
                string[i] = part[part.index('<'+i+'>')+len(i)+2:part.index('</'+i+'>')]
            elif part.count('gpxtpx:'+i)>0:
                start = part.index('gpxtpx:'+i)+len('gpxtpx:'+i)
                string[i] = part[start+1:start + part[start:].index('</')]
                
        if part.count('lat=\"')>0:
            lat_string = part[part.index('lat=\"')+5:part.index('lat=\"')+15]
        if part.count('lon=\"')>0:
            lon_string = part[part.index('lon=\"')+5:part.index('lon=\"')+15]
        records.append({
            'time': t_string,
            'hour': (t_string[:2]),
            'min': (t_string[3:5]),
            'sec': (t_string[6:8]),
            'latitude': float(lat_string),
            'longitude': float(lon_string),
            'elevation': float(string['ele']),
            'power': float(string['power']),
            'heart rate': float(string['hr']),
            'temperature': float(string['atemp']),
            'cadence': float(string['cad'])
        })
                   
    df = pd.DataFrame(records)
    df = df[df.latitude!=0]
    df = df[df.elevation!=0]
    df = df.reset_index()
    
    len_df = len(df)
    df['time_no'] = 3600 * df['hour'].apply(int) + 60 * df['min'].apply(int) + df['sec'].apply(int)
    start_no = df[['time_no']].describe()['time_no']['min']
    df['time_no'] = df['time_no'] - start_no + 17
    df['dt'] =  df['time_no'].shift(-1) - df['time_no']
    cols = df.columns.to_list()
    d = df.describe()
    st.write('your GPS data might not be of highest (1 second) frequency, we need to adjust it.....are working on it, please wait ')
        
    if 'df' not in st.session_state:
        st.session_state.df = df = enrich_gpx(df)
        

    st.write('Thank you for your patience, it is done!')
    st.write('--------------------------------------------')
    df = df.sort_index()
    df = df.reset_index()
    df['one'] = 1
    power_meter = d['power']['mean'] != 0
    label = {'latitude':'latitude',
            'longitude':'longitude', 'elevation':'elevation', 'power':'power',
            'heart rate':'heart rate', 'temperature':'temperature','cadence':'cadence'}
    for item in ('latitude',
            'longitude', 'elevation','power',
            'heart rate', 'temperature','cadence'): 
        gpx_message(item, label[item], where=st.sidebar)

# SUMMARY
    df['d_lat'] = df.latitude - df.latitude.shift(1)
    df['d_lon'] = df.longitude - df.longitude.shift(1) 
    df = miros_filter(df, 'elevation', 'ele1')
    df = miros_filter(df, 'ele1', 'ele2')
    df = miros_filter(df, 'ele2', 'ele3')
    radius = earth_radius * 1000  # in metres
    # verticaly   df.d_lat/180*np.pi*radius
    # horizontaly   np.cos(df.lat)*df.d_lon/180*np.pi*radius
    df['d'] = np.pi*radius/180*(df.d_lat**2 + (np.cos(df.latitude)*df.d_lon)**2)**0.5
    
    # speed = dist
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
    
    df['one'] = 1.
    df['half'] = 1/2
    df['zero'] = 0
    df['ones'] = df['one'].cumsum()
    df['dist_2'] = df['dist7'].shift(1) * df['dist7'] #a['#a['dist7']**2
    df['Ek'] = (float(weight) + float(weight_bike))/2*df['dist_2']

    df['dist5obs'] = df['d'].shift(-2) + df['d'].shift(-1) + df['d'] + df['d'].shift(1) + df['d'].shift(2)
    df['slope'] = (df['dist5obs']>5)*(df['d_ele'].shift(-2) + df['d_ele'].shift(-1) + df['d_ele'] + df['d_ele'].shift(1) + df['d_ele'].shift(2) )/df['dist5obs']  
    
    d = df.describe(percentiles = [.1,.25,.5])
    st.write(f'Your route from **{date}** started at **{df["time"][df.index.min()]}** and ended at **{df["time"][df.index.max()]}**.')
    st.write()
    st.write(f"Your elevation ranged between {round(d['elevation']['min'])}m and {round(d['elevation']['max'])}m above see level and you overall gained **{round(df[df['d_ele']>0]['d_ele'].sum())}m**.")
    st.write(f"Steepest ascend was **{round(100*d['slope']['max'],1)}%** and steepest descend was **{-round(100*d['slope']['min'],1)}%**.")
    st.write()
    if 0 != d['temperature']['max'] or d['temperature']['min'] !=0:
        st.write(f'Temperature ranged between **{round(d["temperature"]["10%"])}** and **{round(d["temperature"]["max"])}** degrees with median value of **{round(d["temperature"]["50%"])}** degrees.')
    heart_rate = False
    if 0 != d['heart rate']['max']:
        heart_rate = True
        st.write(f"Your heart rate ranged between {round(d['heart rate']['min'])} bpm and {round(d['heart rate']['max'])} bpm with median value of {round(d['heart rate']['50%'])} bpm.")
    if 0 != d['cadence']['max']:
        st.write(f"Your cadence ranged between **{round(d['cadence']['10%'])} rpm** and **{round(d['cadence']['max'])} rpm** with median value of **{round(d['cadence']['50%'])} rpm**.")

    st.write(f"Your speed ranged between **{round(3.6*d['dist7']['10%'],1)}km/h** and **{round(3.6*d['dist7']['max'],1)}km/h** with median value of **{round(3.6*d['dist7']['50%'],1)}km/h** (average **{round(3.6*d['dist7']['mean'],1)}km/h**).")
    st.write()
    if power_meter:
        st.write(f"Your average power was **{round(d['power']['mean'])} watts**, when pedalling even **{round(df[df['power']>0]['power'].mean())} watts**.")
    else:
        df = power_estimate(df, tyre_resistance, air_resistance)
        st.write(f"Your average power is estimated to **{round(df['power estimate'].mean())} watts**, when pedalling even **{round(df[df['power estimate']>0]['power estimate0'].mean())} watts**.")
    
    st.write('-----------------------------------------------------')
    
    df['velocity'] = 3.6 * df['dist7']
    dd = df.describe(percentiles = [.02, .98])
    var_list = ['velocity', 'elevation', 'slope']
    if dd['cadence']['98%'] > 0: var_list = var_list + ['cadence']
    if dd['temperature']['98%'] > 0: var_list = var_list + ['temperature']
    if dd['heart rate']['98%'] > 0: var_list = var_list + ['heart rate']
    if power_meter: 
        PC = power_curve('power',st,st,False)
        var_list = var_list + ['power'] + PC['POWER CURVE'].to_list()
    else: 
        PC = power_curve('power estimate',st,st,False)
        var_list = var_list + ['power estimate'] + PC['POWER CURVE'].to_list()
       
    h1,h2 = st.columns((3,5))
    variable = h1.selectbox('What to color on a map?', var_list ,0)
    
    if variable == 'elevation':
        variable = 'ele3'
    pom ='abcde'
    if variable[:3]=='max': 
        pom = variable
        if power_meter: variable = 'power'
        else: variable = 'power estimate'
        
    min_value = dd[variable]['2%']
    max_value = dd[variable]['98%']
    var_range = (max_value - min_value)
    
    df[variable] = df[variable].fillna(0)
    df['rel'] = round((df[variable]-min_value)/var_range,1)
    df['red'] = (df['rel']-0.5)/0.25
    df['red'] = np.where(df['red'] < 0, 0, df['red'])
    df['red'] = np.where(df['red'] > 1, 1, df['red'])

        
    df['green'] = np.where(df['rel']<0.25, df['rel']/0.25,1)
    df['green'] = np.where(df['rel']>0.75, (1-df['rel'])/0.25,  df['green'])
    df['green'] = np.where(df['green'] < 0, 0, df['green'])
    df['green'] = np.where(df['green'] > 1, 1, df['green'])

        
    df['blue'] = (0.5-df['rel'])/0.25
    df['blue'] = np.where(df['rel']<0.25, 1, df['blue'])
    df['blue'] = np.where(df['rel']>0.5, 0, df['blue'])
    df['blue'] = np.where(df['blue'] < 0, 0, df['blue'])
    df['blue'] = np.where(df['blue'] > 1, 1, df['blue'])
    
    if pom[:3]=='max':
        df['red'] = df['red']*df['interval'+pom[-5:]]+ (1-df['interval'+pom[-5:]])*0.8
        df['green'] = df['green']*df['interval'+pom[-5:]]+ (1-df['interval'+pom[-5:]])*0.8
        df['blue'] = df['blue']*df['interval'+pom[-5:]]+ (1-df['interval'+pom[-5:]])*0.8
    
    cols = ['red','green','blue']
    
    df['color'] = df[cols].to_numpy().tolist()
    dict_units = {'velocity':'km/h', 
                  'ele3':'metres above see level',
                  'elevation':'metres above see level',
                  'slope':'DOWN             gradient            UP ', 
                  'cadence':'rotates per minute', 
                  'power':'watts', 'power estimate':'watts',
                  'heart rate':'beats per minute',
                  'temperature':'degrees Celsius'} 
    heatmap(variable, min_value, max_value, dict_units[variable], h2)
    st.map(df, latitude='latitude', longitude='longitude', size = 10, color='color')
   
    # POWER CURVE 
    st.write('----------------------------------------------------------------------------')
    if power_meter:
        st.subheader('Power curve based on real data from your power meter device:')
    else: st.subheader('Power curve based on estimates taken into account your speed, elevation changes, weight, bike type,... :')

    pc1, pc2 = st.columns(2)
    if power_meter:
        power_curve = power_curve('power', pc1, pc1)
        show_power_curve(power_curve, pc2)
    else: 
        df = power_estimate(df, tyre_resistance, air_resistance)
        power_curve = power_curve('power estimate', pc1, pc1)
        show_power_curve(power_curve, pc2)
    
    st.write('---------------------------------------------------------')
    st.subheader('Two variable comparison')
    st.write('------------------------------------------------------')
    if 'slider_start' not in st.session_state:
        slider_start = round(len_df/3)  
    if 'slider_end' not in st.session_state:
        slider_end = round(2*len_df/3)
    #slider_start, slider_end = 356, 539 #1110, 1220
    track_slider = st.slider('**Seconds to consider?**  ', 0, len_df, (slider_start, slider_end))# (round(len_df/7), round(6*len_df/7)))
    st.session_state.slider_start = slider_start = track_slider[0]
    st.session_state.slider_end = slider_end = track_slider[1]
    col1, col2 = st.columns(2)
    var_list_reduced = []
    for item in var_list:
        if item[:3]!='max': var_list_reduced.append(item)
    left_var = col1.selectbox('Left variable to draw?', var_list_reduced, 1)
    right_var = col2.selectbox('Right variable to draw?', var_list_reduced, 0)
    
    profile_plot(df, max(0, round(0.8*slider_start)), min(len(df), round(1.2*slider_end)), left_var, right_var, primaryColor, secondaryBackgroundColor,  bg_color=bg_color, place=st)
    
    if False: #not power_meter:
        st.write('Without powermeter data you can estimate your average power or resistance coefficients:')
        c1, c2, c3 = st.columns((2,3,3))
        no_pm = c1.selectbox('I want to estimate:', ('none', 'power', 'resistance coefficients - no pedaling'), 0)
        if no_pm == 'power':
            tr_str = c2.text_input('Enter tire resistance coeff (in promiles):', '4')
            ar_str = c3.text_input('Enter air resistance coeff (in BLABLA):', '0.2')
            ar = float(ar_str)
            tr = float(tr_str)
            df = power_estimate(df, tr, ar)
            profile_plot(df, track_slider[0], track_slider[1],'power estimate', 'velocity', "#9dfb11", 'white',  bg_color=bg_color, place=st)
        if no_pm == 'resistance coefficients - no pedaling':
            c2.write('Please select part of your route without pedaling (climb descend or so). ')
            c3.write(' We will try tom estimate both resistance coefficients - linear/tire one and qvadratic/air one.')
            lin_regressor = LinearRegression()
            a = df[track_slider[0]:track_slider[1]]
            a['Ep'] = a['d_Ep'].cumsum()
            Y = -a['Ep'] - a['Ek']
            a['total_dist'] = a['dist'].cumsum()
            a['tire'] = a['total_dist'] * (float(weight) + float(weight_bike))* 9.81
            a['total_dist_2'] = a['dist_2']* a['dist7'] 
            a['air'] = a['total_dist_2'].cumsum() 
            X = a[['tire', 'air']]
            lin_regressor = lin_regressor.fit(X, Y)
            c1.write('intercept:')
            c1.write(lin_regressor.intercept_)
            c3.write(f'linear/tire coefficient is {round(-1000 * lin_regressor.coef_[0], 2)}')
            c3.write(f'quadratic/air coefficient is {-round(lin_regressor.coef_[1], 2)}')
            lin_str = st.text_input('Enter tire resistance coeff (in promiles):','-0.00659')
            
            if lin_str != '':
                Y = -a['Ep'] - a['Ek'] + float(lin_str) * (a['tire'])
                X = a[['air']]
                lin_regressor = lin_regressor.fit(X, Y)
                st.write(f'quadratic/air coefficient is {-round(lin_regressor.coef_[0], 2)}')

            if df not in st.session_state:
                st.session_state.df = df








