import holidays, os, requests, sys, math
import pandas as pd
import datetime as dt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import numpy as np
from bs4 import BeautifulSoup
import polars as pl
import holidays, os, requests
import pandas as pd
import matplotlib.pyplot as plt
import torch as T
from torch import nn   
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
from tqdm import trange 

import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree

warnings.filterwarnings('ignore')
pio.renderers.default='browser'

def date_format(date): # changing date format to suit the MET search
    return date.replace(second=0, microsecond=0, minute=0).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+'Z'

def new_frost_key(): # creating a frost key from a email address and returning it
    params = {'email': 'testemail'}
    r = requests.get('https://frost.met.no/auth/requestCredentials.html', params=params)
    html = str(r.text)
    soup = BeautifulSoup(html, "lxml")
    item = soup.select_one("code[id='key']").text
    return item
    
def geolocation(search_term): # Get the geolocation of stations (or any location)
    parameters = { 'text': search_term+' Stasjon', 'lang': 'en', 'boundary.country':'NO'} 
    location = requests.get('https://api.entur.io/geocoder/v1/autocomplete', parameters) 
    try:
        return location.json()['features'][0]['geometry']['coordinates'] 
    except KeyError: # Catch exception when the key expires nd rerun the function
        return geolocation(search_term)
    
def weather_source(latitude, longitude, weather_type):  
    # Get the weather station information from the geolocation of station (or any location)
    parameters1 = { 'geometry': 'nearest(POINT('+str(longitude)+' '+str(latitude)+'))','elements': weather_type,}
    try:
        source = requests.get('https://frost.met.no/sources/v0.jsonld', parameters1, auth=(new_frost_key(),'')).json() # ['data'][0]['id']
        return source  # loc = source.json()['data'][0]['geometry']['coordinates']
    except KeyError: # Catch exception when the key expires nd rerun the function
        return source(latitude, longitude, weather_type)

def weather_element(source,date, weather_type='max(air_temperature PT1H)'):
    # Get the weather element value from the weather station
    parameters = {'sources': source,'elements': weather_type,  'referencetime': date,}
    try:
        reqst = requests.get('https://frost.met.no/observations/v0.jsonld', parameters, auth=(new_frost_key(),''))
        if(reqst.status_code==200):
            elmnt = reqst.json()['data'][0]['observations'][0]['value']  #[0]['value']
            return elmnt
        else:
            return 'error'+str(reqst.status_code)
    except KeyError: # Catch exception when the key expires nd rerun the function
        print('except')
        return weather_element(source,date, weather_type)

def get_weather_values(weather_list, weather_element1,val_weather=dict()):
    # Create a dictionary for locations (keys) and the corresponding weather value (value)
    val_list_done = list()
    try:
        for i in weather_list:
            a, b = i.split('_')[0],i.split('_')[1]
            val = weather_element(a,b, weather_element1)
            val_weather[i] = val
            print(a, b, val, weather_element1)
            val_list_done.append(i) # list of all done weather values
    except AttributeError: # check if error and rerun the function on rest of the values
        get_weather_values(list(set(weather_list)^set(val_list_done)), weather_element1,val_weather)
    return val_weather

#TO BE USED LATER
def nearness_matrix(ref_loc_name, all_loc=dict()): 
    # create a proximity matrix (from closest to furthest) for all the weather stations in the list
    ref_loc = all_loc[ref_loc_name]
    for k,v in all_loc.items():
        all_loc[k] = math.sqrt(pow((ref_loc[0]-v[0]),2) + pow((ref_loc[1]-v[1]),2))

    return all_loc
