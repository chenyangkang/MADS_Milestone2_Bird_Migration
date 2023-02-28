import sys
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import math
import psycopg2
from psycopg2 import Error
import os
import numpy as np
import warnings
import pickle
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, roc_auc_score, confusion_matrix, cohen_kappa_score
from pygam import LinearGAM 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.inspection import partial_dependence
import time
import datetime
from sklearn.preprocessing import MinMaxScaler

def filter_data(qresult,year):
    qresult_copy = qresult.copy()
    qresult_copy['observation_date'] = pd.to_datetime(qresult_copy['observation_date'])
    qresult_copy['DOY'] = qresult_copy['observation_date'].dt.dayofyear
    qresult_copy['month'] = qresult_copy['observation_date'].dt.month
    qresult_copy['week'] = qresult_copy['observation_date'].dt.week
    qresult_copy['year'] = qresult_copy['observation_date'].dt.year

    qresult_copy['time_observation_started_minute_of_day'] = \
        [i.hour*60+i.minute for i in qresult_copy['time_observation_started']]


    ### transform the protocal "Travaling"... into dummies
    dummy = pd.get_dummies(qresult_copy.protocol_type)
    qresult_copy.insert(3,dummy.columns[0],dummy.iloc[:,0])
    qresult_copy.insert(3,dummy.columns[1],dummy.iloc[:,1])
    qresult_copy.insert(3,dummy.columns[2],dummy.iloc[:,2])


    #######################
    fillna_data = qresult_copy[['duration_minutes','time_observation_started_minute_of_day',\
             'Traveling','Stationary','Area',\
             'effort_distance_km','number_observers','DOY','obsvr_species_count']]

    qresult_copy[['duration_minutes','time_observation_started_minute_of_day',\
             'Traveling','Stationary','Area',\
             'effort_distance_km','number_observers','DOY','obsvr_species_count']]=fillna_data.fillna(-1) 


    qresult_copy['effort_distance_km'] = np.where(qresult_copy['effort_distance_km']>0, qresult_copy['effort_distance_km'], -1)
    qresult_copy[['elevation_mean', 'slope_mean', 'eastness_mean', 'northness_mean', 'elevation_std', 'slope_std', 'eastness_std', 'northness_std', 'prec', 'tmax', 'tmin', 'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']] = qresult_copy[['elevation_mean', 'slope_mean', 'eastness_mean', 'northness_mean', 'elevation_std', 'slope_std', 'eastness_std', 'northness_std', 'prec', 'tmax', 'tmin', 'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']].fillna(-1)
    qresult_copy=qresult_copy.fillna(0)  

    return qresult_copy

