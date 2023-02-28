import sys
import pandas as pd
import numpy as np
import numpy
import math
import psycopg2
from psycopg2 import Error
import os
import warnings
import pickle
import time
import statsmodels.api as sm
from tqdm import tqdm

import random
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import seaborn as sns

import json
import networkx as nx
import nx_altair as nxa
import altair as alt
from sklearn.cluster import AgglomerativeClustering
from networkx.algorithms.community import *
from scipy.cluster.hierarchy import dendrogram,leaves_list
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

import contextily as cx
from matplotlib import cm
from sklearn.preprocessing import normalize

from tqdm import tqdm


import rasterio as rs
import math
import psycopg2
from psycopg2 import Error
from osgeo import gdal, osr, ogr
import sys
import os
from pyproj import Transformer
import pycrs
import pyproj
import numpy as np
from io import StringIO
import datetime
import re

import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_tweedie_deviance,\
        mean_absolute_error,mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score,recall_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor,XGBClassifier,XGBRFClassifier,XGBRFRegressor
from sklearn.utils import class_weight
import xgboost as xgb
from sklearn.inspection import partial_dependence

######
from quadtree import *
######


class dummy_model1():
    def __init__(self):
        pass
    def predict(self,X_test):
        return np.array([0] * len(X_test))
    def predict_proba(self,X_test):
        return np.array([[1,0]] * len(X_test))



    
class AdaSTEM():
    '''
        attributes:
        
        self.model1: after init
        self.model2: after init
        self.x_names: after init
        self.sp: after init
        self.positive_count: after init
        
        self.sample_size: after fit
        self.month1_sample_size: after fit
        ...
        self.month12_sample_size: after fit
        
        self.bias_df: after get_bias_df
        
        
        Functions:
        __init__(self,model1, model2,x_names,sp)
        fit(self,X_train, y_train)
        predict(self,X_test)
        get_bias_df(self, data)
        score(self,X_test,y_test)
        plot_error(self)
        
    '''
    def __init__(self,base_model,
                ensemble_fold=1,
                grid_len_long_upper_threshold=25,
                grid_len_long_lower_threshold=5,
                grid_len_lat_upper_threshold=25,
                grid_len_lat_lower_threshold=5,
                points_lower_threshold=50):

        self.base_model = base_model
        self.ensemble_fold = ensemble_fold
        self.min_ensemble_require = int(ensemble_fold*0.9)
        self.grid_len_long_upper_threshold=grid_len_long_upper_threshold
        self.grid_len_long_lower_threshold=grid_len_long_lower_threshold
        self.grid_len_lat_upper_threshold=grid_len_lat_upper_threshold
        self.grid_len_lat_lower_threshold=grid_len_lat_lower_threshold
        self.points_lower_threshold=points_lower_threshold
                

    def split(self, X_train):
        fold = self.ensemble_fold
        ensemble_df = get_ensemble_quadtree(X_train,\
                                            size=fold,\
                                            grid_len_long_upper_threshold=self.grid_len_long_upper_threshold, \
                                            grid_len_long_lower_threshold=self.grid_len_long_lower_threshold, \
                                            grid_len_lat_upper_threshold=self.grid_len_lat_upper_threshold, \
                                            grid_len_lat_lower_threshold=self.grid_len_lat_lower_threshold, \
                                            points_lower_threshold=self.points_lower_threshold)

        grid_dict = {}
        for ensemble_index in ensemble_df.ensemble_index.unique():
            this_ensemble = ensemble_df[ensemble_df.ensemble_index==ensemble_index]
            
            this_ensemble_gird_info = {}
            this_ensemble_gird_info['checklist_index'] = []
            this_ensemble_gird_info['stixel'] = []
            for index,line in this_ensemble.iterrows():
                this_ensemble_gird_info['checklist_index'].extend(line['checklist_indexes'])
                this_ensemble_gird_info['stixel'].extend([line['unique_stixel_id']]*len(line['checklist_indexes']))
            
            cores = pd.DataFrame(this_ensemble_gird_info)
#             return cores
        
            cores2 = pd.DataFrame(list(X_train.index),columns=['data_point_index'])
            cores = pd.merge(cores, cores2, 
                             left_on='checklist_index',right_on = 'data_point_index',how='right')
            
            grid_dict[ensemble_index] = cores.stixel.values
            
        self.grid_dict = grid_dict
        self.ensemble_df = ensemble_df
        return grid_dict
    

    def fit(self, X_train, y_train, time_interval=30.5):
        self.time_interval = time_interval
        self.x_names = list(X_train.columns)
        for i in ['longitude','latitude','sampling_event_identifier','y_true']:
            if i in list(self.x_names):
                del self.x_names[self.x_names.index(i)]

        import copy
        X_train_copy = X_train.copy()
        X_train_copy['true_y'] = y_train
        
        grid_dict = self.split(X_train)

        ##### define model dict
        self.model_dict = {}
        for index,line in tqdm(self.ensemble_df.iterrows(),total=len(self.ensemble_df),desc='training: '):
            name = f'{line.ensemble_index}_{line.unique_stixel_id}'
            sub_X_train = X_train_copy[X_train_copy.sampling_event_identifier.isin(line.checklist_name)]
            if len(sub_X_train)<50: ####### threshold
                continue
            sub_y_train = sub_X_train.iloc[:,-1]
            sub_X_train = sub_X_train[self.x_names]

            ##### fit
            if not np.sum(sub_y_train) == 0:
                sample_weights = \
                    class_weight.compute_sample_weight(class_weight='balanced',y=np.where(sub_y_train>0,1,0))
                
                self.base_model.fit(sub_X_train, np.where(sub_y_train>0,1,0), sample_weight=sample_weights)

                ###### store
                self.model_dict[f'{name}_model'] = copy.deepcopy(self.base_model)

            else:
                self.model_dict[f'{name}_model'] = dummy_model1()

    
    def predict_proba(self,X_test):
        ##### predict
        X_test_copy = X_test.copy()
        
        round_res_list = []
        ensemble_df = self.ensemble_df
        for ensemble in list(ensemble_df.ensemble_index.unique()):
            this_ensemble = ensemble_df[ensemble_df.ensemble_index==ensemble]
            this_ensemble['stixel_calibration_point_transformed_left_bound'] = \
                        [i[0] for i in this_ensemble['stixel_calibration_point(transformed)']]

            this_ensemble['stixel_calibration_point_transformed_lower_bound'] = \
                        [i[1] for i in this_ensemble['stixel_calibration_point(transformed)']]

            this_ensemble['stixel_calibration_point_transformed_right_bound'] = \
                        this_ensemble['stixel_calibration_point_transformed_left_bound'] + this_ensemble['stixel_width']

            this_ensemble['stixel_calibration_point_transformed_upper_bound'] = \
                        this_ensemble['stixel_calibration_point_transformed_lower_bound'] + this_ensemble['stixel_height']

            X_test_copy = self.transform_pred_set_to_STEM_quad(X_test_copy,this_ensemble)
            
            ##### pred each stixel
            res_list = []
            for index,line in tqdm(this_ensemble.iterrows(),total=len(this_ensemble), desc=f'predicting ensemble {ensemble} '):
                grid_index = line['unique_stixel_id']
                sub_X_test = X_test_copy[
                    (X_test_copy.DOY>=line['DOY_start']) & (X_test_copy.DOY<=line['DOY_end']) & \
                    (X_test_copy.long_new>=line['stixel_calibration_point_transformed_left_bound']) &\
                    (X_test_copy.long_new<=line['stixel_calibration_point_transformed_right_bound']) &\
                    (X_test_copy.lat_new>=line['stixel_calibration_point_transformed_lower_bound']) &\
                    (X_test_copy.lat_new<=line['stixel_calibration_point_transformed_upper_bound'])
                ]
                ##### get max value point
                try:
                    sub_X_test['time_observation_started_minute_of_day'] = self.max_value_point_dict['time_observation_started_minute_of_day'][grid_index]
                    sub_X_test['obsvr_species_count'] = self.max_value_point_dict['obsvr_species_count'][grid_index]
                except:
                    sub_X_test['time_observation_started_minute_of_day'] = -1
                    sub_X_test['obsvr_species_count'] = -1
                    
                ##### get training data
                for i in ['longitude','latitude','sampling_event_identifier','y_true']:
                    if i in list(self.x_names):
                        del self.x_names[self.x_names.index(i)]

                sub_X_test = sub_X_test[self.x_names]


                try:
                    model = self.model_dict[f'{ensemble}_{grid_index}_model']
                    pred = model.predict_proba(sub_X_test)[:,1]
                    res = pd.DataFrame({'index':list(sub_X_test.index),
                                        'pred':pred}).set_index('index')
                except Exception as e:
                    # print(e)
                    res = pd.DataFrame({'index':list(sub_X_test.index),
                                        'pred':[np.nan]*len(list(sub_X_test.index))
                                        }).set_index('index')

                res_list.append(res)
                
            res_list = pd.concat(res_list)
            res_list = res_list.reset_index(drop=False).groupby('index').first()
            round_res_list.append(res_list)
       
        ####### only sites that meet the minimum ensemble requirement is remained
        res = pd.concat([df['pred'] for df in round_res_list], axis=1)
        res_median = res.mean(axis=1, skipna=True) ##### mean of all grid model that predicts this geo point
        res_not_nan_count = res.isnull().sum(axis=1)
        res_median = res_median[res_not_nan_count<=self.min_ensemble_require] #### min_ensemble_require
        res_list['pred'] = res_median
    
        res_list = pd.concat([X_test, res_list],axis=1)
        return res_list['pred'].values
            
    def transform_pred_set_to_STEM_quad(self,X_train,ensemble_info):

        x_array = X_train['longitude']
        y_array = X_train['latitude']
        coord = np.array([x_array, y_array]).T
        angle = float(ensemble_info.iloc[0,:]['rotation'])
        r = angle/360
        theta = r * np.pi * 2
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        coord = coord @ rotation_matrix
        calibration_point_x_jitter = \
                float(ensemble_info.iloc[0,:]['space_jitter(first rotate by zero then add this)'][0])
        calibration_point_y_jitter = \
                float(ensemble_info.iloc[0,:]['space_jitter(first rotate by zero then add this)'][1])

        long_new = (coord[:,0] + calibration_point_x_jitter).tolist()
        lat_new = (coord[:,1] + calibration_point_y_jitter).tolist()

        X_train['long_new'] = long_new
        X_train['lat_new'] = lat_new

        return X_train
