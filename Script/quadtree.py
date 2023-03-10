### import libraries

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

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['PROJ_LIB'] = r'/usr/proj80/share/proj'

os.environ['GDAL_DATA'] = r'/beegfs/store4/chenyangkang/miniconda3/share'

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)

np.random.seed(42)

# alt.data_transformers.enable('json')
alt.data_transformers.disable_max_rows()

# In[3]:
# os.environ['PROJ_LIB'] = r'/beegfs/store4/chenyangkang/miniconda3/share/proj'
# os.environ['GDAL_DATA'] = r'/beegfs/store4/chenyangkang/miniconda3/share'


import pickle


class Point():
    def __init__(self, index, x, y):
        self.x = x
        self.y = y
        self.index = index
        
class Node():
    def __init__(self, x0, y0, w, h, points):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.points = points
        self.children = []

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_points(self):
        return self.points
    
    
def recursive_subdivide(node, grid_len_long_upper_threshold, grid_len_long_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold):

    
    if len(node.points)/2 <= points_lower_threshold:
        if not ((node.width > grid_len_long_upper_threshold) or (node.height > grid_len_lat_upper_threshold)):
            return
    
    if (node.width/2 < grid_len_long_lower_threshold) or (node.height/2 < grid_len_lat_lower_threshold):
        return
   
    w_ = float(node.width/2)
    h_ = float(node.height/2)

    p = contains(node.x0, node.y0, w_, h_, node.points)
    x1 = Node(node.x0, node.y0, w_, h_, p)
    recursive_subdivide(x1, grid_len_long_upper_threshold, grid_len_long_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold)

    p = contains(node.x0, node.y0+h_, w_, h_, node.points)
    x2 = Node(node.x0, node.y0+h_, w_, h_, p)
    recursive_subdivide(x2, grid_len_long_upper_threshold, grid_len_long_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold)

    p = contains(node.x0+w_, node.y0, w_, h_, node.points)
    x3 = Node(node.x0 + w_, node.y0, w_, h_, p)
    recursive_subdivide(x3, grid_len_long_upper_threshold, grid_len_long_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold)

    p = contains(node.x0+w_, node.y0+h_, w_, h_, node.points)
    x4 = Node(node.x0+w_, node.y0+h_, w_, h_, p)
    recursive_subdivide(x4, grid_len_long_upper_threshold, grid_len_long_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold)

    node.children = [x1, x2, x3, x4]
    
    
def contains(x, y, w, h, points):
    pts = []
    for point in points:
        if point.x >= x and point.x <= x+w and point.y>=y and point.y<=y+h:
            pts.append(point)
    return pts


def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += (find_children(child))
    return children


import random
import matplotlib.pyplot as plt # plotting libraries
import matplotlib.patches as patches

class QTree():
    def __init__(self, grid_len_long_upper_threshold, grid_len_long_lower_threshold, \
                        grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                        points_lower_threshold, long_lat_equal_grid=True,\
                            rotation_angle = 0, \
                    calibration_point_x_jitter = 0,\
                        calibration_point_y_jitter = 0):

        self.points_lower_threshold = points_lower_threshold
        self.grid_len_long_upper_threshold = grid_len_long_upper_threshold
        self.grid_len_long_lower_threshold = grid_len_long_lower_threshold
        self.grid_len_lat_upper_threshold = grid_len_lat_upper_threshold
        self.grid_len_lat_lower_threshold = grid_len_lat_lower_threshold
        self.long_lat_equal_grid = long_lat_equal_grid
        # self.points = [Point(random.uniform(0, 10), random.uniform(0, 10)) for x in range(n)]
        self.points = []
        self.rotation_angle = rotation_angle
        self.calibration_point_x_jitter = calibration_point_x_jitter
        self.calibration_point_y_jitter = calibration_point_y_jitter


    def add_long_lat_data(self, indexes, x_array, y_array):
        if not len(x_array) == len(y_array) or not len(x_array) == len(indexes):
            raise ValueError("input longitude and latitute and indexes not in same length!")
        
        data = np.array([x_array, y_array]).T
        angle = self.rotation_angle
        r = angle/360
        theta = r * np.pi * 2
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        data = data @ rotation_matrix
        long_new = (data[:,0] + self.calibration_point_x_jitter).tolist()
        lat_new = (data[:,1] + self.calibration_point_y_jitter).tolist()

        for index,long,lat in zip(indexes, long_new, lat_new):
            self.points.append(Point(index, long, lat))


    
    def generate_griding_params(self):
        x_list = [i.x for i in self.points]
        y_list = [i.y for i in self.points]
        self.grid_length_x = np.max(x_list)-np.min(x_list)
        self.grid_length_y = np.max(y_list)-np.min(y_list)

        left_bottom_point_x = np.min(x_list)
        left_bottom_point_y = np.min(y_list)

        self.left_bottom_point = (left_bottom_point_x ,left_bottom_point_y)
        if self.long_lat_equal_grid == True:
            self.root = Node(left_bottom_point_x, left_bottom_point_y, \
                max(self.grid_length_x, self.grid_length_y), \
                    max(self.grid_length_x, self.grid_length_y), self.points)
        elif self.long_lat_equal_grid == False:
            self.root = Node(left_bottom_point_x, left_bottom_point_y, \
                self.grid_length_x, \
                    self.grid_length_y, self.points)
        else:
            raise ValueError('The input long_lat_equal_grid not a boolean value!')            

    
    def get_points(self):
        return self.points
    
    def subdivide(self):
        recursive_subdivide(self.root, self.grid_len_long_upper_threshold, self.grid_len_long_lower_threshold, \
                            self.grid_len_lat_upper_threshold, self.grid_len_lat_lower_threshold, \
                            self.points_lower_threshold)
    
    def graph(self):
        plt.figure(figsize=(20, 20))
        plt.xlim([-180,180])
        plt.ylim([-90,90])
        plt.title("Quadtree")
        c = find_children(self.root)
        print("Number of segments: %d" %len(c))
        areas = set()
        width_set = set()
        height_set = set()
        for el in c:
            areas.add(el.width*el.height)
            width_set.add(el.width)
            height_set.add(el.height)
        print("Minimum segment area: %.3f ,min_long: %.3f units, min_lat: %.3f units" %(min(areas),min(width_set),min(height_set)))

        theta = -(self.rotation_angle/360) * np.pi * 2
        rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
            ])

        for n in c:
            xy0_trans = np.array([[n.x0, n.y0]])
            if self.calibration_point_x_jitter:
                new_x = xy0_trans[:,0] - self.calibration_point_x_jitter
            else:
                new_x = xy0_trans[:,0]
            
            if self.calibration_point_y_jitter:
                new_y = xy0_trans[:,1] - self.calibration_point_y_jitter
            else:
                new_y = xy0_trans[:,1]
            new_xy = np.array([[new_x[0], new_y[0]]]) @ rotation_matrix
            new_x = new_xy[:,0]
            new_y = new_xy[:,1]

            plt.gcf().gca().add_patch(patches.Rectangle((new_x, new_y), n.width, n.height, fill=False,angle=self.rotation_angle))
        
        x = np.array([point.x for point in self.points]) - self.calibration_point_x_jitter
        y = np.array([point.y for point in self.points]) - self.calibration_point_y_jitter

        data = np.array([x,y]).T @ rotation_matrix
        plt.scatter(data[:,0].tolist(), data[:,1].tolist(), s=0.2) # plots the points as red dots
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.show()
        return

    def get_final_result(self):
        ## get points assignment to each grid and transform the data into pandas df.
        all_grids = find_children(self.root)
        point_indexes_list = []
        point_grid_width_list = []
        point_grid_height_list = []
        point_grid_points_number_list = []
        leaf_grid_list = []
        calibration_point_list = []
        for grid in all_grids:
            point_indexes_list.append([point.index for point in grid.points])
            point_grid_width_list.append(grid.width)
            point_grid_height_list.append(grid.height)
            point_grid_points_number_list.append(len(grid.points))
            calibration_point_list.append((round(grid.x0, 6), round(grid.y0, 6)))
        
        result = pd.DataFrame({'checklist_indexes': point_indexes_list,
                'stixel_indexes': list(range(len(point_grid_width_list))),
                'stixel_width':point_grid_width_list,
                'stixel_height': point_grid_height_list,
                'stixel_checklist_count':point_grid_points_number_list,
                'stixel_calibration_point(transformed)':calibration_point_list,
                'rotation':[self.rotation_angle] * len(point_grid_width_list),
                'space_jitter(first rotate by zero then add this)':[(round(self.calibration_point_x_jitter, 6), round(self.calibration_point_y_jitter, 6))] * len(point_grid_width_list)})

        result = result[result['stixel_checklist_count']!=0]
        return result
        



import time



def get_ensemble_quadtree(data,size=1,\
                            grid_len_long_upper_threshold=25, grid_len_long_lower_threshold=5, \
                            grid_len_lat_upper_threshold=25, grid_len_lat_lower_threshold=5, \
                            points_lower_threshold=50):
    ensemble_all_df_list = []
    sub_data_all = data
    for ensemble_count in range(size):
        if ensemble_count==0:
            time_jitter = 0
        else:
            time_jitter = -(ensemble_count/size)*30.5
            
        rotation_angle = np.random.uniform(0,360)
        calibration_point_x_jitter = np.random.uniform(-10,10)
        calibration_point_y_jitter = np.random.uniform(-10,10)

        print(f'ensembel_count: {ensemble_count}')
        
        for time_block_index in range(0, 13):

#             print(f'Processing ensemble {ensemble_count}, time_block {time_block_index}')

            time_start = time_block_index*30.5 + time_jitter

            if time_start + 30.5 <= 0:
                continue

            if time_start-30.5 >= 366:
                continue

            time_end = time_start + 30.5


            if time_start < 0:
                sub_data=sub_data_all[
                    ((sub_data_all['DOY']>=0) & (sub_data_all['DOY']<=time_end)) | \
                                     ((sub_data_all['DOY']>=(366 - (0-time_start))) & (sub_data_all['DOY']<=366))
                ]#.reset_index(drop=True)

            elif time_end > 366:
                continue
#                 sub_data=sub_data_all[
#                     ((sub_data_all['DOY']>=time_start) & (sub_data_all['DOY']<=366)) | \
#                                      ((sub_data_all['DOY']>=0) & (sub_data_all['DOY']<=(0+time_end-366)))
#                 ]#.reset_index(drop=True)

            else:   
                sub_data=sub_data_all[(sub_data_all['DOY']>=time_start) & (sub_data_all['DOY']<time_end)]#.reset_index(drop=True)


            QT_obj = QTree(grid_len_long_upper_threshold=grid_len_long_upper_threshold, \
                            grid_len_long_lower_threshold=grid_len_long_lower_threshold, \
                            grid_len_lat_upper_threshold=grid_len_lat_upper_threshold, \
                            grid_len_lat_lower_threshold=grid_len_lat_lower_threshold, \
                            points_lower_threshold=points_lower_threshold, \

                            long_lat_equal_grid = True, rotation_angle = rotation_angle, \
                                calibration_point_x_jitter = calibration_point_x_jitter,\
                                    calibration_point_y_jitter = calibration_point_y_jitter)

            ## Give the data and indexes. The indexes should be used to assign points data so that base model can run on those points,
            ## You need to generate the splitting parameters once giving the data. Like the calibration point and min,max.
            QT_obj.add_long_lat_data(sub_data.index, sub_data['longitude'].values, sub_data['latitude'].values)
            QT_obj.generate_griding_params()
            ## Call subdivide to precess
            QT_obj.subdivide()
            this_slice = QT_obj.get_final_result()
            this_slice['ensemble_index'] = ensemble_count
            this_slice['DOY_start'] = time_start
            this_slice['DOY_end'] = time_end
            this_slice['checklist_name'] = [sub_data.loc[i,:]['sampling_event_identifier'].values.tolist() for i in this_slice['checklist_indexes']]
            ensemble_all_df = this_slice
#             ensemble_all_df = ensemble_all_df[ensemble_all_df['stixel_checklist_count']>=10]
#             ensemble_all_df['DOY_start'][ensemble_all_df['DOY_start']<1]=1
            ensemble_all_df['DOY_start']=round(ensemble_all_df['DOY_start'],1)
#             ensemble_all_df['DOY_end'][ensemble_all_df['DOY_end']>366]=366
            ensemble_all_df['DOY_end']=round(ensemble_all_df['DOY_end'],1)
            ensemble_all_df['unique_stixel_id'] = [str(time_block_index)+"_"+str(i)+"_"+str(k) for i,k in zip (ensemble_all_df['ensemble_index'].values, ensemble_all_df['stixel_indexes'].values)]
    #            if (ensemble_count == 0) and (time_block_index == 0):
    #                ensemble_all_df.to_csv(f'./ensemble_metadata_all/ensembles_metadata_year{year}_ensemble{ensemble_count}_timeblock{time_block_index}.txt',sep='\t',index=False)
    #            else:

    #         ensemble_all_df.to_csv(f'./ensemble_metadata_all/ensembles_metadata_year{year}_ensemble{ensemble_count}_timeblock{time_block_index}.txt',sep='\t',index=False,header=True)
            ensemble_all_df_list.append(ensemble_all_df)
    ensemble_df = pd.concat(ensemble_all_df_list)
        
    return ensemble_df


