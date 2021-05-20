#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:16:23 2020

@author: carriesu
"""
#The code is used to process the data obtained through using pool on ACI
#Results matrix where entries are unserved energy and total cost will be processed via ALFA

#%% import the data from csv

import numpy as np
import pandas as pd

import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon, LineString

import json
import matplotlib.pyplot as plt

# %% Combines results of several pool runs
# Go throught each csv file, get total cost, usenergy and operation costmatrix, contacenating matrix together
# and print out as two csv files of cost and usenergy
# Also create a dataframe that documents the list of failed components for each event
lst = [0]
cost_frame = []
usenergy_frame = []
opcost_frame = []
event_frame = []

number_of_events = 0

for nu in lst:
    df = pd.read_csv('gcf_pool%d.csv' % nu, index_col=0)

    # Get the event data frame
    temp_dict = df.iloc[0, 5] # column: network addition(s), total cost, unserved energy, operation cost, all the other data
    temp_dict = eval(temp_dict)
    event_df_lst = []

    for i in range(0,100,1):
        lst = temp_dict[i]
        event_df_lst.append([lst[0], lst[1], lst[2]]) # i[0]: event, i[1]: failed component number list, i[2]: failed component name list
    event_df = pd.DataFrame(event_df_lst)
    print(len(event_df_lst))
    number_of_events += len(event_df_lst)
    event_frame.append(event_df)

    # Get the cost dataframe
    cost_dict = {}
    for idx, row in df.iterrows():
        dct = eval(row[1])
        value = []
        for i in range(0,100,1):
            value.append(dct[i])
        cost_dict.update({row[0]: value}) #this does not impact the order of the event because row[0] is used as column, the order of row is the order of events in the list
    cost_df = pd.DataFrame(cost_dict)
    cost_frame.append(cost_df)

    # Get the usenergy dataframe
    usenergy_dict = {}
    for idx, row in df.iterrows():
        dct = eval(row[2])
        value = []
        for i in range(0,100,1):
            value.append(dct[i])
        usenergy_dict.update({row[0]: value})
    usenergy_df = pd.DataFrame(usenergy_dict)
    usenergy_frame.append(usenergy_df)

    # Get the operation cost matrix / dataframe
    opcost_dict = {}
    for idx, row in df.iterrows():
        dct = eval(row[3])
        value = []
        for i in range(0,100,1):
            value.append(dct[i])
        opcost_dict.update({row[0]: value})
    opcost_df = pd.DataFrame(opcost_dict)
    opcost_frame.append(opcost_df)

# the final dataframe is concated from list of dataframs
event_dff = pd.concat(event_frame, ignore_index=True)
usenergy_dff = pd.concat(usenergy_frame, ignore_index=True)
cost_dff = pd.concat(cost_frame, ignore_index=True)
opcost_dff = pd.concat(opcost_frame,ignore_index=True)


# drop_list=list(range(9))
# drop_list = [-x for x in drop_list]
# event_dff.drop(event_dff.index[drop_list], inplace=True)
# usenergy_dff.drop(usenergy_dff.index[drop_list], inplace=True)
# cost_dff.drop(cost_dff.index[drop_list], inplace=True)
# opcost_dff.drop(cost_dff.index[drop_list], inplace=True)

event_dff.to_csv('events_gcf_pool.csv')
usenergy_dff.to_csv('usenergy_gcf.csv')
cost_dff.to_csv('cost_gcf.csv')
opcost_dff.to_csv('opcost_gcf.csv')

print('number of contingency scenarios is', number_of_events, 'in total')

#%% Generate the reduce event dataframe


# event_idx = [50, 35, 22, 17, 19]
# event_idx = [x - 1 for x in event_idx]
# # event_idx=[1,21]
#
# alfa_event_list = []
# for x in event_idx:
#     alfa_event_list.append(event_dff.iloc[x, 0])


#%%Generate the geography of the system

#read belgium shp file

#City location
filename='Belgium_city_GeocodeAddresse10.shp'
cities = geopandas.read_file(filename)
cities.set_index('ResultID',inplace=True)
#print(cities.head)
#print(cities.columns)
#print(cities.iloc[0,:])
#print(cities.total_bounds)
#print(cities.crs)

#Country bound
country = geopandas.read_file('AD_6_Country.shp')
country = country.to_crs(epsg=4326)
#print(country.crs)
#print(country.head)
#country.plot()
#print(country.total_bounds)
#create compressor geodataframe
compressor_gdf = cities.loc[[9,18],:]
#compressor_gdf.head

# power node locationon
power_location_df = pd.read_csv('power_node_location.csv')
power_location_df.set_index('Node',inplace=True)

node_name=list(range(1,15))
power_location_df['node_name']=node_name

power_location_gdf = geopandas.GeoDataFrame(power_location_df,geometry=geopandas.points_from_xy(power_location_df.Longitude, power_location_df.Latitude))
#power_location_gdf = power_location_gdf.set_crs(epsg=4326)
power_location_gdf.crs = {'init' :'epsg:4326'}
#print(power_location_gdf.crs)

#create power transmission lines geodataframe
lst=[]

from busdata import *

branch_list = list(zip(branch[:,0],branch[:,1]))
for x in branch_list:
    a = int(x[0])
    b = int(x[1])
    transline = LineString(power_location_gdf.loc[a,'geometry'].coords[:]+power_location_gdf.loc[b,'geometry'].coords[:])
    lst.append(transline)
power_geoseries = GeoSeries(lst)
#print(power_geoseries)
#power_geoseries.plot()


#pipeline geodataframe
belg = {}
with open('belgian.json','r') as f:
    belg = json.load(f)
    
#create the list of pipes as well as a dictionary with keys of numberings and values of pipe names
c=1    
pipe_dict ={}
pipe_list=[]
for idx,component in belg['pipe'].items():
    print(c,idx,(component['f_junction'],component['t_junction']))
    pipe_dict[c]=idx
    pipe_list.append((component['f_junction'],component['t_junction']))
    c+=1

#Because of junctions are decomposed to represent compressors, the junctions of three pipes need to be revised
pipe_list=[(8,9) if x ==(81,9) else x for x in pipe_list]
pipe_list=[(17,18) if x==(171,18) else x for x in pipe_list]
#print(pipe_list)
    
lst=[]
for x in pipe_list:
    a = int(x[0])
    b = int(x[1])
    gasline = LineString(cities.loc[a,'geometry'].coords[:]+ cities.loc[b,'geometry'].coords[:])
    lst.append(gasline)    
gasline_geoseries=GeoSeries(lst)
#print(gasline_geoseries)
#gasline_geoseries.plot()
#gasline_geoseries[0].intersects(country.loc[0,'geometry'])

#%% Generate the geo data frame of event center or event area
event_center_list=[]
event_area_list=[]

bsp=0.3

# for value in alfa_event_list:
for idx, row in event_dff.iterrows():
    value = row[0]

    point=Point(value)
    event_center_list.append(point)
    
    x=value[0]
    y=value[1]
    polygon = Polygon([(x-bsp, y+bsp), (x-bsp, y-bsp), (x+bsp, y-bsp),(x+bsp,y+bsp)])
    event_area_list.append(polygon)
    
event_area_geo= geopandas.GeoSeries(event_area_list)
event_area_geo.crs = {'init' :'epsg:4326'}

event_center_geo= geopandas.GeoSeries(event_center_list)
event_center_geo.crs= {'init' :'epsg:4326'}

#%% plot the random box with components of the power system and gas system

fig, ax = plt.subplots()
ax.set_aspect('equal')
country.plot(ax=ax, color='white', edgecolor='black')
cities.plot(ax=ax, marker='o', color='navy', markersize=5,label='Gas Node')
# event_area_geo.plot(ax=ax,color='blue',alpha=0.4)
power_location_gdf.plot(ax=ax, color='red',markersize=5,label='Power Node')
power_geoseries.plot(ax=ax,color='green',linewidth=1,label='Power Line')
compressor_gdf.plot(ax=ax,color='yellow',markersize=7,label='Compressor')
gasline_geoseries.plot(ax=ax,color='purple',linewidth=1,label='Gas Line')
event_center_geo.plot(ax=ax,color='blue',marker="^",markersize=10,label='Event Center')

#for x, y, label in zip(power_location_gdf.geometry.x, power_location_gdf.geometry.y, power_location_gdf.node_name):
#    ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")


#plt.legend(loc="upper left")
ax.legend(loc='best')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Coupled Gas and Power Network')  
#plt.savefig('scenarios_cost.png',dpi=1000)  
# plt.savefig('cost5.png',dpi=1000)
plt.savefig('gcf_100.png', dpi = 1000)
plt.show()
        
#%% Create the frequency graph of number of failed components in gcf runs

fc_nu=list()

for idx,row in event_dff.iterrows():
    c_lst=row[1]
    fc_nu.append(len(c_lst))

plt.hist(fc_nu,bins=35,color='#0504aa',alpha=0.7) #density false makes counts
plt.ylabel('Probability')
plt.xlabel('Number of Failed Components')
plt.xticks(np.arange(0, 18, step=1)) 
plt.title('Histogram of Number of Failed Components for 100 Runs')
plt.savefig('Histogram of Number of Failed Components.png',dpi=600)

#add a pdf line # DOES NOT WORK BECAUSE IT IS NOT CONTINUOUS
# import scipy.stats as st
# plt.hist(fc_nu,density=True,bins=30,label='data')
# mn, mx = plt.xlim()
# plt.xlim(mn, mx)
# kde_xs = np.linspace(mn, mx, 301)
# kde = st.gaussian_kde(fc_nu)
# plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
# plt.legend(loc="upper left")
# plt.ylabel('Probability')
# plt.xlabel('Data')
# plt.title("Histogram")













