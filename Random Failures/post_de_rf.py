#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:00:20 2020

@author: carriesu
"""

# This file is used to process results of deterministic equaivalent model in random failure setting
# and draw the results of the deterministic equivalent expansion model

#%% Import pkg
import pandas as pd
import matplotlib.pyplot as plt
import json

# Geographics related packages
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon, LineString

# df1=pd.read_csv('df1_mc24.csv',index_col=0)

# %%Input the investment decisions from deterministic equivalent model
# ne_list = [101,13,19,20,21,221,23,7,(10,11),(12,13),(9,10),(9,14)] # cost 10 expansion decisions
ne_list = [111,12,19,20,21,221,23,24,7,(6,12),(6,13),(9,10)]

net_list = []  # Transmission line expansion choices
nep_list = []  # Pipeline expansion choices

for ne in ne_list:
    if type(ne) is tuple:
        net_list.append(ne)
    else:
        nep_list.append(ne)

print(nep_list)
print(net_list)

# %% Read scenario text into a list
alfa_list = []
f = open('cost_sample10_rf.txt')
for line in f:
    alfa_list.append(int(line.strip('\n')))

index_list = []
for i in alfa_list:
    index_list.append(i - 1)

# selected_df1=pd.read_csv('reduced_scenarios.csv',index_col=0)


# %% Slice the events dataframe into selected dataframe

event_df = pd.read_csv('events_rf.csv', index_col=0)
selected_df = event_df.iloc[index_list, :]

# selected_df=event_df #for trying to draw a figure with all event centers

# %% read belgium shp file

# City location
filename = 'Belgium_city_GeocodeAddresse10.shp'
cities = geopandas.read_file(filename)
cities.set_index('ResultID', inplace=True)
# print(cities.head)
# print(cities.columns)
# print(cities.iloc[0,:])
# print(cities.total_bounds)
# print(cities.crs)

# Country bound
country = geopandas.read_file('AD_6_Country.shp')
country = country.to_crs(epsg=4326)
# print(country.crs)
# print(country.head)
# country.plot()
# print(country.total_bounds)

# %%create compressor geodataframe

compressor_gdf = cities.loc[[9, 18], :]
# compressor_gdf.head

# %% power node locationon

power_location_df = pd.read_csv('power_node_location.csv')
power_location_df.set_index('Node', inplace=True)

node_name = list(range(1, 15))
power_location_df['node_name'] = node_name

power_location_gdf = geopandas.GeoDataFrame(power_location_df,
                                            geometry=geopandas.points_from_xy(power_location_df.Longitude,
                                                                              power_location_df.Latitude))
power_location_gdf = power_location_gdf.set_crs(epsg=4326)
# power_location_gdf.crs = {'init': 'epsg:4326'}
# print(power_location_gdf.crs)

# %%create power transmission lines geodataframe
lst = []

from busdata import *

branch_list = list(zip(branch[:, 0], branch[:, 1]))
for x in branch_list:
    a = int(x[0])
    b = int(x[1])
    transline = LineString(
        power_location_gdf.loc[a, 'geometry'].coords[:] + power_location_gdf.loc[b, 'geometry'].coords[:])
    lst.append(transline)
power_geoseries = GeoSeries(lst)
print(power_geoseries)
# power_geoseries.plot()

# %%Create the geoseries of new transmission lines
lst = [] # temporary list to store transmission line LineString

for x in net_list:
    a = int(x[0])
    b = int(x[1])
    transline = LineString(
        power_location_gdf.loc[a, 'geometry'].coords[:] + power_location_gdf.loc[b, 'geometry'].coords[:])
    lst.append(transline)
new_power_geoseries = GeoSeries(lst)

# new_power_geoseries.plot()
# print(new_power_geoseries)


# %%pipeline geodataframe

belg = {}
with open('belgian.json', 'r') as f:
    belg = json.load(f)

# create the list of pipes as well as a dictionary with keys of numberings and values of pipe names
c = 1
pipe_dict = {}
pipe_list = []
for idx, component in belg['pipe'].items():
    print(c, idx, (component['f_junction'], component['t_junction']))
    pipe_dict[c] = idx
    pipe_list.append((component['f_junction'], component['t_junction']))
    c += 1

# Because of junctions are decomposed to represent compressors, the junctions of three pipes need to be revised
pipe_list = [(8, 9) if x == (81, 9) else x for x in pipe_list]
pipe_list = [(17, 18) if x == (171, 18) else x for x in pipe_list]
# print(pipe_list)

lst = []
for x in pipe_list:
    a = int(x[0])
    b = int(x[1])
    gasline = LineString(cities.loc[a, 'geometry'].coords[:] + cities.loc[b, 'geometry'].coords[:])
    lst.append(gasline)
gasline_geoseries = GeoSeries(lst)
# print(gasline_geoseries)
# gasline_geoseries.plot()
# gasline_geoseries[0].intersects(country.loc[0,'geometry'])


# %%Create the geoseries of new pipelines
nep_list = [str(x) for x in nep_list]
pipe_list = []

for idx, component in belg['pipe'].items():
    if idx in nep_list:
        pipe_list.append((component['f_junction'], component['t_junction']))
    else:
        continue

# print(pipe_list)
pipe_list = [(8, 9) if x == (81, 9) else x for x in pipe_list]
pipe_list = [(17, 18) if x == (171, 18) else x for x in pipe_list]
# print(pipe_list)

lst = [] # Store temp
for x in pipe_list:
    a = int(x[0])
    b = int(x[1])
    gasline = LineString(cities.loc[a, 'geometry'].coords[:] + cities.loc[b, 'geometry'].coords[:])
    lst.append(gasline)
new_gasline_geoseries = GeoSeries(lst)
print(new_gasline_geoseries)
# new_gasline_geoseries.plot()


# %% plot the system expansion

fig, ax = plt.subplots()
ax.set_aspect('equal')
country.plot(ax=ax, color='white', edgecolor='black')
cities.plot(ax=ax, marker='o', color='navy', markersize=5, label='Gas Node')
power_location_gdf.plot(ax=ax, color='red', markersize=5, label='Power Node')
power_geoseries.plot(ax=ax, color='green', linewidth=1, label='Power Line')
compressor_gdf.plot(ax=ax, color='yellow', markersize=7, label='Compressor')
gasline_geoseries.plot(ax=ax, color='purple', linewidth=1, label='Gas Line')
# new_power_geoseries.plot(ax=ax,color='darkgreen',linewidth=2)
# new_gasline_geoseries.plot(ax=ax,color='indigo',linewidth=2)

for x, y, label in zip(power_location_gdf.geometry.x, power_location_gdf.geometry.y, power_location_gdf.node_name):
   ax.annotate(label, xy=(x, y), xytext=(1, 1), color='red',textcoords="offset points")

for x, y, label in zip(cities.geometry.x, cities.geometry.y, cities.index):
   ax.annotate(label, xy=(x, y), xytext=(1.5, 1.5),color='navy', textcoords="offset points")

# plt.legend(loc="upper left")
ax.legend(loc='best')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# ax.set_title('Coupled Gas and Power Network Expansion RF USE')
ax.set_title('Joint Power and Gas System with Node Labels')
# plt.savefig('System Expansion Random Failure USE10.png',dpi=1000)
plt.savefig('Joint power and gas sytem with node labels',dpi=600)
plt.show()


