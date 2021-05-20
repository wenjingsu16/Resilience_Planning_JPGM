#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:16:23 2020

@author: carriesu
"""
# The script is to plot the random failure scenarios selected by ALFA.
# Extract events dataframe into selected events dataframe


# %% import packages

import pandas as pd
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon, LineString
import json
import matplotlib.pyplot as plt

# %% Read in csv file and alfa selected scenarios
event_dff = pd.read_csv('events_gcf.csv',index_col=0)

alfa_list = [] #list of ALFA selected scenarios in MATLAB index
f = open('gcf_cost15.txt')
# f = open('gcf_cost8.txt')
# f = open('gcf_usenergy10.txt')
# f = open('gcf_cost5.txt')
for line in f:
    alfa_list.append(int(line.strip('\n')))

index_list = []
for i in alfa_list:
    index_list.append(i - 1)

# %% print out the reduced event dataframe
# re: reducted events based on unserved energy or cost
selected_df = event_dff.iloc[index_list,:]
selected_df.to_csv('re_cost15_gcf.csv')
# selected_df.to_csv('re_cost5_gcf.csv')
# selected_df.to_csv('re_use10_gcf.csv')


# %% Generate the geographic representation of the system

# read belgium shp file
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
# create compressor geodataframe
compressor_gdf = cities.loc[[9, 18], :]
# compressor_gdf.head

# power node locationon


power_location_df = pd.read_csv('power_node_location.csv')
power_location_df.set_index('Node', inplace=True)

node_name = list(range(1, 15))
power_location_df['node_name'] = node_name

power_location_gdf = geopandas.GeoDataFrame(power_location_df,
                                            geometry=geopandas.points_from_xy(power_location_df.Longitude,
                                                                              power_location_df.Latitude))
# power_location_gdf = power_location_gdf.set_crs(epsg=4326)
power_location_gdf.crs = {'init': 'epsg:4326'}
# print(power_location_gdf.crs)

# create power transmission lines geodataframe
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
# print(power_geoseries)
# power_geoseries.plot()


# pipeline geodataframe

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

# %% Generate the geodataframe of event center and event area
event_center_list = []
event_area_list = []

bsp = 0.3

# for value in alfa_event_list:
for idx, row in selected_df.iterrows():
    value = eval(row[0])

    point = Point(value)
    event_center_list.append(point)

    x = value[0]
    y = value[1]
    polygon = Polygon([(x - bsp, y + bsp), (x - bsp, y - bsp), (x + bsp, y - bsp), (x + bsp, y + bsp)])
    event_area_list.append(polygon)

event_area_geo = geopandas.GeoSeries(event_area_list)
event_area_geo.crs = {'init': 'epsg:4326'}

event_center_geo = geopandas.GeoSeries(event_center_list)
event_center_geo.crs = {'init': 'epsg:4326'}



# %% plot the base map of belgium gas and power system
fig, ax = plt.subplots()
ax.set_aspect('equal')
country.plot(ax=ax, color='white', edgecolor='black')
cities.plot(ax=ax, marker='o', color='navy', markersize=5, label='Gas Node')
event_area_geo.plot(ax=ax, color='blue', alpha=0.4)
power_location_gdf.plot(ax=ax, color='red', markersize=5, label='Power Node')
power_geoseries.plot(ax=ax, color='green', linewidth=1, label='Power Line')
compressor_gdf.plot(ax=ax, color='yellow', markersize=7, label='Compressor')
gasline_geoseries.plot(ax=ax, color='purple', linewidth=1, label='Gas Line')
event_center_geo.plot(ax=ax, color='blue', marker="^", markersize=10, label='Event Center')

# for x, y, label in zip(power_location_gdf.geometry.x, power_location_gdf.geometry.y, power_location_gdf.node_name):
#    ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")


# plt.legend(loc="upper left")
ax.legend(loc='best')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# ax.set_title('Coupled Gas and Power Network')
plt.savefig('gcf_cost15.png',dpi=600)
# plt.savefig('gcf_cost5.png',dpi=600)
plt.show()



