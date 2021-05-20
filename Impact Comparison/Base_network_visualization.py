#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:00:20 2020

@author: carriesu
"""

# This file is used to draw the base network structure with the node labels

#%% Import pkg
import pandas as pd
import matplotlib.pyplot as plt

# Geographics related packages
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon, LineString

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
import pandas as pd

power_location_df = pd.read_csv('power_node_location.csv')
power_location_df.set_index('Node', inplace=True)

node_name = list(range(1, 15))
power_location_df['node_name'] = node_name

power_location_gdf = geopandas.GeoDataFrame(power_location_df,
                                            geometry=geopandas.points_from_xy(power_location_df.Longitude,
                                                                              power_location_df.Latitude))
# power_location_gdf = power_location_gdf.set_crs(epsg=4326)
power_location_gdf = power_location_gdf.set_crs(epsg=4326)
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

# %%pipeline geodataframe
import json

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

# %% dataframe of circles
circle_lst = {'Longitude': [3.579, 4.805], 'Latitude': [51.16, 50.52]}
circle_df = pd.DataFrame(data=circle_lst)
circle_gdf = geopandas.GeoDataFrame(circle_df,
                                            geometry=geopandas.points_from_xy(circle_df.Longitude,
                                                                              circle_df.Latitude))
circle_gdf['geometry'] = circle_gdf['geometry'].buffer(0.1)



# %% plot the system

fig, ax = plt.subplots()
ax.set_aspect('equal')
country.plot(ax=ax, color='white', edgecolor='black')
# circle_gdf.plot(ax=ax,edgecolor='chocolate',color='bisque')#,alpha=0.6)
# cities.plot(ax=ax, marker='o', color='navy', markersize=5, label='Gas Node')
power_location_gdf.plot(ax=ax, color='red', markersize=5, label='Power Node')
power_geoseries.plot(ax=ax, color='green', linewidth=1, label='Power Line')
# compressor_gdf.plot(ax=ax, color='yellow', markersize=7, label='Compressor')
# gasline_geoseries.plot(ax=ax, color='purple', linewidth=1, label='Gas Line')

#
for x, y, label in zip(power_location_gdf.geometry.x, power_location_gdf.geometry.y, power_location_gdf.node_name):
   ax.annotate(label, xy=(x, y), xytext=(1, 1), color='red',textcoords="offset points")

# for x, y, label in zip(cities.geometry.x, cities.geometry.y, cities.index):
#    ax.annotate(label, xy=(x, y), xytext=(1.5, 1.5),color='navy', textcoords="offset points")


# ax.legend(loc='best')
ax.legend(loc='lower left')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# ax.set_title('Gas Network')
# ax.set_title('Coupled Gas and Power Network Visualization')
ax.set_title('Power Network')
# plt.savefig('Gas System Locations with Labels',dpi=600)
plt.savefig('Power System with Node Labels',dpi=600)
# plt.savefig('Joint power and gas sytem with node labels_with_circles',dpi=600)
plt.show()


