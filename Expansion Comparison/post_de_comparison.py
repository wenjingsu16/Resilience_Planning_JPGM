#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:00:20 2020

@author: carriesu
"""

# This file is used to draw the expansion results from deterministic equivalent model for GCF and RF
# The reduced scenarios can be draw in pictures

# The expansion results are from expansion results from the csv file

#%% Import pkg
import pandas as pd

# Geographics related packages
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon, LineString

import matplotlib.pyplot as plt
import json



# %% Slice the events dataframe into selected dataframe

event_df1 = pd.read_csv('events_gcf.csv', index_col=0)
event_df2 = pd.read_csv('events_rf.csv',index_col=0)

expansion_df = pd.read_csv('Comparison of Expansion Decision of RF and GCF.csv')
# selected_df = event_df.iloc[index_list, :]

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

# %% Visualization of the expansion decisions from the csv file of expansion deicions
# each row is visualized and saved as a figure
# expansion_df = pd.read_csv('Comparison of Expansion Decision of RF and GCF.csv')


for idx, row in expansion_df.iterrows():

    net_list = eval(row[2])
    nep_list = eval(row[3])

    scenario_name = row[0]

    # Create the geoseries of new transmission lines
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

    # Create the geoseries of new pipelines
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

    #  plot the random box with components

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    country.plot(ax=ax, color='white', edgecolor='black')
    cities.plot(ax=ax, marker='o', color='navy', markersize=5, label='Gas Node')
    power_location_gdf.plot(ax=ax, color='red', markersize=5, label='Power Node')
    power_geoseries.plot(ax=ax, color='green', linewidth=1, label='Power Line')
    gasline_geoseries.plot(ax=ax, color='purple', linewidth=1, label='Gas Line')
    new_power_geoseries.plot(ax=ax,color='darkgreen',linewidth=2)
    new_gasline_geoseries.plot(ax=ax,color='indigo',linewidth=2)

    # plt.legend(loc="upper left")
    ax.legend(loc='best',fontsize=9)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(scenario_name+' Coupled Gas and Power Network')
    plt.savefig(scenario_name+'_expansion'+'.png',dpi=500)
    plt.show()

# %% Visualize a GCF extrme event with an expanded network

expansion_df = pd.read_csv('Comparison of Expansion Decision of RF and GCF.csv',index_col=0) # previously index_col is not 0

# Choose the expanded network to visualize
exp_choice = 'GCF10'

# Choose number of scenario to be drew
scenario_list = [98]

net_list = eval(expansion_df.loc[exp_choice,'New Transmission Line'])
nep_list = eval(expansion_df.loc[exp_choice,'New Pipeline'])

scenario_name = exp_choice

# Create the geoseries of new transmission lines
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

# Create the geoseries of new pipelines
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

# Draw the gcf event center and area
event_center_list = []
event_area_list = []

bsp = 0.3

for idx in scenario_list:
    value = event_df1.iloc[idx,0]

    point = Point(eval(value))
    event_center_list.append(point)

    x = eval(value)[0]
    y = eval(value)[1]
    polygon = Polygon([(x - bsp, y + bsp), (x - bsp, y - bsp), (x + bsp, y - bsp), (x + bsp, y + bsp)])
    event_area_list.append(polygon)

event_area_geo = geopandas.GeoSeries(event_area_list)
event_area_geo.crs = {'init': 'epsg:4326'}

event_center_geo = geopandas.GeoSeries(event_center_list)
event_center_geo.crs = {'init': 'epsg:4326'}

# Draw ALFA event area and center
alfa_list = [] #list of ALFA selected scenarios in MATLAB index
f = open('gcf_cost10.txt')
for line in f:
    alfa_list.append(int(line.strip('\n')))

index_list = []
for i in alfa_list:
    index_list.append(i - 1)
selected_df = event_df1.iloc[index_list,:]

# Create geoseries of alfa event center and alfa event area
alfa_center_list = []
alfa_area_list = []

bsp = 0.3

for idx, value in selected_df.iloc[:, 0].iteritems():
    point = Point(eval(value))
    alfa_center_list.append(point)

    x = eval(value)[0]
    y = eval(value)[1]
    polygon = Polygon([(x - bsp, y + bsp), (x - bsp, y - bsp), (x + bsp, y - bsp), (x + bsp, y + bsp)])
    alfa_area_list.append(polygon)

alfa_area_geo = geopandas.GeoSeries(alfa_area_list)
alfa_area_geo.crs = {'init': 'epsg:4326'}

alfa_center_geo = geopandas.GeoSeries(alfa_center_list)
alfa_center_geo.crs = {'init': 'epsg:4326'}

fig, ax = plt.subplots()
ax.set_aspect('equal')
country.plot(ax=ax, color='white', edgecolor='black')
cities.plot(ax=ax, marker='o', color='navy', markersize=5, label='Gas Node')
power_location_gdf.plot(ax=ax, color='red', markersize=5, label='Power Node')
power_geoseries.plot(ax=ax, color='green', linewidth=1, label='Power Line')
gasline_geoseries.plot(ax=ax, color='purple', linewidth=1, label='Gas Line')

new_power_geoseries.plot(ax=ax, color='darkgreen', linewidth=2)
new_gasline_geoseries.plot(ax=ax, color='indigo', linewidth=2)

event_area_geo.plot(ax=ax, color='indianred', alpha=0.4)
event_center_geo.plot(ax=ax, color='indianred', marker="^", markersize=10, label='Event Center')
alfa_area_geo.plot(ax=ax, color='blue', alpha=0.4)
alfa_center_geo.plot(ax=ax, color='blue', marker="^", markersize=10, label='Event Center')

# plt.legend(loc="upper left")
ax.legend(loc='best', fontsize=9)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('GCF' + scenario_name + ' Coupled Gas and Power Network')
# plt.savefig(scenario_name + '_expansion' + '.png', dpi=500)
plt.show()

