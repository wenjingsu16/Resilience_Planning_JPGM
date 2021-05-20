#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:16:23 2020

@author: carriesu
"""
# The script is to plot the random failure scenarios selected by ALFA.
# Extract events dataframe into selected events dataframe


# %% import pkg

import pandas as pd
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon, LineString
import json
import matplotlib.pyplot as plt

# %% Read in csv file and alfa selected scenarios
event_dff = pd.read_csv('events_rf.csv',index_col=0)

alfa_list = [] #list of ALFA selected scenarios in MATLAB index
# f = open('rf_cost8.txt')
f = open('rf_cost10.txt')
# f = open('use_sample10_rf.txt')
for line in f:
    alfa_list.append(int(line.strip('\n')))

index_list = []
for i in alfa_list:
    index_list.append(i - 1)

# %% print out the reduced event dataframe
# re: reducted events based on unserved energy or cost
selected_df = event_dff.iloc[index_list,:]
# selected_df.to_csv('re_cost10_rf.csv')
# selected_df.to_csv('re_cost8_rf.csv')
selected_df.to_csv('re_cost10_rf.csv')


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


# %% plot the base map of belgium gas and power system
fig, ax = plt.subplots()
ax.set_aspect('equal')
country.plot(ax=ax, color='white', edgecolor='black')
cities.plot(ax=ax, marker='o', color='navy', markersize=5, label='Gas Node')
# event_area_geo.plot(ax=ax, color='blue', alpha=0.4)
power_location_gdf.plot(ax=ax, color='red', markersize=5, label='Power Node')
power_geoseries.plot(ax=ax, color='green', linewidth=1, label='Power Line')
compressor_gdf.plot(ax=ax, color='yellow', markersize=7, label='Compressor')
gasline_geoseries.plot(ax=ax, color='purple', linewidth=1, label='Gas Line')
# event_center_geo.plot(ax=ax, color='blue', marker="^", markersize=10, label='Event Center')

# for x, y, label in zip(power_location_gdf.geometry.x, power_location_gdf.geometry.y, power_location_gdf.node_name):
#    ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")


# plt.legend(loc="upper left")
ax.legend(loc='best')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Coupled Gas and Power Network')
# plt.savefig('scenarios_cost.png',dpi=1000)
# plt.savefig('rf_cost5.png',dpi=800)
plt.savefig('Base network',dpi=600)
plt.show()


# %% Draw ALFA selected events
index_list =[98]

for nu in index_list:
    selected_row_no=nu

    #selected_row_no = 1  # of the event dataframe will be drawn
    fc_list = eval(event_dff.iloc[selected_row_no, 1])

    fc_power_list = []
    fc_gas_list = []

    for fc in fc_list:
        if type(fc) is tuple:
            fc_power_list.append(fc)
        else:
            fc_gas_list.append(fc)

    # Create the geoseries of failed power lines
    lst = []
    for x in fc_power_list:
        a = int(x[0])
        b = int(x[1])
        transline = LineString(
            power_location_gdf.loc[a, 'geometry'].coords[:] + power_location_gdf.loc[b, 'geometry'].coords[:])
        lst.append(transline)
    fc_power_geoseries = GeoSeries(lst)

    # create the geoseries of failed gas lines
    fc_gas_list = [str(x) for x in fc_gas_list]

    pipe_list = []

    for idx, component in belg['pipe'].items():
        if idx in fc_gas_list:
            pipe_list.append((component['f_junction'], component['t_junction']))
        else:
            continue

    print(pipe_list)
    pipe_list = [(8, 9) if x == (81, 9) else x for x in pipe_list]
    pipe_list = [(17, 18) if x == (171, 18) else x for x in pipe_list]
    print(pipe_list)

    lst = []
    for x in pipe_list:
        a = int(x[0])
        b = int(x[1])
        gasline = LineString(cities.loc[a, 'geometry'].coords[:] + cities.loc[b, 'geometry'].coords[:])
        lst.append(gasline)
    fc_gas_geoseries = GeoSeries(lst)
    print(fc_gas_geoseries)
    # fc_gas_geoseries.plot()
    # plt.show()

    # Plot the map with randomly selected component failure
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    country.plot(ax=ax, color='white', edgecolor='black')
    cities.plot(ax=ax, marker='o', color='navy', markersize=5, label='Gas Node')
    power_location_gdf.plot(ax=ax, color='red', markersize=5, label='Power Node')
    power_geoseries.plot(ax=ax, color='green', linewidth=1, label='Power Line')
    compressor_gdf.plot(ax=ax, color='yellow', markersize=7)
    gasline_geoseries.plot(ax=ax, color='purple', linewidth=1, label='Gas Line')
    fc_power_geoseries.plot(ax=ax, color='mediumblue', linewidth=2, label='Failed Power Lines')
    fc_gas_geoseries.plot(ax=ax, color='darkviolet', linewidth=2, label='Failed Gas Pipes')

    # for x, y, label in zip(power_location_gdf.geometry.x, power_location_gdf.geometry.y, power_location_gdf.node_name):
    #    ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")


    # plt.legend(loc="upper left")
    ax.legend(loc='lower left', fontsize='x-small')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('UCF %d in Coupled Gas and Power Network ' %nu )
    # plt.savefig('scenarios_cost.png',dpi=1000)
    plt.savefig('ALFA UCF Event %d.png' %nu, dpi=400)
    plt.show()
