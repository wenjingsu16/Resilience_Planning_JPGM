# This file is used to generate the geographically correlated failures scenarios
# the output of the code is the csv file with event location and failed component lists
# and generate network expansion columns in padd/tadd

# %% import pkgs

import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon, LineString

import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import random

from busdata import *



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
country = country.to_crs('epsg:4326')
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

power_location_gdf = geopandas.GeoDataFrame(power_location_df,
                                            geometry=geopandas.points_from_xy(power_location_df.Longitude,
                                                                              power_location_df.Latitude))
power_location_gdf = power_location_gdf.set_crs('epsg:4326')
# power_location_gdf.crs = {'init': 'epsg:4326'}
# print(power_location_gdf.crs)

# %%create power transmission lines geodataframe
lst = []

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

# %% Generate random event area
import random

bound_list = country.total_bounds
event_location_list = []
fc_list_column = []

# The dictionary to document the event location and list of failed components
contingency_dict = {}

# box size parameter, 1/2 of the edge length (degree)
bsp = 0.3

##loop through 100 times to get random location and a list of failed components list

while len(event_location_list) < 1:

    x = random.uniform(bound_list[0], bound_list[2])
    y = random.uniform(bound_list[1], bound_list[3])
    # print((x,y))
    location = (x, y)
    location_geo = Point(x,y)

    polygon = Polygon([(x - bsp, y + bsp), (x - bsp, y - bsp), (x + bsp, y - bsp), (x + bsp, y + bsp)])
    event_geo = GeoSeries([polygon])
    event_geo = event_geo.set_crs('epsg:4326')

    # if polygon.intersects(country.iloc[0, 8]):
    if location_geo.within(country.iloc[0, 8]):
        fc_list = []
        n = 0  # number for counting
        event_location_list.append((x, y))

        # transmission line in the event area
        for x in power_geoseries:
            n += 1
            if x.intersects(polygon):
                fc_list.append(n)
            else:
                continue

        # pipelines in the event area
        for x in gasline_geoseries:
            n += 1
            if x.intersects(polygon):
                fc_list.append(n)
            else:
                continue
        # print(fc_list)
    else:
        continue

    fc_list_column.append(fc_list)
    contingency_dict.update({location: fc_list})

print(event_location_list)
print(contingency_dict)
# event_location_list = list(contingency_dict.keys())

gcf_df = pd.DataFrame(data={'event center':event_location_list,'failed component':fc_list_column})

# %% Create the column of names of failed components
fc_lst = []  # create a list of list to document the name of failed components
# translate the number into the components name
for idx, row in gcf_df.iterrows():
    gcf_temp = []

    for j in row[1]:
        if j >= 1 and j <= 20:  # 1-20 is transmission lines
            line = (branch[j - 1, 0], branch[j - 1, 1])
            gcf_temp.append(line)
        else:  # 21 to 44 is gas pipelines
            pipe = pipe_dict[j - 20]
            gcf_temp.append(pipe)
    fc_lst.append(gcf_temp)

gcf_df['Name of Failed Components'] = fc_lst

gcf_df.to_csv('events_gcf.csv')

# %% Generate the geodataframe of event center and event area
event_center_list = []
event_area_list = []

bsp = 0.3

# for value in alfa_event_list:
for idx, row in gcf_df.iterrows():
    value = row[0]

    point = Point(value)
    event_center_list.append(point)

    x = value[0]
    y = value[1]
    polygon = Polygon([(x - bsp, y + bsp), (x - bsp, y - bsp), (x + bsp, y - bsp), (x + bsp, y + bsp)])
    event_area_list.append(polygon)

event_area_geo = geopandas.GeoSeries(event_area_list)
event_area_geo = event_area_geo.set_crs('epsg:4326')

event_center_geo = geopandas.GeoSeries(event_center_list)
event_center_geo = event_center_geo.set_crs('epsg:4326')



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
event_center_geo.plot(ax=ax, color='blue', marker="^", markersize=10, label='Event Center')

# for x, y, label in zip(power_location_gdf.geometry.x, power_location_gdf.geometry.y, power_location_gdf.node_name):
#    ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")


# plt.legend(loc="upper left")
ax.legend(loc='best')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Coupled Gas and Power Network with 100 GCF Events')
plt.savefig('gcf_100events.png',dpi=600)
plt.show()

# %% Draw the histogram of number of failed components
fc_nu=list()

for idx,row in gcf_df.iterrows():
    c_lst=row[1]
    fc_nu.append(len(c_lst))

plt.hist(fc_nu,bins=40,color='#0504aa',alpha=0.7) #density false makes counts
plt.ylabel('Probability')
plt.xlabel('Number of Failed Components')
plt.xticks(np.arange(0, 18, step=1))
plt.title('Histogram of Number of Failed Components for 100 Runs')
plt.savefig('Histogram of Number of Correlated Failed Components.png',dpi=600)
plt.show()

# %% Generate the network expansion columns
col = 30

tadd = []
padd = []

#The number of columsn matches the number in pool function, set at 30 for now
for i in range(col):
    ttemp=[]
    ptemp=[]

    for j in range(20):
        if random.random()<=1/20:
            ttemp.append(1)
        else:
            ttemp.append(0)
    tadd.append(ttemp)

    for k in range(24):
        if random.random()<=1/24:
            ptemp.append(1)
        else:
            ptemp.append(0)
    padd.append(ptemp)

print('tadd: ',tadd)
print(('padd: ',padd))

with open('tadd.txt','w') as f:
    f.writelines(str(tadd))

with open('padd.txt','w') as f:
    f.writelines(str(padd))

# Draw the histogram of number of network additions
add_nu=list()

for x in range(col):
    t_ct = np.count_nonzero(np.array(tadd[x]))
    p_ct = np.count_nonzero(np.array(padd[x]))
    add_nu.append(p_ct + t_ct)

plt.hist(add_nu,bins=20,color='#0504aa',alpha=0.7) #density false makes counts
plt.ylabel('Frequency')
plt.xlabel('Number of Network Additions')
# plt.xticks(np.arange(0, 18, step=1))
plt.title('Histogram of Number of Network Additions for 30 Columns')
plt.savefig('Histogram of Number of Network Additions.png',dpi=600)
plt.show()