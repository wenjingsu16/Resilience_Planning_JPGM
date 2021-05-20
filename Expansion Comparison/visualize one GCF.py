# This script is to visualize one GCF event and the failed component for base network

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



# %% Visualize a GCF extrme event with base network

# Choose number of scenario to be drew
scenario = 98

fct_list = []
fcp_list = []

fc_list = event_df1.iloc[scenario,2]
for x in eval(fc_list):
    if type(x) is tuple:
        fct_list.append(x)
    else:
        fcp_list.append(x)
print(fct_list)
print(fcp_list)


scenario_name = scenario


# Create the geoseries of new transmission lines
lst = [] # temporary list to store transmission line LineString

for x in fct_list:
    a = int(x[0])
    b = int(x[1])
    transline = LineString(
        power_location_gdf.loc[a, 'geometry'].coords[:] + power_location_gdf.loc[b, 'geometry'].coords[:])
    lst.append(transline)
failed_power_geoseries = GeoSeries(lst)

# new_power_geoseries.plot()
# print(new_power_geoseries)

# Create the geoseries of new pipelines
fcp_list = [str(x) for x in fcp_list]
pipe_list = []

for idx, component in belg['pipe'].items():
    if idx in fcp_list:
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
failed_gasline_geoseries = GeoSeries(lst)
print(failed_gasline_geoseries)
# new_gasline_geoseries.plot()

# Draw the gcf event center and area
event_center_list = []
event_area_list = []

bsp = 0.3


value = event_df1.iloc[scenario,0]

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


fig, ax = plt.subplots()
ax.set_aspect('equal')
country.plot(ax=ax, color='white', edgecolor='black')
cities.plot(ax=ax, marker='o', color='navy', markersize=5, label='Gas Node')
power_location_gdf.plot(ax=ax, color='red', markersize=5, label='Power Node')
power_geoseries.plot(ax=ax, color='green', linewidth=1, label='Power Line')
gasline_geoseries.plot(ax=ax, color='purple', linewidth=1, label='Gas Line')

failed_power_geoseries.plot(ax=ax, color='darkgreen', linewidth=2,label='Failed Power Line')
failed_gasline_geoseries.plot(ax=ax, color='indigo', linewidth=2,label='Failed Gas Line')

event_area_geo.plot(ax=ax, color='indianred', alpha=0.4,label='Event Area')
event_center_geo.plot(ax=ax, color='indianred', marker="^", markersize=10, label='EWCE Event Center')
# alfa_area_geo.plot(ax=ax, color='blue', alpha=0.4)
# alfa_center_geo.plot(ax=ax, color='blue', marker="^", markersize=10, label='Event Center')

# plt.legend(loc="upper left")
ax.legend(loc='lower left', fontsize=9)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('GCF Scenario %i in Coupled Gas and Power Network' %scenario)
plt.savefig('GCF Scenario %i in Coupled Gas and Power System.png' %scenario,dpi=500)
# plt.savefig(scenario_name + '_expansion' + '.png', dpi=500)
plt.show()