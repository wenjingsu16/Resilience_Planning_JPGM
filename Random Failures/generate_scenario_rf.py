#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 23:56:24 2020

@author: carriesu
"""

# This code is used to read in the results of geographically correlated failures
# and generate random failure event dataframe

# %%
import pandas as pd
import random
import json
import matplotlib.pyplot as plt
import numpy as np

# %% Read in event dataframe of geographically correlated failures

cf = pd.read_csv('events_gcf.csv', index_col=0)
# the second column is the list of number of failed components
# [1,20] are transmission lines and [21,44] are pipelines

# %%
# create a list of lists to store the randome failure (RF) component numbers

rf_lst = []

for idx, row in cf.iterrows():
    rf_temp = []  # document the number of failed components for each row

    for x in eval(row[1]):
        if int(x) < 21:
            rf_temp.append(random.randrange(1, 21, 1))
        else:
            rf_temp.append(random.randrange(21, 45, 1))  # [start,stop)

    rf_lst.append(str(rf_temp))

rf = pd.DataFrame(rf_lst, columns=['Numbering of Failed Components'])

# %% translate the number into components

# get the data for power system and gas system
from busdata import *

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

# %% Create the column of names of failed components
fc_lst = []  # create a list of list to document the name of failed components
# translate the number into the components name
for idx, row in rf.iterrows():
    rf_temp = []

    for j in eval(row[0]):
        if j >= 1 and j <= 20:  # 1-20 is transmission lines
            line = (branch[j - 1, 0], branch[j - 1, 1])
            rf_temp.append(line)
        else:  # 21 to 44 is gas pipelines
            pipe = pipe_dict[j - 20]
            rf_temp.append(pipe)
    fc_lst.append(rf_temp)

rf['Name of Failed Components'] = fc_lst
rf.to_csv('events_rf.csv')

# %% Draw the histogram of number of failed components

fc_nu=list()

for idx,row in rf.iterrows():
    c_lst=row[1]
    fc_nu.append(len(c_lst))

plt.hist(fc_nu,bins=35,color='#0504aa',alpha=0.7) #density false makes counts
plt.ylabel('Probability')
plt.xlabel('Number of Failed Components')
plt.xticks(np.arange(0, 18, step=1))
plt.title('Histogram of Number of Failed Components for 100 UCF Runs')
plt.savefig('Histogram of Number of Uncorrelated Failed Components.png',dpi=600)
plt.show()