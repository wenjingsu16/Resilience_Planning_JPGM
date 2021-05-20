# This script is used to generate the padd and tadd text file based on the expansion decision CSV file

# %%

import pandas as pd
import json

from busdata import *

# %%
col = 2

branch_list = list(zip(branch[:, 0], branch[:, 1]))

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

# %%
tadd = []
padd = []

# Status parameters are determined by the expansion decisions
exp_df = pd.read_csv('Comparison of Expansion Decision of RF and GCF.csv')

#The number of columsn matches the number in pool function, set at 6 for now
for i in range(col):
    ttemp=[]
    ptemp=[]


    net_list = eval(exp_df.iloc[i,2])
    nep_list = eval(exp_df.iloc[i,3])
    nep_list = [str(x) for x in nep_list]

    for j in branch_list:
        if j in net_list:
            ttemp.append(1)
        else:
            ttemp.append(0)
    tadd.append(ttemp)

    for k in list(belg['pipe'].keys()):
        if k in nep_list:
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
#
