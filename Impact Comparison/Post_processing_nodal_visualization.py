# This code compares the impact of geographically correlated failures and random failures
# The 100 contingency runs uses the same failures as the expansion runs

# %% Import pkgs
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import json

# %% Define gini coefficient based on gini package on github

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array += 0.0000001  # values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient


# %% Read in the results run file
gcf_df = pd.read_csv('df1_gcf100.csv')
rf_df = pd.read_csv('df1_rf100.csv')

# nodal unserved energy matrix
gcfn_df = pd.read_csv('df2_gcf100.csv', index_col=0)
rfn_df = pd.read_csv('df2_rf100.csv', index_col=0)

# %%  For nodal data, first process the data into uniform units
gas_df1 = gcfn_df.iloc[:, [1,2,3,4,5,6,8,9,10]]
power_df1 = gcfn_df.iloc[:, [12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24]]

gas_df2 = rfn_df.iloc[:, [1,2,3,4,5,6,8,9,10]]
power_df2 = rfn_df.iloc[:, [12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24]]

# Convert the dataframe into MWh of Unserved energy for a day
gas_df1 = gas_df1 * 528359.7701149426
gas_df2 = gas_df2 * 528359.7701149426

power_df1 = power_df1 * 2400
power_df2 = power_df2 * 2400

gas_df1[gas_df1<10**3] = 0
gas_df2[gas_df2<10**-3] = 0
power_df1[power_df1<10**-3] = 0
power_df2[power_df2<10**-3] = 0

# Convert the nodal unserved energy dataframe into numpy array for data processing and visualization

gas_a1 = gas_df1.to_numpy()
power_a1 = power_df1.to_numpy()

gas_a2 = gas_df2.to_numpy()
power_a2 = power_df2.to_numpy()

# Replace all the values of unserved energy less than 10**-3 with 0
gas_a1 = np.where(gas_a1 < 10 ** -3, 0, gas_a1)
power_a1 = np.where(power_a1 < 10 ** -3, 0, power_a1)
gas_a2 = np.where(gas_a2 < 10 ** -3, 0, gas_a2)
power_a2 = np.where(power_a2 < 10 ** -3, 0, power_a2)

# %% Visualize the percentage of scenarios that have unserved gas for gas consumer nodes for GCF and RF

gas_pct1 = []
gas_pct2 = []

for column in gas_a1.transpose():
    gas_pct1.append(np.count_nonzero(column))
print('gas_pct1', gas_pct1)

for column in gas_a2.transpose():
    gas_pct2.append(np.count_nonzero(column))
print('gas_pct2', gas_pct2)

gas_labels = [ 'G12', 'G20', 'G6', 'G15', 'G16', 'G7', 'G10', 'G19', 'G3']

x = np.arange(len(gas_labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
gas1 = ax.bar(x - width / 2, gas_pct1, width, label='GCF')
gas2 = ax.bar(x + width / 2, gas_pct2, width, label='UCF')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage of Scenarios')
ax.set_title('Percentage of Scenarios with Unserved Gas')
ax.set_xticks(x)
ax.set_xticklabels(gas_labels)
ax.set_xlabel('Gas Demand Nodes')
ax.legend()
# ax.set_ylim([0, 29])


def autolabel(gas_pcts):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for gas_pct in gas_pcts:
        height = gas_pct.get_height()
        if height > 0:
            ax.annotate('{}'.format(height),
                        xy=(gas_pct.get_x() + gas_pct.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


autolabel(gas1)
autolabel(gas2)

fig.tight_layout()

# plt.savefig('Comparison Percentage of Scenarios with Unserved Gas', dpi=600)
plt.show()

# %% Visualize the percentage of sceanarios that have unserved electricity for electricity consumer nodes for GCF and RF

power_pct1 = []
power_pct2 = []

for column in power_a1.transpose():
    power_pct1.append(np.count_nonzero(column))
print('power_pct1', power_pct1)

for column in power_a2.transpose():
    power_pct2.append(np.count_nonzero(column))
print('power_pct2', power_pct2)

power_labels = ['E2', 'E3', 'E4', 'E5', 'E6', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14']

x = np.arange(len(power_labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
power1 = ax.bar(x - width / 2, power_pct1, width, label='GCF')
power2 = ax.bar(x + width / 2, power_pct2, width, label='UCF')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage of Scenarios')
ax.set_title('Percentage of Scenarios with Unserved Electricity')
ax.set_xticks(x)
ax.set_xticklabels(power_labels)
ax.set_xlabel('Electricity Demand Nodes')
# ax.set_ylim([0, 15])
ax.legend(loc='upper left')


def autolabel(power_pcts):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for power_pct in power_pcts:
        height = power_pct.get_height()
        if height > 0:
            ax.annotate('{}'.format(height),
                        xy=(power_pct.get_x() + power_pct.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


autolabel(power1)
autolabel(power2)

fig.tight_layout()

plt.savefig('Comparison Percentage of Scenarios with Unserved Electricity', dpi=600)

plt.show()

# %% create the list of total demand for gas and electricy nodes so we can calculate the percentage of unsevered gas
gas_col_list = list(gas_df1.columns)

belg = {}
with open('belgian.json', 'r') as f:
    belg = json.load(f)
heat_value = 2.436 * 10 ** -8
baseQ = belg['baseQ']

gas_demand_list = []
for idx in gas_col_list:
    gas_demand = belg['consumer'][idx]['qlmin'] * 86400 * 2.78 * 10 ** (-10) * baseQ / heat_value
    if gas_demand <= 0:
        gas_demand = 1.0
    gas_demand_list.append(gas_demand)

print(gas_demand_list)

# %% create the list of the total electricity demand so we can calculate the percentage electricity
power_col_list = list(power_df1.columns)

from busdata import *
bus_df = pd.DataFrame(bus,index=list(range(1,15)))

power_demand_list = []
for idx in power_col_list:
    power_demand = bus_df.loc[int(float(idx)),2] *24 *100 # MVAbase=100
    power_demand_list.append(power_demand)

print(power_demand_list)

# %% Extract the non-zero in unserved gas or electricity to a list to draw the box plot

# GCF gas
gas_a1_lst = [] # The quantity of unserved gas
gas_p1_lst = [] # The percentage of unserved gas in nodal gas demand

ct = 0 # count the times of the iteration to extract the total gas demand
for row in gas_a1.transpose():
    lst = []
    plst = []
    for x in row:
        if x > 0:
            lst.append(x)
            plst.append(x/gas_demand_list[ct])
    ct+=1
    gas_a1_lst.append(lst)
    gas_p1_lst.append(plst)

# RF gas
gas_a2_lst = []
gas_p2_lst = []

ct=0
for row in gas_a2.transpose():
    lst = []
    plst = []
    for x in row:
        if x > 0:
            lst.append(x)
            plst.append(x/gas_demand_list[ct])
    ct+=1
    gas_p2_lst.append(plst)
    gas_a2_lst.append(lst)

# GCF power
power_a1_lst = []
power_p1_lst = []

ct = 0
for row in power_a1.transpose():
    lst = []
    plst = []
    for x in row:
        if x > 0:
            lst.append(x)
            plst.append(x/power_demand_list[ct])
    ct+=1
    power_p1_lst.append(plst)
    power_a1_lst.append(lst)

# RF power
power_a2_lst = []
power_p2_lst = []
ct = 0
for row in power_a2.transpose():
    lst = []
    plst = []
    for x in row:
        if x > 0:
            lst.append(x)
            plst.append(x/power_demand_list[ct])
    ct+=1
    power_p2_lst.append(plst)
    power_a2_lst.append(lst)

# %% Box and whisker plots for nodal unserved gas GCF

fig, ax = plt.subplots()  # figsize=(8,8))
x = np.arange(len(gas_labels))  # the label locations
ax.boxplot(gas_a1_lst)

# ax.set_xticks(x)
ax.set_xticklabels(gas_labels)
ax.set_xlabel('Gas Consumer Nodes')
ax.set_ylim((0, 95000))
ax.set_ylabel('Unserved Gas in MWh')
ax.set_title('Unserved Gas for Gas Demand Nodes GCF')

plt.tight_layout()

plt.savefig('Box and Whisker Gas GCF', dpi=500)
plt.show()

# %% Box and whisker plot for gas node rf

fig, ax = plt.subplots()  # figsize=(8,8))
x = np.arange(len(gas_labels))  # the label locations
ax.boxplot(gas_a2_lst)


# ax.set_xticks(x)
ax.set_xticklabels(gas_labels)
ax.set_xlabel('Gas Consumer Nodes')
ax.set_ylim((0, 95000))
ax.set_ylabel('Unserved Gas in MWh')
ax.set_title('Unserved Gas for Gas Demand Nodes UCF')

plt.tight_layout()

plt.savefig('Box and Whisker Gas UCF', dpi=500)
plt.show()

# %% Box and whisker plot for gas node percentage GCF

fig, ax = plt.subplots()  # figsize=(8,8))
x = np.arange(len(gas_labels))  # the label locations
ax.boxplot(gas_p1_lst)
# ax.boxplot(gas_p2_lst)
ax.set_ylim((0,1.1))


# ax.set_xticks(x)
ax.set_xticklabels(gas_labels)
ax.set_xlabel('Gas Consumer Nodes')
ax.set_ylabel('Percentage of Unserved Gas')
ax.set_title('Percentage of Unserved Gas for Gas Demand Nodes GCF')
# ax.set_title('Percentage of Unserved Gas for Gas Demand Nodes UCF')

plt.tight_layout()

plt.savefig('Box and Whisker of Percentage of Unserved Gas GCF', dpi=500)
plt.show()


# %% Box and whisker plot for gas node percentage UCF

fig, ax = plt.subplots()  # figsize=(8,8))
x = np.arange(len(gas_labels))  # the label locations
ax.boxplot(gas_p2_lst)

# ax.set_xticks(x)
ax.set_xticklabels(gas_labels)
ax.set_xlabel('Gas Consumer Nodes')
ax.set_ylabel('Percentage of Unserved Gas')
ax.set_title('Percentage of Unserved Gas for Gas Demand Nodes UCF')

plt.tight_layout()

plt.savefig('Box and Whisker of Percentage of Unserved Gas UCF', dpi=500)
plt.show()

# %% Box ans whisker plot for power node GCF

fig, ax = plt.subplots()  # figsize=(8,8))
x = np.arange(len(power_labels))  # the label locations
ax.boxplot(power_a1_lst)


# ax.set_xticks(x)
ax.set_xticklabels(power_labels)
ax.set_xlabel('Electricity Consumer Nodes')
# ax.set_ylim((0, 95000))
ax.set_ylabel('Unserved Electricity in MWh')
# ax.set_title('Nodal Unserved Electricity for Demand Nodes UCF')
ax.set_title('Nodal Unserved Electricity for Demand Nodes GCF')

plt.tight_layout()

plt.savefig('Box and Whisker Power GCF', dpi=500)
plt.show()

# %% Box ans whisker plot for power node GCF

fig, ax = plt.subplots()  # figsize=(8,8))
x = np.arange(len(power_labels))  # the label locations
ax.boxplot(power_p1_lst)


# ax.set_xticks(x)
ax.set_xticklabels(power_labels)
ax.set_xlabel('Electricity Consumer Nodes')
# ax.set_ylim((0, 95000))
ax.set_ylabel('Unserved Electricity in MWh')
ax.set_title('Nodal Unserved Electricity for Demand Nodes GCF')

plt.tight_layout()

plt.savefig('Box and Whisker of Percentage Power GCF',dpi=500)
plt.show()

# %% Box ans whisker plot for power node UCF

fig, ax = plt.subplots()  # figsize=(8,8))
x = np.arange(len(power_labels))  # the label locations
ax.boxplot(power_a2_lst)


# ax.set_xticks(x)
ax.set_xticklabels(power_labels)
ax.set_xlabel('Electricity Consumer Nodes')
# ax.set_ylim((0, 95000))
ax.set_ylabel('Unserved Electricity in MWh')
ax.set_title('Nodal Unserved Electricity for Demand Nodes UCF')

plt.tight_layout()

plt.savefig('Box and Whisker Power UCF', dpi=500)
plt.show()

# %% Box ans whisker plot for power node Percentage UCF

fig, ax = plt.subplots()  # figsize=(8,8))
x = np.arange(len(power_labels))  # the label locations
ax.boxplot(power_p2_lst)

# ax.set_xticks(x)
ax.set_xticklabels(power_labels)
ax.set_xlabel('Electricity Consumer Nodes')
ax.set_ylim((0, 1.1))
ax.set_title('Percentage of Nodal Unserved Electricity for Demand Nodes UCF')

plt.tight_layout()

plt.savefig('Box and Whisker of Percentage of Unserved Power UCF',dpi=500)
plt.show()

# %% Calculate the number of nodes with unserved energy across scenarios
num_gnode1 = []
num_gnode2 = []

for idx, row in gas_df1.iterrows():
    ct = 0 # count the value of non-zeros of unserved gas in the row
    for value in row:
        if value > 10: ct+=1
    num_gnode1.append(ct)

for idx, row in gas_df2.iterrows():
    ct=0
    for value in row:
        if value>10: ct+=1
    num_gnode2.append(ct)

fig,ax = plt.subplots()
ax.plot(np.sort(num_gnode1),label='GCF')
ax.plot(np.sort(num_gnode2),label='UCF')

ax.set_xlabel('Contingency scenarios')
ax.set_ylabel('Number of consumer nodes with unserved gas')
ax.legend(loc='upper left')
ax.set_title('Number of Gas Consumer Nodes with Unserved Gas')

plt.savefig('Number of Consumer Nodes with Unserved Gas',dpi=500)
plt.show()


# %% Calculate the number of nodes with unserved energy across scenarios
num_pnode1 = []
num_pnode2 = []

for idx, row in power_df1.iterrows():
    ct = 0 # count the value of non-zeros of unserved gas in the row
    for value in row:
        if value > 10: ct+=1
    num_pnode1.append(ct)

for idx, row in power_df2.iterrows():
    ct=0
    for value in row:
        if value>10: ct+=1
    num_pnode2.append(ct)

fig,ax = plt.subplots()
ax.plot(np.sort(num_pnode1),label='GCF')
ax.plot(np.sort(num_pnode2),label='UCF')

ax.set_xlabel('Contingency scenarios')
ax.set_ylabel('Number of consumer nodes with unserved electricity')
ax.legend(loc='upper left')
ax.set_title('Number of Gas Consumer Nodes with Unserved Electricity')

plt.savefig('Number of Consumer Nodes with Unserved Electricity',dpi=500)
plt.show()

# %% Calculate the gini coefficient and attach it to the dataframe, then print the dataframe into csv file
i=0
csv_name_list = ['gas_df1','gas_df2','power_df1','power_df2']

for df in [gas_df1,gas_df2,power_df1,power_df2]:
    gini_list = []

    for idx, row in df.iterrows():
        gini_value = gini(np.array(row))
        gini_list.append(gini_value)

    df['gini'] = gini_list # append the gini coefficients list as the last row of the dataframe

    df.to_csv('%s.csv' %csv_name_list[i])
    i+=1


