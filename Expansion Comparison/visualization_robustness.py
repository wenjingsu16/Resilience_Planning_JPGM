# Feb 23 2021

# This script is to visualize the results from robustness check
# to compare the resilience enhanced by different expansion decisions

# %% import packages

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# %% Read in dataframe

col_name = ['GCF10', 'GCF8', 'GCF5', 'UCF10', 'UCF8', 'UCF5', 'Base']

cost_df1 = pd.read_csv('cost_gcf.csv', index_col=0)
cost_df1.columns = col_name
cost_df2 = pd.read_csv('cost_rf.csv', index_col=0)
cost_df2.columns = col_name

use_df1 = pd.read_csv('usenergy_gcf.csv', index_col=0)
use_df1.columns = col_name
use_df2 = pd.read_csv('usenergy_rf.csv', index_col=0)
use_df2.columns = col_name

# %% Plot the unserved energy for geographically correlated failures

x = np.arange(0, 100, 1)

fig, ax = plt.subplots()
color_lst = ['indianred', 'saddlebrown', 'tan', 'rebeccapurple', 'royalblue', 'slategrey', 'darkgreen']

# Loop through the columns to create lines
for idx in [0, 1, 3, 4, 6]:
    use_a1 = np.sort(use_df1.iloc[:, idx].values)
    ax.plot(x, use_a1, color=color_lst[idx], label=col_name[idx])

ax.set_ylim([-3000, 170000])
ax.set_xlabel('Number of Contingency Scenrios')
ax.set_ylabel('Total Unserved Energy in MWh')
ax.set_title('Total Unserved Energy after Expansion against GCF')
ax.legend()
plt.tight_layout()

plt.savefig('Total Unserved Energy of Expanded Network against GCF', dpi=500)
plt.show()

# %% Plot the difference of unserved energy beween the expanded network and base network for GCF
# essentially the benefits of network expansion

# create the dataframe of reduced unserved energy from a list of list (lol)
gcf_lol = []

for idx, row in use_df1.iterrows():
    temp_list = []  # the list is to store a row of differences that will be appended to gcf list of list
    for x in row.iloc[0:6]:
        temp_list.append(row.iloc[-1] - x)
    gcf_lol.append(temp_list)

diff1_df = pd.DataFrame(gcf_lol, columns=col_name[0:6])
diff1_df.to_csv('diff_gcf.csv')

# %% Plot the difference of unserved energy beween the expanded network and base network for RF
# essentially the benefits of network expansion aginast RF

# create the dataframe of reduced unserved energy from a list of list (lol)
rf_lol = []

for idx, row in use_df2.iterrows():
    temp_list = []  # the list is to store a row of differences that will be appended to gcf list of list
    for x in row.iloc[0:6]:
        temp_list.append(row.iloc[-1] - x)
    rf_lol.append(temp_list)

diff2_df = pd.DataFrame(rf_lol, columns=col_name[0:6])
diff2_df.to_csv('diff_rf.csv')

# %% plot the unserved energy for random failures

x = np.arange(0, 100, 1)

fig, ax = plt.subplots()
color_lst = ['indianred', 'saddlebrown', 'tan', 'rebeccapurple', 'royalblue', 'slategrey', 'darkgreen']

# Loop through the columns to create the CDF lines
for idx in [0, 1, 3, 4, 6]:
    use_a2 = np.sort(use_df2.iloc[:, idx].values)
    ax.plot(x, use_a2, color=color_lst[idx], label=col_name[idx])

ax.set_ylim([-3000, 170000])
ax.set_xlabel('Number of Contingency Scenrios')
ax.set_ylabel('Total Unserved Energy in MWh')
ax.set_title('Total Unserved Energy for Expanded Network for RF')
ax.legend()
plt.tight_layout()

plt.savefig('Unserved Energy after Expansion against UCF', dpi=500)
plt.show()

# %% Plot the reduced energy for gcf

x = np.arange(0, 100, 1)

fig, ax = plt.subplots()
color_lst = ['indianred', 'saddlebrown', 'tan', 'rebeccapurple', 'royalblue', 'slategrey', 'darkgreen']

# for idx in [0, 1, 3, 4]:
for idx in [0,3]:
    diff_a1 = np.sort(diff1_df.iloc[:, idx].values)
    ax.plot(x, diff_a1, color=color_lst[idx], label=col_name[idx])

ax.legend()
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Reduced Unserved Energy in MWh')
ax.set_title('Unserved Energy Reduced by Expansion against GCF')


plt.tight_layout()
plt.savefig('Unserved Energy Reduced by Expansion against GCF',dpi=500)
plt.show()

# %% Plot the box and whisker plot of the reduced energy for GCF

fig, ax = plt.subplots()

ax.boxplot(diff1_df)
ax.set_xticklabels(col_name[0:6])
ax.set_xlabel('Expanded Network Structure')
ax.set_ylabel('Reduced Unserved Energy in MWh')
ax.set_title('Unserved Energy Reduced by Expansion against GCF')

plt.tight_layout()
plt.savefig('Box and Whisker of Unserved Energy Reduced by Expansion against GCF',dpi=500)
plt.show()

# %% Plot the reduced energy for rf

x = np.arange(0,100,1)

fig, ax = plt.subplots()
color_lst = ['indianred', 'saddlebrown', 'tan', 'rebeccapurple', 'royalblue', 'slategrey', 'darkgreen']

# for idx in [0, 1, 3, 4]:
for idx in [0, 3]:
    diff_a2 = np.sort(diff2_df.iloc[:, idx].values)
    ax.plot(x, diff_a2, color=color_lst[idx], label=col_name[idx])

ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Reduced Unserved Energy in MWh')
ax.set_title('Unserved Energy Reduced by Expansion against UCF')
ax.legend()

plt.tight_layout()
plt.savefig('Unserved Energy Reduced by Expansion against UCF',dpi=500)
plt.show()

# %% Plot the box and whisker plot of the reduced energy for UCF

fig, ax = plt.subplots()

ax.boxplot(diff2_df)
ax.set_xticklabels(col_name[0:6])
ax.set_xlabel('Expanded Network Structure')
ax.set_ylabel('Reduced Unserved Energy in MWh')
ax.set_title('Unserved Energy Reduced by Expansion against UCF')

plt.tight_layout()
plt.savefig('Box and Whisker of Unserved Energy Reduced by Expansion against UCF',dpi=500)
plt.show()

# %% plot the same data (unserved energy for expanded networks) as above with box and whisker plot for RF

fig, ax = plt.subplots()
# color_lst = ['indianred','saddlebrown','tan','rebeccapurple','royalblue','slategrey','darkgreen']

# Loop through the columns to create the
# for idx in range(7):
#     use_a2 = np.sort(use_df2.iloc[:,idx].values)
#     ax.boxplot(use_a2)
ax.boxplot(use_df2)

# ax.set_ylim([-3000,175000])
ax.set_xlabel('Expanded Network Structure')
ax.set_xticklabels(col_name)
ax.set_ylabel('Total Unserved Energy in MWh')
ax.set_title('Total Unserved Energy for Expanded Network for UCF')
# ax.legend()
plt.tight_layout()

# plt.savefig('Resilience of Expanded Network against UCF',dpi=500)
plt.show()
