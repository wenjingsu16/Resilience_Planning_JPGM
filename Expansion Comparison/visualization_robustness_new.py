# Feb 23 2021

# This script is to visualize the results from robustness check
# to compare the resilience enhanced by different expansion decisions

# %% import packages

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# %% Read in dataframe

# col_name = ['GCF10', 'GCF8', 'GCF5', 'UCF10', 'UCF8', 'UCF5', 'Base']
col_name = ['GCF10','UCF10']

cost_df1 = pd.read_csv('cost_gcf_check.csv', index_col=0)
cost_df1.columns = col_name
cost_df1[cost_df1<10**(-2)] = 0
cost_df2 = pd.read_csv('cost_rf_check.csv', index_col=0)
cost_df2.columns = col_name
cost_df2[cost_df2<10**(-2)] = 0

col_name = ['GCF10','UCF10','Base']
use_df1 = pd.read_csv('usenergy_gcf_check.csv', index_col=0)
use_df1.columns = col_name
use_df1[use_df1<10**(-3)] = 0
use_df2 = pd.read_csv('usenergy_rf_check.csv', index_col=0)
use_df2.columns = col_name
use_df2[use_df2<10**(-3)] = 0

# %% Plot the unserved energy for geographically correlated failures

x = np.arange(0, 100, 1)

fig, ax = plt.subplots()
# color_lst = ['indianred', 'saddlebrown', 'tan', 'rebeccapurple', 'royalblue', 'slategrey', 'darkgreen']
color_lst = ['indianred','royalblue','darkgreen']

# Loop through the columns to create lines
for idx in [0,1,2]:
    use_a1 = np.sort(use_df1.iloc[:, idx].values)
    ax.plot(x, use_a1, color=color_lst[idx], label=col_name[idx])

# ax.set_ylim([-3000, 170000])
ax.set_xlabel('Number of Contingency Scenrios')
ax.set_ylabel('Total Unserved Energy in MWh')
ax.set_title('Total Unserved Energy after Expansion against GCF')
ax.legend()
plt.tight_layout()

plt.savefig('Total Unserved Energy with Expanded Network against GCF', dpi=500)
plt.show()

# %% Plot the difference of unserved energy beween the expanded network and base network for GCF
# essentially the benefits of network expansion

# create the dataframe of reduced unserved energy from a list of list (lol)
gcf_lol = []
gcf_pct = [] # the list documents the percentage of unserved energy reduced from GCF and UCF in total USE

for idx, row in use_df1.iterrows():
    temp_list = [] # the list is to store a row of differences that will be appended to gcf list of list
    pct_list = []
    if row.iloc[-1] > 10**(-1):
        for x in row.iloc[0:2]:
            value = row[2] - x
            if value < 10**(-2): value = 0
            temp_list.append(value)
            pct_list.append(value/row.iloc[2])
        gcf_lol.append(temp_list)
        gcf_pct.append(pct_list)

    else:continue

diff1_df = pd.DataFrame(gcf_lol, columns=col_name[0:2])
diff1_df.to_csv('diff_gcf.csv')

diffpct1_df = pd.DataFrame(gcf_pct,columns=col_name[0:2])
diffpct1_df.to_csv('diffpct_gcf.csv')

# %% Plot the difference of unserved energy beween the expanded network and base network for RF
# essentially the benefits of network expansion aginast RF

# create the dataframe of reduced unserved energy from a list of list (lol)
rf_lol = []
rf_pct = [] # the list documents the percentage of unserved energy reduced from GCF and UCF in total USE

for idx, row in use_df2.iterrows():
    temp_list = []  # the list is to store a row of differences that will be appended to gcf list of list
    pct_list = []
    if row.iloc[-1] > 10**(-1):
        for x in row.iloc[0:2]:
            value = row[2] - x
            if value < 10**(-2): value = 0
            temp_list.append(row.iloc[-1] - x)
            pct_list.append((row.iloc[-1]-x)/row.iloc[-1])
        rf_lol.append(temp_list)
        rf_pct.append(pct_list)
    else: continue


diff2_df = pd.DataFrame(rf_lol, columns=col_name[0:2])
diff2_df.to_csv('diff_rf.csv')

diffpct2_df = pd.DataFrame(rf_pct,columns=col_name[0:2])
diffpct2_df.to_csv('diffpct_rf.csv')

# %% plot the unserved energy for random failures

x = np.arange(0, 100, 1)

fig, ax = plt.subplots()
# color_lst = ['indianred', 'saddlebrown', 'tan', 'rebeccapurple', 'royalblue', 'slategrey', 'darkgreen']
color_lst = ['indianred','royalblue','darkgreen']

# Loop through the columns to create the CDF lines
for idx in [0, 1,2]:
    use_a2 = np.sort(use_df2.iloc[:, idx].values)
    ax.plot(x, use_a2, color=color_lst[idx], label=col_name[idx])

# ax.set_ylim([-3000, 170000])
ax.set_xlabel('Number of Contingency Scenrios')
ax.set_ylabel('Total Unserved Energy in MWh')
ax.set_title('Total Unserved Energy for Expanded Network for UCF')
ax.legend()
plt.tight_layout()

plt.savefig('Total unserved Energy after Expansion against UCF', dpi=500)
plt.show()

# %%%%%%%%%%%%%%%%%%%
# plot the reduced unserved energy

# %% Plot the reduced energy for gcf

x = np.arange(0,len(gcf_lol), 1)

fig, ax = plt.subplots()
# color_lst = ['indianred', 'saddlebrown', 'tan', 'rebeccapurple', 'royalblue', 'slategrey', 'darkgreen']
color_lst = ['indianred','royalblue','darkgreen']

# for idx in [0, 1, 3, 4]:
for idx in [0,1]:
    diff_a1 = np.sort(diff1_df.iloc[:, idx].values)
    ax.plot(x, diff_a1, color=color_lst[idx], label=col_name[idx])

ax.legend()
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Reduced Unserved Energy in MWh')
ax.set_title('Unserved Energy Reduced by Expansion against GCF')


plt.tight_layout()
plt.savefig('Unserved Energy Reduced by Expansion against GCF',dpi=500)
plt.show()

# %% Plot the box and whisker plot of the reduced energy for GCF conditioning on non-0 unserved energy

fig, ax = plt.subplots()

ax.boxplot(diff1_df)
ax.set_xticklabels(col_name[0:2])
ax.set_xlabel('Expanded Network Structure')
ax.set_ylabel('Reduced Unserved Energy in MWh')
ax.set_title('Unserved Energy Reduced by Expansion against GCF')

plt.tight_layout()
plt.savefig('Box and Whisker of Unserved Energy Reduced by Expansion against GCF',dpi=500)
plt.show()

# %% Plot the box and whisker plot of the percentage reduced energy for GCF conditioning on non-0 unserved energy

fig, ax = plt.subplots()

ax.boxplot(diffpct1_df)
ax.set_xticklabels(col_name[0:2])
ax.set_xlabel('Expanded Network Structure')
ax.set_ylabel('Percentage of Unserved Energy Being Reduced')
ax.set_title('Percntage of Unserved Energy Reduced by Expansion against GCF')

plt.tight_layout()
plt.savefig('Box and Whisker of Percentage of Unserved Energy Reduced by Expansion against GCF',dpi=500)
plt.show()

# %% Plot the box and whisker plot of the percentage reduced energy for UCF conditioning on non-0 unserved energy

fig, ax = plt.subplots()

ax.boxplot(diffpct2_df)
ax.set_xticklabels(col_name[0:2])
ax.set_xlabel('Expanded Network Structure')
ax.set_ylabel('Percentage of Unserved Energy Being Reduced')
ax.set_title('Percntage of Unserved Energy Reduced by Expansion against UCF')

plt.tight_layout()
plt.savefig('Box and Whisker of Percentage of Unserved Energy Reduced by Expansion against UCF',dpi=500)
plt.show()

# %% Plot the difference in reduced energy scenario-wise against GCF conditioning on non-0 unserved energy

diff_diff1 = []
for lst in gcf_lol:
    diff_diff1.append(lst[0]-lst[1])
diff_diff1= np.sort(diff_diff1)

fig, ax = plt.subplots()
ax.plot(diff_diff1)
ax.set_xlabel('Number of Scenarios with Unserved Energy')
ax.set_ylabel('Unserved Energy Difference in MWh')
ax.set_title('Unserved Energy Reduced between Two Model Settings against GCF')
plt.savefig('Difference in Unserved Energy Reduced between Two Model Settings GCF',bbox_inches='tight',dpi=500)
plt.show()



# %% Plot the difference in reduced energy scenario-wise against UCF conditioning on non-0 unserved energy

diff_diff2 = []
for lst in rf_lol:
    diff_diff2.append(lst[1]-lst[0])
diff_diff2 = np.sort(diff_diff2)

fig, ax = plt.subplots()
ax.plot(diff_diff2)
ax.set_xlabel('Number of Scenarios with Unserved Energy')
ax.set_ylabel('Unserved Energy Difference in MWh')
ax.set_title('Unserved Energy Reduced between Two Model Settings against UCF')
plt.savefig('Difference in Unserved Energy Reduced between Two Model Settings UCF',bbox_inches='tight',dpi=500)
plt.show()



# %% Plot the reduced energy for rf conditioning on non-0 unserved energy

x = np.arange(0,len(rf_lol),1)

fig, ax = plt.subplots()
# color_lst = ['indianred', 'saddlebrown', 'tan', 'rebeccapurple', 'royalblue', 'slategrey', 'darkgreen']
color_lst = ['indianred','royalblue','darkgreen']

# for idx in [0, 1, 3, 4]:
for idx in [0,1]:
    diff_a2 = np.sort(diff2_df.iloc[:, idx].values)
    ax.plot(x, diff_a2, color=color_lst[idx], label=col_name[idx])

ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Reduced Unserved Energy in MWh')
ax.set_title('Unserved Energy Reduced by Expansion against UCF')
ax.legend()

plt.tight_layout()
plt.savefig('Unserved Energy Reduced by Expansion against UCF',dpi=500)
plt.show()

# %% Plot the box and whisker plot of the reduced energy for UCF conditioning on non-0 unserved energy

fig, ax = plt.subplots()

ax.boxplot(diff2_df)
ax.set_xticklabels(col_name[0:2])
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
