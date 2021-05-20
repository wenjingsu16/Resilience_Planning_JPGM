# This code compares the impact of geographically correlated failures and random failures
# The 100 contingency runs uses the same failures as the expansion runs

# %% Import pkgs
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick


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
gcfn_df = pd.read_csv('df2_gcf100.csv',index_col=0)
rfn_df = pd.read_csv('df2_rf100.csv',index_col=0)

# %% Compare total cost for geographically correlated failures and random failures
# 1 in the array name will represent gcf while 2 represents

x = np.arange(0, 100, 1)

total_cost_a1 = np.sort(gcf_df.loc[:, 'System cost'].values)
total_cost_a2 = np.sort(rf_df.loc[:, 'System cost'].values)

fig, ax = plt.subplots()
ax.plot(x, total_cost_a1, label='Geographically Correlated Failures')
ax.plot(x, total_cost_a2, label='Uncorrelated Failures')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('System Total Cost in Dollar')
ax.set_title("Comparison of Total Cost")
ax.legend()
# plt.savefig('Comparison of Total Cost', dpi=600)
plt.show()

# %% Compare total unserved energy for geographically correlated failures and random failures

total_use_a1 = np.sort(gcf_df.loc[:, 'Unserved energy'].values)
total_use_a2 = np.sort(rf_df.loc[:, 'Unserved energy'].values)

fig, ax = plt.subplots()
ax.plot(x, total_use_a1, label='Geographically Correlated Failures')
ax.plot(x, total_use_a2, label='Uncorrelated Failures')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Total Unserved Energy in MWh')
ax.set_title("Comparison of Total Unserved Energy")
ax.legend()
plt.tight_layout()  # this make sure that the figure was captured in full when saved as a figure
plt.savefig('Comparison of Total Unserved Energy', dpi=500)  # ,bbox_inches='tight')
plt.show()  # show command should be latter than savefig


# %% Compare total unserved gas (usg)

baseQ = 535.8564814814815
heat_value = 2.436e-08

total_usg_a1 = np.sort(gcf_df.loc[:, 'Unserved gas'].values)
total_usg_a2 = np.sort(rf_df.loc[:, 'Unserved gas'].values)

# Turn unserved gas into MWh for a day
total_usg_a1 = [x * 86400 * 2.78 * 10 ** (-10) * baseQ/heat_value for x in total_usg_a1]
total_usg_a2 = [x * 86400 * 2.78 * 10 ** (-10) * baseQ/heat_value for x in total_usg_a2]

fig, ax = plt.subplots()
ax.plot(x, total_usg_a1, 'sandybrown', label='GCF Unserved Gas')
ax.plot(x, total_usg_a2, 'cornflowerblue', label='UCF Unserved Gas')
# ax.plot(x, total_use_a1, 'saddlebrown', label='GCF Unserved Energy')
# ax.plot(x, total_use_a2, 'navy', label='RF Unserved Energy')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Total Unserved Gas in MWh')
ax.set_title("Comparison of Total Unserved Gas")
ax.legend()
plt.tight_layout()  # this make sure that the figure was captured in full when saved as a figure
# plt.savefig('Comparison of Total Unserved Gas',dpi=500)#,bbox_inches='tight')
plt.show()  # show command should be latter than savefig

# %% Compare total unserved electricity/power (usp)

total_usp_a1 = np.sort(gcf_df.loc[:, 'Unserved power'].values)
total_usp_a2 = np.sort(rf_df.loc[:, 'Unserved power'].values)

# Turn unserved electricity into MWh for a day
total_usp_a1 = [x * 2400 for x in total_usp_a1]
total_usp_a2 = [x * 2400 for x in total_usp_a2]

fig, ax = plt.subplots()
ax.plot(x, total_usp_a1, 'navy', label='GCF Unserved Electricity')
ax.plot(x, total_usp_a2, 'royalblue', label='UCF Unserved Electricity')
# ax.plot(x, total_use_a1, 'saddlebrown', label='GCF Unserved Energy')
# ax.plot(x, total_use_a2, 'navy', label='UCF Unserved Energy')
ax.plot(x,total_usg_a1,'darkred',label = 'GCF Unserved Gas')
ax.plot(x,total_usg_a2,'tomato',label='UCF Unserved Gas')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Total Unserved Electricity in MWh')
ax.set_title("Comparison of Total Unserved Electricity and Gas")
ax.legend()
plt.tight_layout()  # this make sure that the figure was captured in full when saved as a figure
plt.savefig('Comparison of Total Unserved Electricity and Gas ', dpi=500)  # ,bbox_inches='tight')
plt.show()  # show command should be latter than savefig


# %% Compare total unserved electricity/power (usp)

total_usp_a1 = np.sort(gcf_df.loc[:, 'Unserved power'].values)
total_usp_a2 = np.sort(rf_df.loc[:, 'Unserved power'].values)

# Turn unserved electricity into MWh for a day
total_usp_a1 = [x * 2400 for x in total_usp_a1]
total_usp_a2 = [x * 2400 for x in total_usp_a2]

fig, ax = plt.subplots()
ax.plot(x, total_usp_a1, 'navy', label='GCF Unserved Electricity')
ax.plot(x, total_usp_a2, 'royalblue', label='UCF Unserved Electricity')
ax.plot(x, total_use_a1, 'darkgreen', label='GCF Unserved Energy')
ax.plot(x, total_use_a2, 'olivedrab', label='UCF Unserved Energy')
ax.plot(x,total_usg_a1,'darkred',label = 'GCF Unserved Gas')
ax.plot(x,total_usg_a2,'tomato',label='UCF Unserved Gas')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Total Unserved Electricity in MWh')
ax.set_title("Comparison of Total Unserved Energy & Electricity & Gas")
ax.legend()
plt.tight_layout()  # this make sure that the figure was captured in full when saved as a figure
# plt.savefig('Comparison of Total Unserved Electricity & Energy & Gas', dpi=500)  # ,bbox_inches='tight')
plt.show()  # show command should be latter than savefig

#%% Verify the sum of unserved electricity and gas equals the unserved energy

# Create the list of unserved energy by summing up unserved electricity with unserved gas
use_list1 = []
use_list2 = []

for idx in x:
    use_list1.append(gcf_df.loc[idx, 'Unserved power']*2400 + gcf_df.loc[idx, 'Unserved gas']*528359.7701149426)
    use_list2.append(rf_df.loc[idx,'Unserved power'] *2400 + rf_df.loc[idx,'Unserved gas']*528359.7701149426)

use_list1a = np.sort(np.array(use_list1))
use_list2a = np.sort(np.array(use_list2))

fig, ax = plt.subplots()
ax.plot(x, use_list1a, 'sandybrown', label='GCF Calculated USE')
ax.plot(x, use_list2a, 'cornflowerblue', label='UCF Calculated USE')
ax.plot(x, total_use_a1, 'saddlebrown', label='GCF Extracted USE')
ax.plot(x, total_use_a2, 'navy', label='UCF Extracted USE')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Total Unserved Energy in MWh')
ax.set_title("Verifying Total Unserved Energy")
ax.legend()
plt.tight_layout()  # this make sure that the figure was captured in full when saved as a figure
# plt.savefig('Comparison of Total Unserved Gas',dpi=500)#,bbox_inches='tight')
plt.show()  # show command should be latter than savefig

#It is verified that the total unserved energy calculated by two ways are the same

#%% Stack plot of unserved energy with unserved electricity and unserved gas for GCF

fig, ax = plt.subplots()

# create an array of total unserved energy that is not sorted
use_a1 = gcf_df.loc[:, 'Unserved energy'].values

# create arrays of unserved power and unserved gas that are not sorted based on values
y1 = gcf_df.loc[:, 'Unserved power'].values * 2400
y2 = gcf_df.loc[:, 'Unserved gas'].values * 528359.7701149426

# Sort the unserved electricity and unserved gas based on the sort order of unserved energy array
y1 = [y1[i] for i in np.argsort(use_a1)]
y2 = [y2[i] for i in np.argsort(use_a1)]

# draw the stack plot in matplotlib
ax.stackplot(x,y1,y2,labels=['Unserved electricity','Unserved gas'],colors=['steelblue','darksalmon'])
ax.set_title('Stackplot of Unserved Energy for Geographically Correlated Failures')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Total Unserved Energy in MWh')
ax.legend(loc='upper left')

# Another say of drawing stackplot
# y = np.sort(np.vstack([y1, y2]))
# ax.stackplot(x, y,labels=['Unserved power','Unserved gas'])

# Plot the total unserved energy line in the figure to verify
# ax.plot(x,total_use_a1,label='USE')

plt.tight_layout()
# plt.savefig('Stackplot of Unserved Energy for Geographically Correlated Failures.png',dpi=500,bbox_inches='tight')
plt.show()


#%% Stack plot of unserved energy with unserved electricity and unserved gas for GCF

fig, ax = plt.subplots()

# create an array of total unserved energy that is not sorted
use_a2 = rf_df.loc[:, 'Unserved energy'].values

# create arrays of unserved power and unserved gas that are not sorted based on values
y3 = rf_df.loc[:, 'Unserved power'].values * 2400
y4 = rf_df.loc[:, 'Unserved gas'].values * 528359.7701149426

# Sort the unserved electricity and unserved gas based on the sort order of unserved energy array
y3 = [y3[i] for i in np.argsort(use_a2)]
y4 = [y4[i] for i in np.argsort(use_a2)]

# draw the stack plot in matplotlib
ax.stackplot(x,y3,y4,labels=['Unserved electricity','Unserved gas'],colors=['steelblue','darksalmon'])
ax.set_title('Stackplot of Unserved Energy for Uncorrelated Failures')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Total Unserved Energy in MWh')
ax.legend(loc='upper left')

# Another say of drawing stackplot
# y = np.sort(np.vstack([y1, y2]))
# ax.stackplot(x, y,labels=['Unserved power','Unserved gas'])

# Plot the total unserved energy line in the figure to verify
# ax.plot(x,total_use_a2,label='USE')

plt.tight_layout()
# plt.savefig('Stackplot of Unserved Energy for Uncorrelated Failures.png',dpi=500,bbox_inches='tight')
plt.show()

#%% Scatter plot of unserved electricity and unserved gas for GCF and RF

# create arrays of unserved power and unserved gas that are not sorted based on values for GCF
y1 = gcf_df.loc[:, 'Unserved power'].values * 2400
y2 = gcf_df.loc[:, 'Unserved gas'].values * 528359.7701149426

# create arrays of unserved power and unserved gas that are not sorted based on values for RF
y3 = rf_df.loc[:, 'Unserved power'].values * 2400
y4 = rf_df.loc[:, 'Unserved gas'].values * 528359.7701149426

fig,(ax1,ax2) = plt.subplots(2,figsize=(8,6),sharex=True,sharey=True)
fig.suptitle('Scatter Plot of Unserved Electricity and Unserved Gas',fontsize=13)

ax1.scatter(y2,y1,s=16,marker='D',color='firebrick',alpha=0.5)
ax2.scatter(y4,y3,s=16,marker='D',color='steelblue',alpha=0.5)

ax1.set_title('Geographically Correlated Failures',fontsize=12)
ax2.set_title('Uncorrelated Failures',fontsize=12)

ax2.set_ylabel('Unserved Electricity in MWh',fontsize=10)
ax1.set_ylabel('Unserved Electricity in MWh',fontsize=10)
ax2.set_xlabel('Unserved Gas in MWh',fontsize=10)

plt.tight_layout()
plt.savefig('Comparison of scatter plot of unserved electricity and unserved gas',dpi=500)
plt.show()


# %% Compare the operatin cost of power system and gas system

# operation cost of power system and gas system for GCF
opcost_p1 = np.sort(gcf_df.loc[:, 'Total power cost'].values)
opcost_g1 = np.sort(gcf_df.loc[:, 'Total gas cost'].values)

# operation cost of power system and gas system for RF
opcost_p2 = np.sort(rf_df.loc[:,'Total power cost'].values)
opcost_g2 = np.sort(rf_df.loc[:,'Total gas cost'].values)

fig, ax = plt.subplots()
ax.plot(x, opcost_p1, 'rosybrown',label='Power operation cost GCF')
ax.plot(x, opcost_g1, 'firebrick',label='Gas operation cost GCF')
ax.plot(x,opcost_p2,'steelblue',label='Power operation cost UCF')
ax.plot(x,opcost_g2,'royalblue',label='Gas operation cost UCF')
ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Operation Cost in Dollar')
ax.set_title("Comparison of Power and Gas System Operation Cost")
ax.legend()
plt.tight_layout()
plt.savefig('Comparison of Power and Gas System Operation Cost', dpi=600)
plt.show()


# %% Compare the total electricity demand served and average electricity price (scatter plot) of GCF and RF

generation_a1 = gcf_df.loc[:,'Total power generation'].values
price_a1 = gcf_df.loc[:,'Electricity price'].values

generation_a2 = rf_df.loc[:,'Total power generation'].values
price_a2 = rf_df.loc[:,'Electricity price'].values


fig,(ax1,ax2) = plt.subplots(2,figsize=(8,8)) #,sharex=True) # Share the same x axis
fig.suptitle('Average Electricity Cost with Total Demand Served')
# ax1 = plt.subplot(211) # Another way to have subplot
ax1.scatter(generation_a1,price_a1,s=50,c='firebrick',marker='+',alpha=0.4)
ax1.set_xlabel('Electricity Demand Served in MWh')
ax1.set_ylabel('Electricity Cost $/MWh')
ax1.set_title('GCF')

ax2.scatter(generation_a2,price_a2,s=50,c='steelblue',marker='+',alpha=0.4)
ax2.set_xlabel('Electricity Demand Served in MWh')
ax2.set_ylabel('Electricity Cost $/MWh')
ax2.set_title('UCF')

fig.tight_layout()
# plt.savefig('Comparison of Electricity Cost with Electricity Demand Served',dpi=500,bbox_inches='tight')
plt.show()


#%% The stackbar plot of stackplot of nodal power generation for GCF

# Convert the nodal electricity generation data from the dataframe into arrays for drawing stackplot
node_series1 = gcf_df.loc[:,'Nodal power generation']
node_series1 = [node_series1[i] for i in np.flip(np.argsort(generation_a1))]
node_list1 = [sum(eval(node_series1[x]).values()) for x in range(100)]
# plt.plot(node_list1)
# plt.show()

nodes_arrays1 = np.zeros((5,100))
# Nodal generation arrays
for k in range(5):
    for idx in range(100):
        nodes_arrays1[k][idx] = eval(node_series1[idx])[k+1]


# ax2 = plt.subplot(212)
fig,ax = plt.subplots(figsize=(10,7))
# ax.rcParams['font.size'] = '14'
wid = 0.8
ax.bar(x,nodes_arrays1[0],width=wid,color='cornflowerblue',label='NG1')
ax.bar(x,nodes_arrays1[1],width=wid,bottom=nodes_arrays1[0],color='lightsalmon',label='G2')
ax.bar(x,nodes_arrays1[2],width=wid,bottom=np.array([nodes_arrays1[0,x]+nodes_arrays1[1,x] for x in range(100)]),color='seagreen',label='G3')
ax.bar(x,nodes_arrays1[3],width=wid,bottom=np.array([nodes_arrays1[0,x]+nodes_arrays1[1,x]+nodes_arrays1[2,x] for x in range(100)]),color='tomato',label='NG4')
ax.bar(x,nodes_arrays1[4],width=wid,bottom=np.array([nodes_arrays1[0,x]+nodes_arrays1[1,x]+nodes_arrays1[2,x]+nodes_arrays1[3,x] for x in range(100)]),color='darkorchid',label='NG5')

ax.set_xlabel('Contingency Scenarios',fontsize=14)
ax.set_ylabel('Electricity Demand Served in MWh',fontsize=14)
ax.set_title('Electricity Generation by Nodes for GCF',fontsize=16)
ax.set_ylim([0,270])

ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.2)
plt.tight_layout()

plt.savefig('Electricity Generation by Nodes for GCF',dpi=600)
plt.show()


# fig, ax = plt.subplots()
# ax.stackplot(x, nodes_arrays[0,:], nodes_arrays[1,:],nodes_arrays[2,:],nodes_arrays[3,:],nodes_arrays[4,:])
# plt.show()

#%% The stackbar plot of stackplot of nodal power generation for UCF

# Convert the nodal electricity generation data from the dataframe into arrays for drawing stackplot
node_series2 = rf_df.loc[:,'Nodal power generation']
node_series2 = [node_series2[i] for i in np.flip(np.argsort(generation_a2))]
node_list2 = [sum(eval(node_series2[x]).values()) for x in range(100)]
plt.plot(node_list2)
plt.show()

nodes_arrays2 = np.zeros((5,100))
# Nodal generation arrays
for k in range(5):
    for idx in range(100):
        nodes_arrays2[k][idx] = eval(node_series2[idx])[k+1]


# ax2 = plt.subplot(212)
fig,ax = plt.subplots(figsize=(10,7))
wid = 0.8
ax.bar(x,nodes_arrays2[0],width=wid,color='cornflowerblue',label='NG1')
ax.bar(x,nodes_arrays2[1],width=wid,bottom=nodes_arrays2[0],color='lightsalmon',label='G2')
ax.bar(x,nodes_arrays2[2],width=wid,bottom=np.array([nodes_arrays2[0,x]+nodes_arrays2[1,x] for x in range(100)]),color='seagreen',label='G3')
ax.bar(x,nodes_arrays2[3],width=wid,bottom=np.array([nodes_arrays2[0,x]+nodes_arrays2[1,x]+nodes_arrays2[2,x] for x in range(100)]),color='tomato',label='NG4')
ax.bar(x,nodes_arrays2[4],width=wid,bottom=np.array([nodes_arrays2[0,x]+nodes_arrays2[1,x]+nodes_arrays2[2,x]+nodes_arrays2[3,x] for x in range(100)]),color='darkorchid',label='NG5')

ax.set_xlabel('Contingency Scenarios',fontsize=14)
ax.set_ylabel('Electricity Demand Served in MWh',fontsize=14)
ax.set_title('Electricity Generation by Nodes for UCF',fontsize=16)
ax.set_ylim([0,270])

ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.2)
plt.tight_layout()

plt.savefig('Electricity Generation by Nodes for UCF',dpi=600)
plt.show()

#%%  For nodal data, first process the data into uniform units
gas_df1 = gcfn_df.iloc[:,0:10]
power_df1 = gcfn_df.iloc[:,[12,13,14,15,16,19,20,21,22,23,24]]

gas_df2 = rfn_df.iloc[:,0:10]
power_df2 = rfn_df.iloc[:,[12,13,14,15,16,19,20,21,22,23,24]]

# Convert the dataframe into MWh of Unserved energy for a day
gas_df1 = gas_df1 * 528359.7701149426
gas_df2 = gas_df2 * 528359.7701149426

power_df1 = power_df1 * 2400
power_df2 = power_df2 *2400

# Convert the nodal unserved energy dataframe into numpy array for data processing and visualization

gas_a1 = gas_df1.to_numpy()
power_a1 = power_df1.to_numpy()

gas_a2 = gas_df2.to_numpy()
power_a2 = power_df2.to_numpy()

# Replace all the values of unserved energy less than 10**-3 with 0
gas_a1 = np.where(gas_a1 < 10**-3, 0, gas_a1)
power_a1 = np.where(power_a1 < 10**-3, 0, power_a1)
gas_a2 = np.where(gas_a2 < 10**-3, 0, gas_a2)
power_a2 = np.where(power_a2 < 10**-3, 0, power_a2)

#%% Gini coefficients for 100 runs for GCF and RF

# create list to document gini coefficients for rows in the unserved energy matrix
gini_gas1 = []
gini_gas2 = []
gini_power1 = []
gini_power2 = []

x = np.array(range(100))

# Loop through 100 events to calculate the gini coefficients for gas nodes or power nodes
for row in gas_a1:
    gini_gas1.append(gini(row))
gini_gas_a1 = np.sort(np.array(gini_gas1))

for row in gas_a2:
    gini_gas2.append(gini(row))
gini_gas_a2 = np.sort(np.array(gini_gas2))

for row in power_a1:
    gini_power1.append(gini(row))
gini_power_a1 = np.sort(np.array(gini_power1))

for row in power_a2:
    gini_power2.append(gini(row))
gini_power_a2 = np.sort(np.array(gini_power2))

# Calulate the gini coefficients for all energy nodes for GCF and RF

gini_gcf = []
gini_rf = []

for idx in range(100):
    gini_gcf.append(gini(np.concatenate((gas_a1[idx],power_a1[idx] ), axis=0)))
    gini_rf.append(gini(np.concatenate((gas_a2[idx],power_a2[idx]),axis=0)))

gini_gcf_a = np.sort(np.array(gini_gcf))
gini_rf_a = np.sort(np.array(gini_rf))

#%% Comparison of Gini coefficients for gas nodes and power nodes for GCF and RF

# fig,(ax1,ax2) = plt.subplots(2) # Should i seperate the gini for power system and gas system
fig,ax = plt.subplots()

x = np.array(range(100))

ax.plot(x,gini_gas_a1,'*-',color='indianred',label='Gini coefficient of gas nodes for GCF')
ax.plot(x,gini_gas_a2,'*-',color='royalblue',label='Gini coefficient of gas nodes for UCF')
ax.plot(x,gini_power_a1,'1--',color='peru',label='Gini coefficient of power nodes for GCF')
ax.plot(x,gini_power_a2,'1--',color='mediumpurple',label='Gini coefficient of power nodes for UCF')

ax.set_xlabel('Contingency Scenarios',fontsize=11)
ax.set_ylabel('Gini Coefficients',fontsize=11)
ax.set_title('Gini Coefficients of Gas Nodes and Power Nodes for GCF and UCF',fontsize=12)
ax.set_ylim((-0.07,1))
# ax.set_ylim((0.75,0.95))
# ax.set_xlim((30,100))

ax.legend(loc='upper left',fontsize='xx-small')
# ax.legend(bbox_to_anchor=(1.03, 1), loc='below')
# ax.legend(loc='upper center',bbox_to_anchor=(0.2,1.05))
plt.tight_layout()

plt.savefig('Gini Comparison',dpi=600)
# plt.savefig('Gini Comparison ZOOM IN',dpi=600)


plt.show()

#%% Comparison of gini coefficients of all energy consumer nodes
fig, ax = plt.subplots()

x = np.array(range(100))
ax.plot(x,gini_gcf_a,'*-',color='firebrick',label='Geographically Correlated Failures')
ax.plot(x,gini_rf_a,'*-',color='steelblue',label='Random Failures')

ax.set_xlabel('Contingency Scenarios')
ax.set_ylabel('Gini Coefficient')
ax.set_ylim((0,1.1))
ax.set_title('Gini Coefficients of All Nodes for GCF and UCF')
ax.legend()
# plt.tight_layout()

plt.savefig('Gini Coefficients of All Nodes for GCF and UCF',dpi=500)
plt.show()


#%% Box and whisker plots for nodal unserved energy
fig, ax = plt.subplots()
ax.boxplot(power_a1)
plt.show()

# plt.savefig('Box and whisker power gcf',dpi=400)

#%% Attempt to plot the scatter plot with pie chart as marker for GCF
fig,ax = plt.subplots()
count = 0

for idx in range(100):

    if gcf_df.loc[idx,'Unserved energy'] > 0:

        r1 = gcf_df.loc[idx,'Unserved gas']*528359.7701149426/gcf_df.loc[idx,'Unserved energy']

        sizes = 150

        # calculate the points of the first pie marker
        # these are just the origin (0,0) +
        # some points on a circle cos,sin
        x = [0] + np.cos(np.linspace(0, 2 * np.pi * r1, 10)).tolist()
        y = [0] + np.sin(np.linspace(0, 2 * np.pi * r1, 10)).tolist()
        xy1 = np.column_stack([x, y])

        x = [0] + np.cos(np.linspace(2 * np.pi * r1, 2 * np.pi, 10)).tolist()
        y = [0] + np.sin(np.linspace(2 * np.pi * r1, 2 * np.pi, 10)).tolist()
        xy2 = np.column_stack([x, y])

        locx = gcf_df.loc[idx,'Unserved energy'] # X axis is the total unserved energy in MWh
        locy = gini_gcf_a[idx] # Y axis is the gini coefficient for this scenario

        ax.scatter(locx, locy, marker=xy1, s=sizes,
                   color='salmon',alpha=0.7) #Unserved gas
        ax.scatter(locx, locy, marker=xy2, s=sizes,
                   color='royalblue',alpha=0.7) #Unserved electricity
        count += 1

ax.set_xlabel('Total Unserved Energy in MWh (pink:gas;blue:electricity)')
ax.set_ylabel('Gini Coefficient')
ax.set_ylim((-0.1,1.05))
ax.set_title('Gini Coefficient with Total Unserved Energy GCF')

plt.savefig('Gini Coefficients with the Total Unserved Energy GCF with Pie Marker',dpi=600)
plt.show()
print(count)

#%% Attempt to plot the scatter plot with pie chart as marker for RF
fig,ax = plt.subplots()
count = 0

for idx in range(100):

    if rf_df.loc[idx,'Unserved energy'] > 0:

        r1 = rf_df.loc[idx,'Unserved gas']*528359.7701149426/rf_df.loc[idx,'Unserved energy']

        sizes = 150

        # calculate the points of the first pie marker
        # these are just the origin (0,0) +
        # some points on a circle cos,sin
        x = [0] + np.cos(np.linspace(0, 2 * np.pi * r1, 10)).tolist()
        y = [0] + np.sin(np.linspace(0, 2 * np.pi * r1, 10)).tolist()
        xy1 = np.column_stack([x, y])

        x = [0] + np.cos(np.linspace(2 * np.pi * r1, 2 * np.pi, 10)).tolist()
        y = [0] + np.sin(np.linspace(2 * np.pi * r1, 2 * np.pi, 10)).tolist()
        xy2 = np.column_stack([x, y])

        locx = rf_df.loc[idx,'Unserved energy'] # X axis is the total unserved energy in MWh
        locy = gini_rf_a[idx] # Y axis is the gini coefficient for this scenario

        ax.scatter(locx, locy, marker=xy1, s=sizes,
                   color='salmon',alpha=0.7) #Unserved gas
        ax.scatter(locx, locy, marker=xy2, s=sizes,
                   color='royalblue',alpha=0.7) #Unserved electricity
        count += 1

ax.set_xlabel('Total Unserved Energy in MWh (pink:gas;blue:electricity)')
ax.set_ylabel('Gini Coefficient')
ax.set_ylim((-0.1,1.05))
ax.set_title('Gini Coefficient with Total Unserved Energy UCF')

plt.savefig('Gini Coefficients with the Total Unserved Energy UCF with Pie Marker',dpi=600)
plt.show()
print(count)

#%% Scatter plot of gini coefficients vs unserved gas /power

fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(8,14))
fig.suptitle('Gini Coefficients with Unserved Energy, Gas and Electricity',fontsize=13)

ax1.scatter(total_use_a1,gini_gcf_a,color='indianred',marker='^',alpha = 0.7,label='GCF')
ax1.scatter(total_use_a2,gini_rf_a,color='steelblue',marker='*',alpha =0.7,label='UCF')
ax1.set_xlabel('Unserved Energy in MWh',fontsize=9)
ax1.set_ylabel('Gini Coefficient for Energy Nodes',fontsize=9)
ax1.set_ylim((-0.1,1.05))
ax1.legend(loc=4)
ax1.set_title('Energy',fontsize=11)

ax2.scatter(total_usg_a1,gini_gas_a1,color='indianred',marker='^',alpha = 0.7,label='GCF')
ax2.scatter(total_usg_a2,gini_gas_a2,color='steelblue',marker='*',alpha=0.7,label='UCF')
ax2.set_xlabel('Unserved Gas in MWh',fontsize=9)
ax2.set_ylabel('Gini Coefficient for Gas Nodes',fontsize=9)
ax2.set_ylim((-0.1,1.05))
ax2.legend(loc=4)
ax2.set_title('Gas',fontsize=11)

ax3.scatter(total_usp_a1,gini_power_a1,color='indianred',marker='^',alpha = 0.7,label='GCF')
ax3.scatter(total_usp_a2,gini_power_a2,color='steelblue',marker='*',alpha=0.7,label='UCF')
ax3.set_xlabel('Unserved Electricity in MWh',fontsize=9)
ax3.set_ylabel('Gini Coefficient for Electricity Nodes',fontsize=9)
ax3.set_ylim((-0.1,1.05))
ax3.legend(loc=4)
ax3.set_title('Electricity',fontsize=11)

plt.tight_layout()
# plt.savefig('Gini Coefficients with Unserved Energy, Gas and Electricity ZOOM IN',dpi=600)
plt.savefig('Gini Coefficients with Unserved Energy, Gas and Electricity',dpi=600)
plt.show()

#%% Scatter plot of gini coefficients vs unserved gas /power ZOOM IN

fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(8,14))
fig.suptitle('Gini Coefficients with Unserved Energy, Gas and Electricity',fontsize=13)

ax1.scatter(total_use_a1,gini_gcf_a,color='indianred',marker='^',alpha = 0.7,label='GCF')
ax1.scatter(total_use_a2,gini_rf_a,color='steelblue',marker='*',alpha =0.7,label='UCF')
ax1.set_xlabel('Unserved Energy in MWh',fontsize=9)
ax1.set_ylabel('Gini Coefficient for Energy Nodes',fontsize=9)
ax1.set_ylim((0.75,1.0))
ax1.legend(loc=4)
ax1.set_title('Energy',fontsize=11)

ax2.scatter(total_usg_a1,gini_gas_a1,color='indianred',marker='^',alpha = 0.7,label='GCF')
ax2.scatter(total_usg_a2,gini_gas_a2,color='steelblue',marker='*',alpha=0.7,label='UCF')
ax2.set_xlabel('Unserved Gas in MWh',fontsize=9)
ax2.set_ylabel('Gini Coefficient for Gas Nodes',fontsize=9)
ax2.set_ylim((0.75,1.0))
ax2.legend(loc=4)
ax2.set_title('Gas',fontsize=11)

ax3.scatter(total_usp_a1,gini_power_a1,color='indianred',marker='^',alpha = 0.7,label='GCF')
ax3.scatter(total_usp_a2,gini_power_a2,color='steelblue',marker='*',alpha=0.7,label='UCF')
ax3.set_xlabel('Unserved Electricity in MWh',fontsize=9)
ax3.set_ylabel('Gini Coefficient for Electricity Nodes',fontsize=9)
ax3.set_ylim((0.75,1.0))
ax3.legend(loc=4)
ax3.set_title('Electricity',fontsize=11)

plt.tight_layout()
plt.savefig('Gini Coefficients with Unserved Energy, Gas and Electricity ZOOM IN',dpi=600)
# plt.savefig('Gini Coefficients with Unserved Energy, Gas and Electricity',dpi=600)
plt.show()

#%% CDF of percentage of unserved gas in total unserved energy for RF and GCF
# THIS FIGURE SEEMS TO BE INCORRECT SOMEHOW

# calculate the percentage of unserved gas in total unserved energy to get an array
gas_pct1 = []
gas_pct2 =[]

for idx in range(100):
    if gcf_df.loc[idx,'Unserved energy'] > 0:
        gas_pct1.append(gcf_df.loc[idx,'Unserved gas']*528359.7701149426/gcf_df.loc[idx,'Unserved energy']*100)
    if rf_df.loc[idx,'Unserved energy']>0:
        gas_pct2.append(rf_df.loc[idx,'Unserved gas']*528359.7701149426/rf_df.loc[idx,'Unserved energy']*100)

fig,ax = plt.subplots()
n1, bins1, patches1 = ax.hist(gas_pct1,len(gas_pct1), density=True, histtype='step',
                           cumulative=True, label='GCF')
n2, bins2, patches2 = ax.hist(gas_pct2,len(gas_pct2), density=True, histtype='step',
                           cumulative=True, label='UCF')
# ax.plot(np.sort(gas_pct1),label='gcf')
# ax.plot(np.sort(gas_pct2),label='ucf')
ax.xaxis.set_major_formatter(mtick.PercentFormatter()) # use ticker package in matplotlib to chang x-axis into pct

ax.set_xlabel('Percentage of Unserved Gas in Total Unserved Energy')
ax.set_ylabel('Likelihood of Occurance')
ax.set_title('CDF of Percentage of Unserved Gas in Total Unserved Energy')

plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig('Comparison of Percentage of Unserved Gas in Total Unserved Energy',dpi=500)
plt.show()

# %% percentage of unserved gas in total unserved energy
# calculate the percentage of unserved gas in total unserved energy to get an array
gas_pct1 = []
gas_pct2 =[]

for idx in range(100):
    if gcf_df.loc[idx,'Unserved energy'] > 0:
        gas_pct1.append(gcf_df.loc[idx,'Unserved gas']*528359.7701149426/gcf_df.loc[idx,'Unserved energy']*100)
    if rf_df.loc[idx,'Unserved energy']>0:
        gas_pct2.append(rf_df.loc[idx,'Unserved gas']*528359.7701149426/rf_df.loc[idx,'Unserved energy']*100)

fig,ax = plt.subplots()
ax.plot(np.sort(gas_pct1),label='GCF')
ax.plot(np.sort(gas_pct2),label='UCF')
# ax.xaxis.set_major_formatter(mtick.PercentFormatter()) # use ticker package in matplotlib to chang x-axis into pct

ax.set_ylabel('Percentage of Unserved Gas in Total Unserved Energy')
ax.set_xlabel('Scenarios with Unserved Energy')
ax.set_title('Percentage of Unserved Gas in Total Unserved Energy')

plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig('Comparison of Percentage of Unserved Gas in Total Unserved Energy',dpi=500)
plt.show()
