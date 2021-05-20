#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:30:08 2020

@author: carriesu
"""
# This version of code is deterministic equavilent using scenarios reduced by ALFA
# compressor is not an option for expansion

# For different runs, change the events csv, change weight csv, change code related to status, change the name of the gams file, time frame in the objective function

# %%
from __future__ import division
import numpy as np
from numpy import array
from pyomo.environ import *
import pandas as pd
from csv import reader

model = ConcreteModel()

penalty_value = 10**4

# %% Create the set of scenarios and assign weight
# rs: reduced scenarios by ALFA
# df_rs=pd.read_csv('reduced_scenarios.csv')
# df_rs = pd.read_csv('re_cost5_rf.csv')
df_rs = pd.read_csv('re_cost10_rf.csv')
# df_rs = pd.read_csv('re_cost15_rf.csv')

# The number of set equal to the length of the dataframe
scenario_list = list((range(len(df_rs))))
# scenario_list=list(range(5))
# scenario_list = list(range(10))
model.SCENARIO = Set(initialize=scenario_list)
# model.SCENARIO.pprint()

# import the weight of the scenario and define a parameter of weight


with open('rf_cost_weight10.csv', 'r') as read_obj:
# with open('rf_cost_weight5.csv', 'r') as read_obj:
# with open('rf_cost_weight15.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    weight_list = list(csv_reader)
weight_list = sum(weight_list, [])
weight_list = [float(x) for x in weight_list]
total_weight = sum(weight_list)
weight_list = [x / total_weight for x in weight_list]

# weight_list=[0.2]*5
# weight_list = [0.125] * 8
# weight_list=[0.1]*10
# weight_list = [float(1/len(df_rs))] * len(df_rs)

weight_dict = dict(zip(scenario_list, weight_list))
model.Weight = Param(model.SCENARIO, initialize=weight_dict)
model.Weight.pprint()

# %% Construct the IEEE 14 bus power system model

from busdata import *

newline = np.loadtxt('power_ne.txt')
# f_bus	t_bus	br_r	br_x	br_b	rate_a	rate_b	rate_c	tap	shift	br_status	angmin	angmax	construction_cost
# %% Define set, param and var for nodes
# Sets will be in all capital letters. Param will have first letter as capital letter
# Var will have all lowercase letters
a = bus[:, 0].tolist()
model.NODES = Set(initialize=a)
# model.NODES.pprint()

b = dict(zip(a, bus[:, 2]))
model.Demand = Param(model.NODES, initialize=b)
# model.Demand.pprint()

model.angle = Var(model.NODES, model.SCENARIO)


def uspower_bound(model, node, scenario):
    return (0, model.Demand[node])


model.uspower = Var(model.NODES, model.SCENARIO, domain=NonNegativeReals, bounds=uspower_bound, initialize=0)
# model.uspower.pprint()

# %% Define set, param and var for lines
# model.del_component(model.LINES)
c = list(zip(branch[:, 0], branch[:, 1]))
model.LINES = Set(initialize=c)  # ,within=model.NODES*model.NODES)
# model.LINES.pprint()

d = dict(zip(c, branch[:, 3]))
model.Reactance = Param(model.LINES, initialize=d)

# e=dict(zip(c,branch[:,5])) when rate A is 9900
e = dict(zip(c, rateA))
model.Flowlimit = Param(model.LINES, initialize=e)

# The status parameters for transmission lines which will be used for defining contingency
model.St = Param(model.LINES, model.SCENARIO, default=1, mutable=True)
# model.St.pprint()
# initialize=dict(zip(c,[1 for i in range(20)]))

model.lineflow = Var(model.LINES, model.SCENARIO)


def lineflow_rule1(model, i, j, scenario):
    return -model.St[i, j, scenario] * model.Flowlimit[i, j] <= model.lineflow[i, j, scenario]


model.lineflow1 = Constraint(model.LINES, model.SCENARIO, rule=lineflow_rule1)


# model.lineflow.pprint()

def lineflow_rule2(model, i, j, scenario):
    return model.lineflow[i, j, scenario] <= model.St[i, j, scenario] * model.Flowlimit[i, j]


model.lineflow2 = Constraint(model.LINES, model.SCENARIO, rule=lineflow_rule2)

# %%
# model.del_component(model.NEWLINES)
h = list(zip(newline[:, 0], newline[:, 1]))
model.NEWLINES = Set(initialize=h)  # ,within=model.NODES*model.NODES)
# model.NEWLINES.pprint()

i = dict(zip(h, newline[:, 3]))
model.Reactance_ne = Param(model.NEWLINES, initialize=i)

j = dict(zip(h, newline[:, 5]))
model.Flowlimit_ne = Param(model.NEWLINES, initialize=j)

model.lineflow_ne = Var(model.NEWLINES, model.SCENARIO)

# The binary variable to define whether a line is built or not
model.zt = Var(model.NEWLINES, domain=Binary, initialize=0)
# model.zt.pprint()

k = dict(zip(h, newline[:, 13]))
model.Pricet = Param(model.NEWLINES, initialize=k)


# line flow is within the bound*binary investment decision variable
def linelimit_ne_rule1(model, i, j, scenario):
    return -model.zt[i, j] * model.Flowlimit_ne[i, j] <= model.lineflow_ne[i, j, scenario]


model.linelimit_ne1 = Constraint(model.NEWLINES, model.SCENARIO, rule=linelimit_ne_rule1)


# model.linelimit_ne1.pprint()

def linelimit_ne_rule2(model, i, j, scenario):
    return model.lineflow_ne[i, j, scenario] <= model.zt[i, j] * model.Flowlimit_ne[i, j]


model.linelimit_ne2 = Constraint(model.NEWLINES, model.SCENARIO, rule=linelimit_ne_rule2)
# model.linelimit_ne2.pprint()

# %%Define set, param and var for generator
f = [1, 2, 3, 4, 5]
model.GENERATORS = Set(initialize=f)

g = list(zip(gen[:, 0], f))
model.GNCONNECT = Set(initialize=g, dimen=2)
# model.GNCONNECT.pprint()

model.Maxout = Param(model.GENERATORS, initialize=dict(zip(f, gen[:, 8])))
# model.Maxout.pprint()

model.Cost1 = Param(model.GENERATORS, initialize=dict(zip(f, gencost[:, 5])))
model.Cost2 = Param(model.GENERATORS, initialize=dict(zip(f, gencost[:, 4])))


def supply_bound(model, generator, scenario):
    return (0, model.Maxout[generator])


model.supply = Var(model.GENERATORS, model.SCENARIO, domain=NonNegativeReals, bounds=supply_bound)


# model.supply.pprint()

# %% Define constraints and objective for power system

# Nodal balance, Kirchhoff's current law
def NodeBalance_rule(model, node, scenario):
    return sum(
        model.supply[generator, scenario] for generator in model.GENERATORS if (node, generator) in model.GNCONNECT) \
           + sum(model.lineflow[i, node, scenario] for i in model.NODES if (i, node) in model.LINES) \
           - model.Demand[node] \
           - sum(model.lineflow[node, j, scenario] for j in model.NODES if (node, j) in model.LINES) \
           + sum(model.lineflow_ne[i, node, scenario] for i in model.NODES if (i, node) in model.NEWLINES) \
           - sum(model.lineflow_ne[node, j, scenario] for j in model.NODES if (node, j) in model.NEWLINES) \
           + model.uspower[node, scenario] \
           == 0


model.NodeBalance = Constraint(model.NODES, model.SCENARIO, rule=NodeBalance_rule)
# model.NodeBalance.pprint()

Bigm = 1000000


# Constraints of Ohm's law for lines that exist and did not fail
def OhmsLaw_rule1(model, nodei, nodej, s):
    m = model.lineflow[nodei, nodej, s] - (model.angle[nodei, s] - model.angle[nodej, s]) / model.Reactance[
        nodei, nodej]
    return -Bigm * (1 - model.St[nodei, nodej, s]) <= m


model.OhmsLaw1 = Constraint(model.LINES, model.SCENARIO, rule=OhmsLaw_rule1)


# model.OhmsLaw1.pprint()

def OhmsLaw_rule2(model, nodei, nodej, s):
    m = model.lineflow[nodei, nodej, s] - (model.angle[nodei, s] - model.angle[nodej, s]) / model.Reactance[
        nodei, nodej]
    return m <= Bigm * (1 - model.St[nodei, nodej, s])


model.OhmsLaw2 = Constraint(model.LINES, model.SCENARIO, rule=OhmsLaw_rule2)


# model.OhmsLaw2.pprint()

# Constraints of Ohm's law for line candidates
# BigM = Param(initialize=1000000)
def OhmsLaw_ne_rule1(model, i, j, s):
    m = model.lineflow_ne[i, j, s] - ((model.angle[i, s] - model.angle[j, s]) / model.Reactance_ne[i, j])
    return -Bigm * (1 - model.zt[i, j]) <= m


model.OhmsLaw_ne1 = Constraint(model.NEWLINES, model.SCENARIO, rule=OhmsLaw_ne_rule1)


# model.OhmsLaw_ne1.pprint()

def OhmsLaw_ne_rule2(model, i, j, s):
    m = model.lineflow_ne[i, j, s] - ((model.angle[i, s] - model.angle[j, s]) / model.Reactance_ne[i, j])
    return m <= Bigm * (1 - model.zt[i, j])


model.OhmsLaw_ne2 = Constraint(model.NEWLINES, model.SCENARIO, rule=OhmsLaw_ne_rule2)
# model.OhmsLaw_ne2.pprint()

# %% Construct natural gas system part of the model
# import gas data from json and turn into dictionary

# figure out how to turn dictionary into param. Maybe i can use dict directly.
import json

belg = {}
with open('belgian.json', 'r') as f:
    belg = json.load(f)

print(belg.keys())

# %% Read the gas data
# Compute the maximum volume for gas modeL
Max_flow = 0
for idx, producer in belg['producer'].items():
    if producer['qgmax'] > 0:
        Max_flow += producer['qgmax']

# Compute the max mass flow in the gas moel
Max_mass_flow = Max_flow * belg['standard_density']

lst = ['pipe', 'compressor', 'short_pipe', 'resistor', 'valve', 'control_valve', 'ne_pipe', 'ne_compressor', 'junction',
       'consumer', 'producer']
# The field yp and yn are used for directionality. If directed in included, we add yp and yn
for key in belg.keys():
    for k in lst:
        for idx, component in belg[k].items():
            if 'directed' in component:
                if component['directed'] == 1:
                    component.update({'yp': 1, 'yn': 0})
                if component['directed'] == -1:
                    component.update({'yp': 0, 'yn': 1})

# Add degree information on junction? No parallel connection in the belgian network
if 'parallel_connections' in belg:
    print('There is parallel connections and ADD DEGREE')
if 'all_parallel_connections' in belg:
    print('There is all parallel connection and CHANGE DEGREE')

# Add the bound for minimum and maximum pressure for the pipes/connections
import itertools

for idx, connection in belg['connection'].items():
    i_idx = connection['f_junction']
    j_idx = connection['t_junction']

    i = belg['junction'][str(i_idx)]  # dictionary of the junction that connection comes from
    j = belg['junction'][str(j_idx)]

    pd_max = i['pmax'] ** 2 - j['pmin'] ** 2
    pd_min = i['pmin'] ** 2 - j['pmax'] ** 2

    connection["pd_max"] = pd_max
    connection["pd_min"] = pd_min

# Calculates pipeline resistance from thie paper Thorley and CH Tiley. Unsteady
# and transient flow of compressible fluids in pipelines- a review of theoretical and some experimental studies
# international journal of heat and fluid flow
# This calculation expresses resistance in terms of mass flow equations
R = 8.314  # universal gas constant
z = belg['compressibility_factor']
T = belg['temperature']
m = belg['gas_molar_mass']
import math

for idx, pipe in belg['pipe'].items():
    lambdaff = pipe['friction_factor']
    D = pipe['diameter']
    L = pipe['length']

    a_sqr = z * (R / m) * T
    A = (math.pi * D ** 2) / 4
    resistance = ((D * A ** 2) / (lambdaff * L * a_sqr)) * (belg['baseP'] ** 2 / belg['baseQ'] ** 2)

    pipe.update({'resistance': resistance})

# Check if per-unit is true. If not, scale junction, consumer and producer by pbase and qbase
if belg['per_unit']:
    print('per_unit is true')
else:
    print('Need to scale junction, consumer, produer because per_unit is not true')

# f: mass flow  q:volume flow  l:consumer  g:producer
standard_density = belg['standard_density']

# Calculates minimum mass flow consumption
for idx, consumer in belg['consumer'].items():
    flmin = consumer['qlmin'] * standard_density
    consumer.update({'flmin': flmin})

# Calculates maximum mass flow consumption
for idx, consumer in belg['consumer'].items():
    flmax = consumer['qlmax'] * standard_density
    consumer.update({'flmax': flmax})

# Calculates constant mass flow consumption
for idx, consumer in belg['consumer'].items():
    fl = consumer['ql'] * standard_density
    consumer.update({'fl': fl})

# Calculates minimum mass flow production
for idx, producer in belg['producer'].items():
    fgmin = producer['qgmin'] * standard_density
    producer.update({'fgmin': fgmin})

# Calculates maximum mass flow production
for idx, producer in belg['producer'].items():
    fgmax = producer['qgmax'] * standard_density
    producer.update({'fgmax': fgmax})

# Calculates constant mass flow production
for idx, producer in belg['producer'].items():
    fg = producer['qg'] * standard_density
    producer.update({'fg': fg})

# %%Define Set for natural gas model

# Define junction related set, parameter and variables
model.JUNCTION = Set(initialize=belg['junction'].keys())
# model.JUNCTION.pprint()

pmin_dict = dict((k, belg['junction'][k]['pmin']) for k in belg['junction'].keys())
model.Pmin = Param(model.JUNCTION, initialize=pmin_dict)
# model.Pmin.pprint()

pmax_dict = dict((k, belg['junction'][k]['pmax']) for k in belg['junction'].keys())
model.Pmax = Param(model.JUNCTION, initialize=pmax_dict)


# model.Pmax.pprint()

# Variable pressure squared for each junction
def pressure_sqr_bound(model, junction, scenario):
    return (model.Pmin[junction] ** 2, model.Pmax[junction] ** 2)


model.pressure_sqr = Var(model.JUNCTION, model.SCENARIO, bounds=pressure_sqr_bound)
# model.pressure_sqr.pprint()

# %%Define connection related set, parameter and variables
model.CONNECTION = Set(initialize=belg['connection'].keys())
# model.CONNECTION.pprint()

Pdmax_dict = dict((k, belg['connection'][k]['pd_max']) for k in belg['connection'].keys())
model.Pdmax = Param(model.CONNECTION, initialize=Pdmax_dict)
# model.Pdmax.pprint()

Pdmin_dict = dict((k, belg['connection'][k]['pd_min']) for k in belg['connection'].keys())
model.Pdmin = Param(model.CONNECTION, initialize=Pdmin_dict)


# model.Pdmin.pprint()

# Variable mass flow for connection
def flow_bound(model, connection, scenario):
    return (-Max_mass_flow, Max_mass_flow)


model.flow = Var(model.CONNECTION, model.SCENARIO, bounds=flow_bound)
# model.flow.pprint()

# Binary variables associated with direction of the flow in the connections
ypdict = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 0, '10': 1, '11': 1, '12': 1, '13': 1,
          '14': 1, '15': 1, '16': 1, '17': 0, '18': 1, '19': 1, '20': 1,
          '21': 1, '22': 1, '23': 1, '24': 1, '100002': 0, '100001': 0, '111': 1, '101': 1, '100000': 0, '221': 1}
yndict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 1, '10': 0, '11': 0, '12': 0, '13': 0,
          '14': 0, '15': 0, '16': 0, '17': 1, '18': 0, '19': 0, '20': 0,
          '21': 0, '22': 0, '23': 0, '24': 0, '100002': 1, '100001': 1, '111': 0, '101': 0, '100000': 1, '221': 0}
# model.yp = Param(model.CONNECTION,domain=Binary,initialize=ypdict)
# model.yn = Param(model.CONNECTION,domain=Binary,initialize=yndict)
model.yp = Var(model.CONNECTION, model.SCENARIO, domain=Binary, initialize=1)
model.yn = Var(model.CONNECTION, model.SCENARIO, domain=Binary, initialize=0)
# model.yp.pprint()
# model.yn.pprint()


# Auxiliary relaxation variable
model.lamd = Var(model.CONNECTION, model.SCENARIO, initialize=1)
# model.lamd.pprint()

# %% Define pipe related set, parameter
model.PIPE = Set(within=model.CONNECTION, initialize=belg['pipe'].keys())
# model.PIPE.pprint()

resistance_dict = dict((k, belg['pipe'][k]['resistance']) for k in belg['pipe'].keys())
model.Resistance = Param(model.PIPE, initialize=resistance_dict)
# model.Resistance.pprint()

# Status of PIPE
model.Sp = Param(model.PIPE, model.SCENARIO, initialize=1, mutable=True)
# model.Sp.pprint()

# %% New connection Set
# There can be parameters for new compressors but no varialbes
# Constraints on the set of new connection will be changed into on the set of new pipe
# Constraints on the set of new compressors will be removed

model.NE_CONNECTION = Set(initialize=belg['connection'].keys())
model.NE_CONNECTION.pprint()

# New pipeline candidates
model.NE_PIPE = Set(within=model.NE_CONNECTION, initialize=belg['pipe'].keys())
# model.NE_PIPE.pprint()

model.Pdmax_ne = Param(model.NE_CONNECTION, initialize=Pdmax_dict)
# model.Pdmax.pprint()

model.Pdmin_ne = Param(model.NE_CONNECTION, initialize=Pdmin_dict)
# model.Pdmin.pprint()

# Variable mass flow for ne connection
model.flow_ne = Var(model.NE_PIPE, model.SCENARIO, bounds=flow_bound)
# model.flow_ne.pprint()

# Flow direction binary variables
model.yp_ne = Var(model.NE_PIPE, model.SCENARIO, domain=Binary, initialize=1)
model.yn_ne = Var(model.NE_PIPE, model.SCENARIO, domain=Binary, initialize=0)
# model.yp_ne.pprint()
# model.yn_ne.pprint()

model.lamd_ne = Var(model.NE_PIPE, model.SCENARIO, initialize=1)
# model.lamd_ne.pprint()

# There is not binary variable for connection investment decisions because they are seperate investment decisions

model.Resistance_ne = Param(model.NE_PIPE, initialize=resistance_dict)

model.zp = Var(model.NE_PIPE, domain=Binary, initialize=0)
# model.zp.pprint()


pipe_cost = 7 * 10 ** 6
pricep_dict = {k: pipe_cost for k in belg['pipe'].keys()}
model.Pricep = Param(model.NE_PIPE, initialize=pricep_dict)
# model.Pricep.pprint()

# %%Set related with compressors
model.COMPRESSOR = Set(within=model.CONNECTION, initialize=belg['compressor'].keys())
# model.COMPRESSOR.pprint()

max_ratio_dict = dict((k, belg['compressor'][k]['c_ratio_max']) for k in belg['compressor'].keys())
model.Max_ratio = Param(model.COMPRESSOR, initialize=max_ratio_dict)
# model.Max_ratio.pprint()

min_ratio_dict = dict((k, belg['compressor'][k]['c_ratio_min']) for k in belg['compressor'].keys())
model.Min_ratio = Param(model.COMPRESSOR, initialize=min_ratio_dict)
# model.Min_ratio.pprint()

# Joint set of compressor set and from junction
compressor_junctionf_list = list((k, str(belg['compressor'][k]['f_junction'])) for k in belg['compressor'].keys())
model.COMPRESSOR_JUNCTIONF = Set(dimen=2, initialize=compressor_junctionf_list)
# model.COMPRESSOR_JUNCTIONF.pprint()

# Joint set of compressor and to junction
compressor_junctiont_list = list((k, str(belg['compressor'][k]['t_junction'])) for k in belg['compressor'].keys())
model.COMPRESSOR_JUNCTIONT = Set(dimen=2, initialize=compressor_junctiont_list)
# model.COMPRESSOR_JUNCTIONT.pprint()

# %% Set related to new compressors
# model.NE_COMPRESSOR = Set(within=model.NE_CONNECTION,initialize=belg['compressor'].keys())

# model.Max_ratio_ne = Param(model.NE_COMPRESSOR,initialize=max_ratio_dict)

# model.Min_ratio_ne = Param(model.NE_COMPRESSOR,initialize=min_ratio_dict)

# Binary variable to represent whether compressors are built
# model.zc = Var(model.NE_COMPRESSOR,domain=Binary,initialize=0)
# model.zc.pprint()

# compressor_cost = 1.5*10**6
# pricec_dict ={k:compressor_cost for k in belg['compressor'].keys()}
# model.Pricec= Param(model.NE_COMPRESSOR,initialize=pricec_dict)
# model.Pricec.pprint()

# %% Set and variables for consumer/demand

# Conclusion: consumer 4 and consumer 10012 are electricity generators for natural gas
# The rest of the nodes just have fixed non-electricity demand for natural gas
# according to the graph, 4 and 12 junction has gas consumers. both 12 and 10012 in consumer
# are connected to junction 12.

model.CONSUMER = Set(initialize=belg['consumer'].keys())
# model.CONSUMER.pprint()

# fl : mass flow load
# gas demand for non-electricity use ql
fl_dict = dict((k, belg['consumer'][k]['fl']) for k in belg['consumer'].keys())
model.Fl = Param(model.CONSUMER, initialize=fl_dict)
# model.Fl.pprint()

# gas demand maximum at consumer node
flmax_dict = dict((k, belg['consumer'][k]['flmax']) for k in belg['consumer'].keys())
model.Flmax = Param(model.CONSUMER, initialize=flmax_dict)
# model.Flmax.pprint()

# gas demand minimum at consumer node
flmin_dict = dict((k, belg['consumer'][k]['flmin']) for k in belg['consumer'].keys())
model.Flmin = Param(model.CONSUMER, initialize=flmin_dict)


# model.Flmin.pprint()

# Gas demand for electricity use
def fl_elec_bound(model, consumer, scenario):
    return (0, model.Flmax[consumer])
    # return (0,model.Flmax[consumer])


model.fl_elec = Var(model.CONSUMER, model.SCENARIO, bounds=fl_elec_bound)


# model.fl_elec.pprint()

def usgas_bound(model, consumer, scenrio):
    return (0, model.Flmax[consumer])


model.usgas = Var(model.CONSUMER, model.SCENARIO, domain=NonNegativeReals, bounds=usgas_bound, initialize=0)


# model.usgas.pprint()

def junction_gas_demand_rule(model, consumer, s):
    return inequality(model.Flmin[consumer], model.usgas[consumer, s] + model.fl_elec[consumer, s],
                      model.Flmax[consumer])


model.junction_gas_demand = Constraint(model.CONSUMER, model.SCENARIO, rule=junction_gas_demand_rule)
# model.junction_gas_demand.pprint()

# %% Set, parameter and variables related to Producer
model.PRODUCER = Set(initialize=belg['producer'].keys())
# model.PRODUCER.pprint()

# fg: mass flow generation
# constant gas production
fg_dict = dict((k, belg['producer'][k]['fg']) for k in belg['producer'].keys())
model.Fg = Param(model.PRODUCER, initialize=fg_dict)
# model.Fg.pprint()

# gas supply/generation maximum at producer node
fgmax_dict = dict((k, belg['producer'][k]['fgmax']) for k in belg['producer'].keys())
model.Fgmax = Param(model.PRODUCER, initialize=fgmax_dict)
# model.Fgmax.pprint()

# gas supply minimum at producer node
fgmin_dict = dict((k, belg['producer'][k]['fgmin']) for k in belg['producer'].keys())
model.Fgmin = Param(model.PRODUCER, initialize=fgmin_dict)


# model.Fgmin.pprint()

# Gas generation variable
def fg_extra_bound(model, producer, scenario):
    # return (0, model.Fgmax[producer])
    return (0, model.Fgmax[producer])


model.fg_extra = Var(model.PRODUCER, model.SCENARIO, bounds=fg_extra_bound)
# model.fg_extra.pprint()

# model.gas_waste= Var(model.PRODUCER,bounds=fg_extra_bound)
# def junction_gas_production_rule(model,producer):
#    return model.Fgmin[producer]<= model.gas_waste[producer]+model.fg_extra[producer]<= model.Fgmax[producer]
# model.junction_gas_production = Constraint(model.PRODUCER,rule=junction_gas_production_rule)
# model.junction_gas_production.pprint()

# %% Joint set that connects producer with junction because there are multiple producers at one junction
# also, joint set that connects consumers with junction because of multiple consumers at one junction

consumer_junction_list = list((k, str(belg['consumer'][k]['ql_junc'])) for k in belg['consumer'].keys())
model.CONSUMER_JUNCTION = Set(dimen=2, initialize=consumer_junction_list)
# model.CONSUMER_JUNCTION.pprint()


producer_junction_list = list((k, str(belg['producer'][k]['qg_junc'])) for k in belg['producer'].keys())
model.PRODUCER_JUNCTION = Set(dimen=2, initialize=producer_junction_list)
# model.PRODUCER_JUNCTION.pprint()

# Joint set of connections and from junctions
connection_junctionf_list = list((k, str(belg['connection'][k]['f_junction'])) for k in belg['connection'].keys())
model.CONNECTION_JUNCTIONF = Set(dimen=2, initialize=connection_junctionf_list)
# model.CONNECTION_JUNCTIONF.pprint()

# Joint set of connection and to junctions
connection_junctiont_list = list((k, str(belg['connection'][k]['t_junction'])) for k in belg['connection'].keys())
model.CONNECTION_JUNCTIONT = Set(dimen=2, initialize=connection_junctiont_list)


# model.CONNECTION_JUNCTIONT.pprint()

# %%Define constraints that does not involve new expansion or load shedding

# mass flow balance equation for junctions where demand and production is fixed. Equation 2
def junction_mass_flow_balance_rule(model, junction, s):
    return sum(
        model.fg_extra[producer, s] for producer in model.PRODUCER if (producer, junction) in model.PRODUCER_JUNCTION) \
           - sum(model.fl_elec[consumer, s]for consumer in model.CONSUMER if
                 (consumer, junction) in model.CONSUMER_JUNCTION) \
           + sum(model.flow[connection, s] for connection in model.CONNECTION if
                 (connection, junction) in model.CONNECTION_JUNCTIONT) \
           - sum(model.flow[connection, s] for connection in model.CONNECTION if
                 (connection, junction) in model.CONNECTION_JUNCTIONF) \
           + sum(
        model.flow_ne[ne_pipe, s] for ne_pipe in model.NE_PIPE if (ne_pipe, junction) in model.CONNECTION_JUNCTIONT) \
           - sum(
        model.flow_ne[ne_pipe, s] for ne_pipe in model.NE_PIPE if (ne_pipe, junction) in model.CONNECTION_JUNCTIONF) \
           == 0


model.junction_mass_flow_balance = Constraint(model.JUNCTION, model.SCENARIO, rule=junction_mass_flow_balance_rule)


# model.junction_mass_flow_balance.pprint()

# %%
# constraint on the flow direction on connection. Equation 3
def flow_direction_choice1_rule(model, compressor, s):
    return model.yp[compressor, s] + model.yn[compressor, s] == 1


model.flow_direction_choice1 = Constraint(model.COMPRESSOR, model.SCENARIO, rule=flow_direction_choice1_rule)


# model.flow_direction_choice1 = Constraint(model.CONNECTION,rule=flow_direction_choice1_rule)
# model.flow_direction_choice1.pprint()

def flow_direction_choice2_rule(model, pipe, s):
    return model.yp[pipe, s] + model.yn[pipe, s] == model.Sp[pipe, s]


model.flow_direction_choice2 = Constraint(model.PIPE, model.SCENARIO, rule=flow_direction_choice2_rule)


# model.flow_direction_choice2.pprint()

# flow direction for new connection

def flow_direction_choice_ne2_rule(model, nepipe, s):
    return model.yp_ne[nepipe, s] + model.yn_ne[nepipe, s] == model.zp[nepipe]


model.flow_direction_choice_ne2 = Constraint(model.NE_PIPE, model.SCENARIO, rule=flow_direction_choice_ne2_rule)


# model.flow_direction_choice_ne2.pprint()

def flow_direction_choice2_rule(model, compressor):
    return model.yp_ne[compressor] + model.yn_ne[compressor] == 1


# model.flow_direction_choice1 = Constraint(model.COMPRESSOR,rule=flow_direction_choice1_rule)
# model.flow_direction_choice2 = Constraint(model.NE_CONNECTION,rule=flow_direction_choice2_rule)

# %%
# Constraint on pressure drop across pipes equation. APPLIES TO PIPE. Equation 7
# Pressure drop alighs with flow direction
def pressure_drop_direction1_rule(model, pipe, s):
    yp = model.yp[pipe, s]
    yn = model.yn[pipe, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONT)
    return (1 - yp) * model.Pdmin[pipe] <= pi - pj


model.pressure_drop_direction1 = Constraint(model.PIPE, model.SCENARIO, rule=pressure_drop_direction1_rule)


# model.pressure_drop_direction1.pprint()

def pressure_drop_direction2_rule(model, pipe, s):
    yp = model.yp[pipe, s]
    yn = model.yn[pipe, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONT)
    return pi - pj <= (1 - yn) * model.Pdmax[pipe]


model.pressure_drop_direction2 = Constraint(model.PIPE, model.SCENARIO, rule=pressure_drop_direction2_rule)


# model.pressure_drop_direction2.pprint()

# Equation 7 for NEW PIPE
# flow goes from high pressure to low pressure, pressure drop is within the pressure drop limit for a pipe
def pressure_drop_direction_ne1_rule(model, ne_pipe, s):
    yp_ne = model.yp_ne[ne_pipe, s]
    yn_ne = model.yn_ne[ne_pipe, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_pipe, i) in model.CONNECTION_JUNCTIONT)
    return (1 - yp_ne) * model.Pdmin_ne[ne_pipe] <= pi - pj


model.pressure_drop_direction_ne1 = Constraint(model.NE_PIPE, model.SCENARIO, rule=pressure_drop_direction_ne1_rule)


# model.pressure_drop_direction_ne1.pprint()

def pressure_drop_direction_ne2_rule(model, ne_pipe, s):
    yp_ne = model.yp_ne[ne_pipe, s]
    yn_ne = model.yn_ne[ne_pipe, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_pipe, i) in model.CONNECTION_JUNCTIONT)
    return pi - pj <= (1 - yn_ne) * model.Pdmax_ne[ne_pipe]


model.pressure_drop_direction_ne2 = Constraint(model.NE_PIPE, model.SCENARIO, rule=pressure_drop_direction_ne2_rule)


# model.pressure_drop_direction_ne2.pprint()

# %%
# Constraint that ensure the flow directionality is tied to the sign of flow.
# FOR CONNECTION. Equation 5 and 6
# Connect the flow direction yp yn with flow. flow between 0 and maxflow based on the direction
def compressor_flow_direction1_rule(model, compressor, s):
    yp = model.yp[compressor, s]
    mf = Max_mass_flow
    return -(1 - yp) * mf <= model.flow[compressor, s]


model.compressor_flow_direction1 = Constraint(model.COMPRESSOR, model.SCENARIO, rule=compressor_flow_direction1_rule)


# model.compressor_flow_direction1.pprint()

def compressor_flow_direction2_rule(model, compressor, s):
    yn = model.yn[compressor, s]
    mf = Max_mass_flow
    return model.flow[compressor, s] <= (1 - yn) * mf


model.compressor_flow_direction2 = Constraint(model.COMPRESSOR, model.SCENARIO, rule=compressor_flow_direction2_rule)


# model.compressor_flow_direction2.pprint()

def pipe_flow_direction1_rule(model, pipe, s):
    yp = model.yp[pipe, s]
    mf = Max_mass_flow
    pdmin = model.Pdmin[pipe]
    pdmax = model.Pdmax[pipe]
    w = model.Resistance[pipe]
    f = model.flow[pipe, s]
    return -(1 - yp) * min(mf, sqrt(w * max(pd_max, abs(pd_min)))) <= f


model.pipe_flow_direction1 = Constraint(model.PIPE, model.SCENARIO, rule=pipe_flow_direction1_rule)


# model.pipe_flow_direction1.pprint()

def pipe_flow_direction2_rule(model, pipe, s):
    yn = model.yn[pipe, s]
    mf = Max_mass_flow
    pdmin = model.Pdmin[pipe]
    pdmax = model.Pdmax[pipe]
    w = model.Resistance[pipe]
    f = model.flow[pipe, s]
    return f <= (1 - yn) * min(mf, sqrt(w * max(pd_max, abs(pd_min))))


model.pipe_flow_direction2 = Constraint(model.PIPE, model.SCENARIO, rule=pipe_flow_direction2_rule)


# model.pipe_flow_direction2.pprint()

# %%
# Flow directionality is tied to the sign of flow for new connetions
def pipe_flow_direction_ne1_rule(model, nepipe, s):
    yp = model.yp_ne[nepipe, s]
    mf = Max_mass_flow
    pdmin = model.Pdmin[nepipe]
    pdmax = model.Pdmax[nepipe]
    w = model.Resistance[nepipe]
    f = model.flow_ne[nepipe, s]
    return -(1 - yp) * min(mf, sqrt(w * max(pd_max, abs(pd_min)))) <= f


model.pipe_flow_direction_ne1 = Constraint(model.NE_PIPE, model.SCENARIO, rule=pipe_flow_direction_ne1_rule)


# model.pipe_flow_direction_ne1.pprint()

def pipe_flow_direction_ne2_rule(model, nepipe, s):
    yn = model.yn_ne[nepipe, s]
    mf = Max_mass_flow
    pdmin = model.Pdmin[nepipe]
    pdmax = model.Pdmax[nepipe]
    w = model.Resistance[nepipe]
    f = model.flow_ne[nepipe, s]
    return f <= (1 - yn) * min(mf, sqrt(w * max(pd_max, abs(pd_min))))


model.pipe_flow_direction_ne2 = Constraint(model.NE_PIPE, model.SCENARIO, rule=pipe_flow_direction_ne2_rule)


# model.pipe_flow_direction_ne2.pprint()


# %%
# Weymouth equation for pipes, define multiple equations within one function
# FOR PIPES WITHIN CONNECTIONS
def weymouth_rule1(model, connection, s):
    yp = model.yp[connection, s]
    yn = model.yn[connection, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin[connection]
    pdmax = model.Pdmax[connection]
    return model.lamd[connection, s] >= pj - pi + pdmin * (yp - yn + 1)


model.weymouth1 = Constraint(model.PIPE, model.SCENARIO, rule=weymouth_rule1)


# model.weymouth1.pprint()

def weymouth_rule2(model, connection, s):
    yp = model.yp[connection, s]
    yn = model.yn[connection, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin[connection]
    pdmax = model.Pdmax[connection]
    return model.lamd[connection, s] >= pi - pj + pdmax * (yp - yn - 1)


model.weymouth2 = Constraint(model.PIPE, model.SCENARIO, rule=weymouth_rule2)


# model.weymouth2.pprint()

def weymouth_rule3(model, connection, s):
    yp = model.yp[connection, s]
    yn = model.yn[connection, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin[connection]
    pdmax = model.Pdmax[connection]
    return model.lamd[connection, s] <= pj - pi + pdmax * (yp - yn + 1)


model.weymouth3 = Constraint(model.PIPE, model.SCENARIO, rule=weymouth_rule3)


# model.weymouth3.pprint()

def weymouth_rule4(model, connection, s):
    yp = model.yp[connection, s]
    yn = model.yn[connection, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin[connection]
    pdmax = model.Pdmax[connection]
    return model.lamd[connection, s] <= pi - pj + pdmin * (yp - yn - 1)


model.weymouth4 = Constraint(model.PIPE, model.SCENARIO, rule=weymouth_rule4)


# model.weymouth4.pprint()

def weymouth_rule5(model, connection, s):
    f = model.flow[connection, s]
    w = model.Resistance[connection]
    sp = model.Sp[connection, s]
    return sp * w * model.lamd[connection, s] >= f ** 2


model.weymouth5 = Constraint(model.PIPE, model.SCENARIO, rule=weymouth_rule5)


# model.weymouth5.pprint()

def weymouth_rule1_ne(model, ne_connection, s):
    yp_ne = model.yp_ne[ne_connection, s]
    yn_ne = model.yn_ne[ne_connection, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin_ne[ne_connection]
    pdmax = model.Pdmax_ne[ne_connection]
    return model.lamd_ne[ne_connection, s] >= pj - pi + pdmin * (yp_ne - yn_ne + 1)


model.weymouth1_ne = Constraint(model.NE_PIPE, model.SCENARIO, rule=weymouth_rule1_ne)


# model.weymouth1_ne.pprint()

def weymouth_rule2_ne(model, ne_connection, s):
    yp_ne = model.yp_ne[ne_connection, s]
    yn_ne = model.yn_ne[ne_connection, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin_ne[ne_connection]
    pdmax = model.Pdmax_ne[ne_connection]
    return model.lamd_ne[ne_connection, s] >= pi - pj + pdmax * (yp_ne - yn_ne - 1)


model.weymouth2_ne = Constraint(model.NE_PIPE, model.SCENARIO, rule=weymouth_rule2_ne)


# model.weymouth2_ne.pprint()

def weymouth_rule3_ne(model, ne_connection, s):
    yp_ne = model.yp_ne[ne_connection, s]
    yn_ne = model.yn_ne[ne_connection, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin_ne[ne_connection]
    pdmax = model.Pdmax_ne[ne_connection]
    return model.lamd_ne[ne_connection, s] <= pj - pi + pdmax * (yp_ne - yn_ne + 1)


model.weymouth3_ne = Constraint(model.NE_PIPE, model.SCENARIO, rule=weymouth_rule3_ne)


# model.weymouth3_ne.pprint()

def weymouth_rule4_ne(model, ne_connection, s):
    yp_ne = model.yp_ne[ne_connection, s]
    yn_ne = model.yn_ne[ne_connection, s]
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin_ne[ne_connection]
    pdmax = model.Pdmax_ne[ne_connection]
    return model.lamd_ne[ne_connection, s] <= pi - pj + pdmin * (yp_ne - yn_ne - 1)


model.weymouth4_ne = Constraint(model.NE_PIPE, model.SCENARIO, rule=weymouth_rule4_ne)


# model.weymouth4_ne.pprint()

def weymouth_rule5_ne(model, ne_connection, s):
    f = model.flow_ne[ne_connection, s]
    w = model.Resistance[ne_connection]
    zp = model.zp[ne_connection]
    return w * zp * model.lamd_ne[ne_connection, s] >= f ** 2


model.weymouth5_ne = Constraint(model.NE_PIPE, model.SCENARIO, rule=weymouth_rule5_ne)


# model.weymouth5_ne.pprint()


# %%
# Compression falls within the compression ratio limits of the compressors
# Equation 8 to 11
def compression_rule1(model, connection, s):
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection, s]
    yn = model.yn[connection, s]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]

    return pj - max_ratio ** 2 * pi <= (1 - yp) * (j_pmax ** 2)


model.compression1 = Constraint(model.COMPRESSOR, model.SCENARIO, rule=compression_rule1)


# model.compression1.pprint()

def compression_rule2(model, connection, s):
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection, s]
    yn = model.yn[connection, s]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return min_ratio ** 2 * pi - pj <= (1 - yp) * (i_pmax ** 2)


model.compression2 = Constraint(model.COMPRESSOR, model.SCENARIO, rule=compression_rule2)


# model.compression2.pprint()

def compression_rule3(model, connection, s):
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection, s]
    yn = model.yn[connection, s]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return pi - pj <= (1 - yn) * (i_pmax ** 2)


model.compression3 = Constraint(model.COMPRESSOR, model.SCENARIO, rule=compression_rule3)


# model.compression3.pprint()

def compression_rule4(model, connection, s):
    pi = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i, s] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection, s]
    yn = model.yn[connection, s]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return pj - pi <= (1 - yn) * (j_pmax ** 2)


model.compression4 = Constraint(model.COMPRESSOR, model.SCENARIO, rule=compression_rule4)


# model.compression4.pprint()


def pipe_flow_ne1_rule(model, nepipe, s):
    return model.flow_ne[nepipe, s] <= model.zp[nepipe] * Max_mass_flow


model.pipe_flow_ne1 = Constraint(model.NE_PIPE, model.SCENARIO, rule=pipe_flow_ne1_rule)
model.pipe_flow_ne1.pprint()


def pipe_flow_ne2_rule(model, nepipe, s):
    return model.flow_ne[nepipe, s] >= -model.zp[nepipe] * Max_mass_flow


model.pipe_flow_ne2 = Constraint(model.NE_PIPE, model.SCENARIO, rule=pipe_flow_ne2_rule)
model.pipe_flow_ne2.pprint()

# %% Connect gas with power model
# column_names%  consumer  heat_rate_quad_coeff   heat_rate_linear_coeff   heat_rate_constant_coeff
gen_gas = array([
    [-1, 0, 0, 0],
    [4, 0, 140674.111111, 0],
    [10012, 0, 140674.111111, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0]
])

# Set of gasnode with gas-fired power plants
gasnode_list = [str(int(k)) for k in gen_gas[:, 0] if k > 0]
model.GASNODE = Set(within=model.CONSUMER, initialize=gasnode_list)
# model.GASNODE.pprint()

model.NONGASGEN = [1, 4, 5]

# Joint set of generator and consumer
gen_consumer_list = [(k, str(int(gen_gas[k - 1, 0]))) for k in [1, 2, 3, 4, 5]]
model.GENERATOR_CONSUMER = Set(initialize=gen_consumer_list, dimen=2)
# model.GENERATOR_CONSUMER.pprint()

# Define the linear heat rate coefficient
model.Heatrate_linear = Param(model.GENERATORS, initialize=dict(zip([1, 2, 3, 4, 5], gen_gas[:, 2])))
# model.Heatrate_linear.pprint()

heat_value = 2.436 * 10 ** -8  # Joules/s --> m3/s: m3/joules
energy_factor = heat_value / belg['baseQ'] * standard_density

# MVA is a scaling factor that is commonly used in IEEE bus networks of normalization
# mva= mega volt amper
mvaBase = 100


# heat rate curve contraint for gas nodes with gas fired power plants
def heatrate_rule(model, consumer, scenario):
    # d = model.fl_elec[consumer]
    # H2 = sum(model.Heatrate_linear[generator] for generator in model.GENERATORS if (generator,consumer) in model.GENERATOR_CONSUMER)
    # pg = (model.supply[generator]  for generator in model.GENERATORS if (generator,consumer) in model.GENERATOR_CONSUMER)
    return model.fl_elec[consumer, scenario] == 0.167 * energy_factor * sum(
        model.Heatrate_linear[generator] * mvaBase * model.supply[generator, scenario] for generator in model.GENERATORS
        if (generator, consumer) in model.GENERATOR_CONSUMER)


model.heatrate = Constraint(model.GASNODE, model.SCENARIO, rule=heatrate_rule)
# model.heatrate.pprint()

# %% change the status parameter of pipelines and transmission lines of each scenario
# based on the reduce scenario data

for scenario in model.SCENARIO:
    fc_list = eval(df_rs.iloc[scenario, 2])

    for fc in fc_list:
        if type(fc) is tuple:
            model.St[fc, scenario] = 0
        else:
            model.Sp[fc, scenario] = 0

# %%Add unserved electricity and natural gas as variables

# power_penalty = 10 ** 6
# gas_penalty = 220 * 10 ** 6
penalty = penalty_value

# %%Objective of the model
# gascost1 = 86400*86400*(5.917112528134105e-8)*(belg['baseQ']**2)
# print('gas cost quadratic coefficient',gascost1)
# 0.01822660578829469
costq = 0.0778  # $/m3
gascost = 86400 * costq
print('gas cost linear coefficient', gascost)


# (model.fl_elec[consumer]/standard_density)**2 *gascost1\
# model.totalpowercost = Var()    
# def total_power_cost_rule(model):
#     return model.totalpowercost == (24*sum(model.supply[generator]*model.Cost1[generator]\
#                + model.Cost2[generator]*(model.supply[generator])**2 for generator in model.NONGASGEN))
# model.total_power_cost = Constraint(rule=total_power_cost_rule)
# model.totalpowercost.pprint()

# model.totalgascost=Var()
# def total_gas_cost_rule(model):
#     return model.totalgascost == sum(model.fl_elec[consumer]*gascost/standard_density*belg['baseQ'] for consumer in model.CONSUMER if consumer in model.GASNODE)
# model.total_gas_cost = Constraint(rule=total_gas_cost_rule)
# model.totalgascost.pprint()


def Obj_rule(model):
    return sum(model.Weight[scenario] * (24 * sum(model.supply[generator, scenario] * model.Cost1[generator] \
                                                  + model.Cost2[generator] * (
                                                      model.supply[generator, scenario]) ** 2 for generator in
                                                  model.NONGASGEN) \
                                         + sum(
                model.fl_elec[consumer, scenario] / standard_density * belg['baseQ'] * gascost for consumer in
                model.CONSUMER if consumer in model.GASNODE) \
                                         + 24 * mvaBase * penalty * sum(
                model.uspower[node, scenario] for node in model.NODES) \
                                         + 86400 * penalty * 2.78 * 10 ** (-10) * belg['baseQ']  * sum(
                model.usgas[consumer, scenario] for consumer in model.CONSUMER)/heat_value ) for scenario in model.SCENARIO) \
           + sum(model.Pricet[newline] * model.zt[newline] for newline in model.NEWLINES) \
           + sum(model.Pricep[newpipe] * model.zp[newpipe] for newpipe in model.NE_PIPE)  # \
    # + sum(model.Pricec[newcomp]*model.zc[newcomp] for newcomp in model.NE_COMPRESSOR)


model.Obj = Objective(rule=Obj_rule, sense=minimize)
model.Obj.pprint()

# base p, junction pressure normalization
# base q, volume flow normalization

# %% Additional constraints
# Set the reference bus
for s in model.SCENARIO:
    model.angle[1, s].fix(0)

# Contingency scenarios
# model.St[6.0,13.0]=0
# model.St[12.0,13.0]=0
# model.St[13.0,14.0]=0

# model.Sp['9']=0
# model.Sp['8']=0
# model.Sp['5']=0


# model.parallel_rule = ConstraintList()
# model.parallel_rule.add(model.yp['1']==model.yp['2'])
# model.parallel_rule.add(model.yp['3']==model.yp['4'])
# model.parallel_rule.add(model.yp['14']==model.yp['15'])
# model.parallel_rule.add(model.yp['12']==model.yp['13'])
# model.parallel_rule.add(model.yp['101']==model.yp['111'])

# model.dual = Suffix(direction=Suffix.IMPORT)

# Fixing flow direction
# pos_list=['24', '12', '221', '101', '20', '6', '23', '22', '100001', '13', '21', '7', '8', '19', '10', '9', '18']
# neg_list=['2', '11', '100002', '100000', '5', '16', '14', '17']
# change_list=['4', '1', '15', '111', '3']

# for s in model.SCENARIO:
#     for idx in pos_list:
#         model.yp[idx,s].fix(1)
#     for idx in neg_list:
#         model.yp[idx,s].fix(0)
# model.yp.pprint()


# %%
io_options = dict()
io_options['solver'] = "baron"
io_options['mtype'] = "minlp"
io_options['symbolic_solver_labels'] = True
# # io_options['add_options']=['option resLim=3000;']

# model.write('rf_base5.gms', io_options=io_options)
# model.write('rf_cost8_h.gms', io_options=io_options)
model.write('de_rf10_e4.gms', io_options=io_options)
# model.write('de_rf15.gms', io_options=io_options)
# model.write('de_base10.gms', io_options=io_options)

# %%Solve the model
# io_options = dict()
# solver = SolverFactory('baron')
# solver =SolverFactory('cplex')
# solver=SolverFactory('knitroampl')
# io_options['MaxTime'] =-1
# io_options['solver'] = 'cplex'
# opt=SolverFactory('gams')
# io_options['solver']='couenne'
# io_options['solver']='cplex'
# io_options['solver']='knitro'
# io_options['add_options']=['optfile=1;']
# io_options['add_options']=['option resLim=1000000']
# io_options['solver']= ''
# io_options['mipgap']= 0.001
# opt.options['iisfind'] = 1
# opt.options['outlev'] = 1
# results = opt.solve(model,tee=True,logfile=True,keepfiles=True,tmpdir='/Users/carriesu/Desktop/Research/Coupled System Modeling Under Extreme Events/Code/14Belgium/Expansion/temp',warmstart=True,io_options=io_options)
# results = solver.solve(model,tee=True,timelimit=None)
# results = opt.solve(model,solver='knitro',tee=True,logfile=True,keepfiles=True,add_options=['GAMS_MODEL.optfile=1'])

# results=opt.solve(model,tee=True,logfile=True,io_options=io_options)
# results=opt.solve(model,solver='knitro',tee=True,logfile=True)

print(results)
model.display(filename="model_results_denc.txt")
# model.display(filename="model_results_minlp_no_contingency.txt")

text_file = open("Solver_results_denc.txt", 'w')
# text_file = open("Solver_results_minlp.txt",'w')

text_file.write(str(results))
text_file.close()

# Show infeasible constraints
from pyomo.util.infeasible import log_infeasible_constraints
# log_infeasible_constraints(model)

# model.supply.pprint()
# model.uspower.pprint()
# model.usgas.pprint()
# model.zt.pprint()
# model.zp.pprint()
# model.zc.pprint()
# model.lineflow.pprint()
# model.flow.pprint()
# model.yp.pprint()
# model.yn.pprint()
# model.flow_ne.pprint()

# %%

# # display all duals
# print ("Duals")
# for c in model.component_objects(Var, active=True):
#     print ("   Variable",c)
#     for index in c:
#         print ("      ", index, model.dual[c[index]])


# %%Print out model structure in text file
# text_file = open("MODEL_minlp_no_contingency.txt",'w')
# model.pprint(text_file)

# %% Write the code into GAMS file
# io_options = dict()
# io_options['solver'] = "knitro"
# io_options['mtype'] ="minlp"
# io_options['symbolic_solver_labels']=True
# io_options['add_options']=['GAMS_MODEL.optfile=1']

# model.write('GAMS_file.gms', io_options=io_options)
