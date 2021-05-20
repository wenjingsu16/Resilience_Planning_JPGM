#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:30:08 2020

@author: carriesu
"""
# This version of the code is Monte Carlo to get the results matrix
# where rows are different contingency scenarios of geographically correlated failures and columns are sampled network structure
# This file does not generate gcf events. It uses existing gcf events.
# The unserved gas is deleted in the junctional flow balance constraints

# %%

from __future__ import division
import numpy as np
from numpy import array
from pyomo.environ import *
import json
import math
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon, LineString
import pandas as pd
import random
import multiprocess as mp

# %% Set number of sampled network structure and number of contingency scenarios
# number of columns
# col = 30
col = 30

# row_start = 66
# row_end = 99
row_start = 0
row_end = 99

nu = 100 # numbering of the pool file

# %% create pyomo model

model = ConcreteModel()

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

model.angle = Var(model.NODES)


def uspower_bound(model, node):
    return (0, model.Demand[node])


model.uspower = Var(model.NODES, domain=NonNegativeReals, bounds=uspower_bound, initialize=0)
# model.uspower.pprint()

# %% Define set, param and var for lines
# model.del_component(model.LINES)
c = list(zip(branch[:, 0], branch[:, 1]))
model.LINES = Set(initialize=c)  # ,within=model.NODES*model.NODES)
# model.LINES.pprint()

d = dict(zip(c, branch[:, 3]))
model.Reactance = Param(model.LINES, initialize=d)

e = dict(zip(c, branch[:, 5]))  # when rate A is 9900
# e=dict(zip(c,rateA))
model.Flowlimit = Param(model.LINES, initialize=e)

# The status parameters for transmission lines which will be used for defining contingency
model.St = Param(model.LINES, initialize=dict(zip(c, [1 for i in range(20)])), mutable=True)
# model.St.pprint()

model.lineflow = Var(model.LINES)


def lineflow_rule1(model, i, j):
    return -model.St[i, j] * model.Flowlimit[i, j] <= model.lineflow[i, j]


model.lineflow1 = Constraint(model.LINES, rule=lineflow_rule1)


# model.lineflow1.pprint()

def lineflow_rule2(model, i, j):
    return model.lineflow[i, j] <= model.St[i, j] * model.Flowlimit[i, j]


model.lineflow2 = Constraint(model.LINES, rule=lineflow_rule2)

# %%
# model.del_component(model.NEWLINES)
h = list(zip(newline[:, 0], newline[:, 1]))
model.NEWLINES = Set(initialize=h)  # ,within=model.NODES*model.NODES)
# model.NEWLINES.pprint()

i = dict(zip(h, newline[:, 3]))
model.Reactance_ne = Param(model.NEWLINES, initialize=i)

j = dict(zip(h, newline[:, 5]))
model.Flowlimit_ne = Param(model.NEWLINES, initialize=j)

model.lineflow_ne = Var(model.NEWLINES)

# The binary variable to define whether a line is built or not
model.zt = Param(model.NEWLINES, domain=Binary, initialize=dict(zip(h, [0 for i in range(20)])), mutable=True)


# model.zt.pprint()

# k= dict(zip(h,[7200000]*20))
# model.Pricet = Param(model.NEWLINES,initialize=k)

# line flow is within the bound*binary investment decision variable
def linelimit_ne_rule1(model, i, j):
    return -model.zt[i, j] * model.Flowlimit_ne[i, j] <= model.lineflow_ne[i, j]


model.linelimit_ne1 = Constraint(model.NEWLINES, rule=linelimit_ne_rule1)


# model.linelimit_ne1.pprint()

def linelimit_ne_rule2(model, i, j):
    return model.lineflow_ne[i, j] <= model.zt[i, j] * model.Flowlimit_ne[i, j]


model.linelimit_ne2 = Constraint(model.NEWLINES, rule=linelimit_ne_rule2)
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


def supply_bound(model, generator):
    return (0, model.Maxout[generator])


model.supply = Var(model.GENERATORS, domain=NonNegativeReals, bounds=supply_bound)


# model.supply.pprint()

# %% Define constraints and objective for power system

# Nodal balance, Kirchhoff's current law
def NodeBalance_rule(model, node):
    return sum(model.supply[generator] for generator in model.GENERATORS if (node, generator) in model.GNCONNECT) \
           + sum(model.lineflow[i, node] for i in model.NODES if (i, node) in model.LINES) \
           - model.Demand[node] \
           - sum(model.lineflow[node, j] for j in model.NODES if (node, j) in model.LINES) \
           + sum(model.lineflow_ne[i, node] for i in model.NODES if (i, node) in model.NEWLINES) \
           - sum(model.lineflow_ne[node, j] for j in model.NODES if (node, j) in model.NEWLINES) \
           + model.uspower[node] \
           == 0


model.NodeBalance = Constraint(model.NODES, rule=NodeBalance_rule)
# model.NodeBalance.pprint()

Bigm = 1000000


# Constraints of Ohm's law for lines that exist and did not fail
def OhmsLaw_rule1(model, nodei, nodej):
    m = model.lineflow[nodei, nodej] - (model.angle[nodei] - model.angle[nodej]) / model.Reactance[nodei, nodej]
    return -Bigm * (1 - model.St[nodei, nodej]) <= m


model.OhmsLaw1 = Constraint(model.LINES, rule=OhmsLaw_rule1)


def OhmsLaw_rule2(model, nodei, nodej):
    m = model.lineflow[nodei, nodej] - (model.angle[nodei] - model.angle[nodej]) / model.Reactance[nodei, nodej]
    return m <= Bigm * (1 - model.St[nodei, nodej])


model.OhmsLaw2 = Constraint(model.LINES, rule=OhmsLaw_rule2)


# model.OhmsLaw2.pprint()

# Constraints of Ohm's law for line candidates
# BigM = Param(initialize=1000000)
def OhmsLaw_ne_rule1(model, i, j):
    m = model.lineflow_ne[i, j] - ((model.angle[i] - model.angle[j]) / model.Reactance_ne[i, j])
    return -Bigm * (1 - model.zt[i, j]) <= m


model.OhmsLaw_ne1 = Constraint(model.NEWLINES, rule=OhmsLaw_ne_rule1)


# model.OhmsLaw_ne1.pprint()

def OhmsLaw_ne_rule2(model, i, j):
    m = model.lineflow_ne[i, j] - ((model.angle[i] - model.angle[j]) / model.Reactance_ne[i, j])
    return m <= Bigm * (1 - model.zt[i, j])


model.OhmsLaw_ne2 = Constraint(model.NEWLINES, rule=OhmsLaw_ne_rule2)
# model.OhmsLaw_ne2.pprint()

# %% Construct natural gas system part of the model
# import gas data from json and turn into dictionary

# figure out how to turn dictionary into param. Maybe i can use dict directly.
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

for idx, pipe in belg['pipe'].items():
    lambdaff = pipe['friction_factor']
    D = pipe['diameter']
    L = pipe['length']

    a_sqr = z * (R / m) * T
    A = (math.pi * D ** 2) / 4
    resistance = ((D * A ** 2) / (lambdaff * L * a_sqr)) * (belg['baseP'] ** 2 / belg['baseQ'] ** 2)

    price = L * 5 * 10 ** 3 / 1.6

    pipe.update({'resistance': resistance})
    pipe.update({'price': price})

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
def pressure_sqr_bound(model, junction):
    return (model.Pmin[junction] ** 2, model.Pmax[junction] ** 2)


model.pressure_sqr = Var(model.JUNCTION, bounds=pressure_sqr_bound)
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
def flow_bound(model, connection):
    return (-Max_mass_flow, Max_mass_flow)


model.flow = Var(model.CONNECTION, bounds=flow_bound)
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
model.yp = Var(model.CONNECTION, domain=Binary, initialize=ypdict)
model.yn = Var(model.CONNECTION, domain=Binary, initialize=yndict)
# model.yp.pprint()
# model.yn.pprint()


# Auxiliary relaxation variable
model.lamd = Var(model.CONNECTION, initialize=1)
# model.lamd.pprint()

# %% Define pipe related set, parameter
model.PIPE = Set(within=model.CONNECTION, initialize=belg['pipe'].keys())
# model.PIPE.pprint()

resistance_dict = dict((k, belg['pipe'][k]['resistance']) for k in belg['pipe'].keys())
model.Resistance = Param(model.PIPE, initialize=resistance_dict)
# model.Resistance.pprint()

# Status of PIPE
model.Sp = Param(model.PIPE, initialize=dict(zip(belg['pipe'].keys(), [1] * 24)), mutable=True)
model.Sp.pprint()

# %% New connection Set
model.NE_CONNECTION = Set(initialize=belg['connection'].keys())
model.NE_CONNECTION.pprint()

model.Pdmax_ne = Param(model.NE_CONNECTION, initialize=Pdmax_dict)
# model.Pdmax.pprint()

model.Pdmin_ne = Param(model.NE_CONNECTION, initialize=Pdmin_dict)
# model.Pdmin.pprint()

# Variable mass flow for ne connection
model.flow_ne = Var(model.NE_CONNECTION, bounds=flow_bound)

# Flow direction binary variables
model.yp_ne = Var(model.NE_CONNECTION, domain=Binary, initialize=ypdict)
model.yn_ne = Var(model.NE_CONNECTION, domain=Binary, initialize=yndict)

model.lamd_ne = Var(model.NE_CONNECTION, initialize=1)

# There is not binary variable for connection investment decisions because they are seperate investment decisions

# %% New pipeline candidates
model.NE_PIPE = Set(within=model.NE_CONNECTION, initialize=belg['pipe'].keys())
model.NE_PIPE.pprint()

model.Resistance_ne = Param(model.NE_PIPE, initialize=resistance_dict)

# model.zp = Var(model.NE_PIPE,domain=Binary,initialize=0)
model.zp = Param(model.NE_PIPE, initialize=dict(zip(belg['pipe'].keys(), [0] * 24)), mutable=True)
# model.zp.pprint()


# pricep_dict=dict((k,belg['pipe'][k]['price']) for k in belg['pipe'].keys())
# model.Pricep =Param(model.NE_PIPE,initialize=pricep_dict)
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
model.NE_COMPRESSOR = Set(within=model.NE_CONNECTION, initialize=belg['compressor'].keys())

model.Max_ratio_ne = Param(model.NE_COMPRESSOR, initialize=max_ratio_dict)

model.Min_ratio_ne = Param(model.NE_COMPRESSOR, initialize=min_ratio_dict)

# Binary variable to represent whether compressors are built
model.zc = Param(model.NE_COMPRESSOR, domain=Binary, initialize=dict((k, 0) for k in belg['compressor'].keys()))
# model.zc.pprint()

# compressor_cost = 2*10**6
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
def fl_elec_bound(model, consumer):
    # return (model.Flmin[consumer], model.Flmax[consumer])
    return (0, model.Flmax[consumer])


model.fl_elec = Var(model.CONSUMER, bounds=fl_elec_bound)


# model.fl_elec.pprint()

def usgas_bound(model, consumer):
    return (0, model.Flmax[consumer])


model.usgas = Var(model.CONSUMER, domain=NonNegativeReals, bounds=usgas_bound, initialize=0)


# model.usgas.pprint()

def junction_gas_demand_rule(model, consumer):
    return inequality(model.Flmin[consumer], model.usgas[consumer] + model.fl_elec[consumer], model.Flmax[consumer])


model.junction_gas_demand = Constraint(model.CONSUMER, rule=junction_gas_demand_rule)
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
def fg_extra_bound(model, producer):
    # return (0, model.Fgmax[producer])
    return (0, model.Fgmax[producer])


model.fg_extra = Var(model.PRODUCER, bounds=fg_extra_bound)
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
def junction_mass_flow_balance_rule(model, junction):
    return sum(
        model.fg_extra[producer] for producer in model.PRODUCER if (producer, junction) in model.PRODUCER_JUNCTION) \
           - sum(model.fl_elec[consumer]  for consumer in model.CONSUMER if
                 (consumer, junction) in model.CONSUMER_JUNCTION) \
           + sum(model.flow[connection] for connection in model.CONNECTION if
                 (connection, junction) in model.CONNECTION_JUNCTIONT) \
           - sum(model.flow[connection] for connection in model.CONNECTION if
                 (connection, junction) in model.CONNECTION_JUNCTIONF) \
           + sum(model.flow_ne[ne_connection] for ne_connection in model.NE_CONNECTION if
                 (ne_connection, junction) in model.CONNECTION_JUNCTIONT) \
           - sum(model.flow_ne[ne_connection] for ne_connection in model.NE_CONNECTION if
                 (ne_connection, junction) in model.CONNECTION_JUNCTIONF) \
           == 0


model.junction_mass_flow_balance = Constraint(model.JUNCTION, rule=junction_mass_flow_balance_rule)


# model.junction_mass_flow_balance.pprint()

# %%
# constraint on the flow direction on connection. Equation 3
def flow_direction_choice1_rule(model, compressor):
    return model.yp[compressor] + model.yn[compressor] == 1


model.flow_direction_choice1 = Constraint(model.COMPRESSOR, rule=flow_direction_choice1_rule)


# model.flow_direction_choice1 = Constraint(model.CONNECTION,rule=flow_direction_choice1_rule)


def flow_direction_choice2_rule(model, pipe):
    return model.yp[pipe] + model.yn[pipe] == model.Sp[pipe]


model.flow_direction_choice2 = Constraint(model.PIPE, rule=flow_direction_choice2_rule)


# flow direction for new connection
def flow_direction_choice_ne1_rule(model, necompressor):
    return model.yp_ne[necompressor] + model.yn_ne[necompressor] == model.zc[necompressor]


model.flow_direction_choice_ne1 = Constraint(model.NE_COMPRESSOR, rule=flow_direction_choice_ne1_rule)


# model.flow_direction_choice_ne1.pprint()

def flow_direction_choice_ne2_rule(model, nepipe):
    return model.yp_ne[nepipe] + model.yn_ne[nepipe] == model.zp[nepipe]


model.flow_direction_choice_ne2 = Constraint(model.NE_PIPE, rule=flow_direction_choice_ne2_rule)


# model.flow_direction_choice_ne2.pprint()

def flow_direction_choice2_rule(model, compressor):
    return model.yp_ne[compressor] + model.yn_ne[compressor] == 1


# model.flow_direction_choice1 = Constraint(model.COMPRESSOR,rule=flow_direction_choice1_rule)
# model.flow_direction_choice2 = Constraint(model.NE_CONNECTION,rule=flow_direction_choice2_rule)

# %%
# Constraint on pressure drop across pipes equation. APPLIES TO PIPE. Equation 7
# Pressure drop alighs with flow direction
def pressure_drop_direction1_rule(model, pipe):
    yp = model.yp[pipe]
    yn = model.yn[pipe]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONT)
    return (1 - yp) * model.Pdmin[pipe] <= pi - pj


model.pressure_drop_direction1 = Constraint(model.PIPE, rule=pressure_drop_direction1_rule)


def pressure_drop_direction2_rule(model, pipe):
    yp = model.yp[pipe]
    yn = model.yn[pipe]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONT)
    return pi - pj <= (1 - yn) * model.Pdmax[pipe]


model.pressure_drop_direction2 = Constraint(model.PIPE, rule=pressure_drop_direction2_rule)


# Equation 7 for NEW PIPE
# flow goes from high pressure to low pressure, pressure drop is within the pressure drop limit for a pipe
def pressure_drop_direction_ne1_rule(model, ne_pipe):
    yp_ne = model.yp_ne[ne_pipe]
    yn_ne = model.yn_ne[ne_pipe]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_pipe, i) in model.CONNECTION_JUNCTIONT)
    return (1 - yp_ne) * model.Pdmin_ne[ne_pipe] <= pi - pj


model.pressure_drop_direction_ne1 = Constraint(model.NE_PIPE, rule=pressure_drop_direction_ne1_rule)


# model.pressure_drop_direction_ne1.pprint()

def pressure_drop_direction_ne2_rule(model, ne_pipe):
    yp_ne = model.yp_ne[ne_pipe]
    yn_ne = model.yn_ne[ne_pipe]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_pipe, i) in model.CONNECTION_JUNCTIONT)
    return pi - pj <= (1 - yn_ne) * model.Pdmax_ne[ne_pipe]


model.pressure_drop_direction_ne2 = Constraint(model.NE_PIPE, rule=pressure_drop_direction_ne2_rule)


# model.pressure_drop_direction_ne2.pprint()

# %%
# Constraint that ensure the flow directionality is tied to the sign of flow.
# FOR CONNECTION. Equation 5 and 6
# Connect the flow direction yp yn with flow. flow between 0 and maxflow based on the direction
def compressor_flow_direction1_rule(model, compressor):
    yp = model.yp[compressor]
    mf = Max_mass_flow
    return -(1 - yp) * mf <= model.flow[compressor]


model.compressor_flow_direction1 = Constraint(model.COMPRESSOR, rule=compressor_flow_direction1_rule)
model.compressor_flow_direction1.pprint()


def compressor_flow_direction2_rule(model, compressor):
    yn = model.yn[compressor]
    mf = Max_mass_flow
    return model.flow[compressor] <= (1 - yn) * mf


model.compressor_flow_direction2 = Constraint(model.COMPRESSOR, rule=compressor_flow_direction2_rule)


def pipe_flow_direction1_rule(model, pipe):
    yp = model.yp[pipe]
    mf = Max_mass_flow
    pdmin = model.Pdmin[pipe]
    pdmax = model.Pdmax[pipe]
    w = model.Resistance[pipe]
    f = model.flow[pipe]
    return -(1 - yp) * min(mf, sqrt(w * max(pd_max, abs(pd_min)))) <= f


model.pipe_flow_direction1 = Constraint(model.PIPE, rule=pipe_flow_direction1_rule)
model.pipe_flow_direction1.pprint()


def pipe_flow_direction2_rule(model, pipe):
    yn = model.yn[pipe]
    mf = Max_mass_flow
    pdmin = model.Pdmin[pipe]
    pdmax = model.Pdmax[pipe]
    w = model.Resistance[pipe]
    f = model.flow[pipe]
    return f <= (1 - yn) * min(mf, sqrt(w * max(pd_max, abs(pd_min))))


model.pipe_flow_direction2 = Constraint(model.PIPE, rule=pipe_flow_direction2_rule)
model.pipe_flow_direction2.pprint()


# %%
# Flow directionality is tied to the sign of flow for new connetions
# equation 5 and 6 for new connections
def compressor_flow_direction_ne1_rule(model, ne_compressor):
    yp_ne = model.yp_ne[ne_compressor]
    mf = Max_mass_flow
    return -(1 - yp_ne) * mf <= model.flow_ne[ne_compressor]


model.compressor_flow_direction_ne1 = Constraint(model.NE_COMPRESSOR, rule=compressor_flow_direction_ne1_rule)


# model.compressor_flow_direction_ne1.pprint()

def compressor_flow_direction_ne2_rule(model, ne_compressor):
    yn_ne = model.yn_ne[ne_compressor]
    mf = Max_mass_flow
    return model.flow_ne[ne_compressor] <= (1 - yn_ne) * mf


model.compressor_flow_direction_ne2 = Constraint(model.NE_COMPRESSOR, rule=compressor_flow_direction_ne2_rule)


# model.compressor_flow_direction_ne2.pprint()


def pipe_flow_direction_ne1_rule(model, nepipe):
    yp = model.yp_ne[nepipe]
    mf = Max_mass_flow
    pdmin = model.Pdmin[nepipe]
    pdmax = model.Pdmax[nepipe]
    w = model.Resistance[nepipe]
    f = model.flow_ne[nepipe]
    return -(1 - yp) * min(mf, sqrt(w * max(pd_max, abs(pd_min)))) <= f


model.pipe_flow_direction_ne1 = Constraint(model.NE_PIPE, rule=pipe_flow_direction_ne1_rule)


# model.pipe_flow_direction_ne1.pprint()

def pipe_flow_direction_ne2_rule(model, nepipe):
    yn = model.yn_ne[nepipe]
    mf = Max_mass_flow
    pdmin = model.Pdmin[nepipe]
    pdmax = model.Pdmax[nepipe]
    w = model.Resistance[nepipe]
    f = model.flow_ne[nepipe]
    return f <= (1 - yn) * min(mf, sqrt(w * max(pd_max, abs(pd_min))))


model.pipe_flow_direction_ne2 = Constraint(model.NE_PIPE, rule=pipe_flow_direction_ne2_rule)


# model.pipe_flow_direction_ne2.pprint()


# %%
# Weymouth equation for pipes, define multiple equations within one function
# FOR PIPES WITHIN CONNECTIONS
def weymouth_rule1(model, connection):
    yp = model.yp[connection]
    yn = model.yn[connection]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin[connection]
    pdmax = model.Pdmax[connection]
    return model.lamd[connection] >= pj - pi + pdmin * (yp - yn + 1)


model.weymouth1 = Constraint(model.PIPE, rule=weymouth_rule1)
model.weymouth1.pprint()


def weymouth_rule2(model, connection):
    yp = model.yp[connection]
    yn = model.yn[connection]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin[connection]
    pdmax = model.Pdmax[connection]
    return model.lamd[connection] >= pi - pj + pdmax * (yp - yn - 1)


model.weymouth2 = Constraint(model.PIPE, rule=weymouth_rule2)


# model.weymouth2.pprint()

def weymouth_rule3(model, connection):
    yp = model.yp[connection]
    yn = model.yn[connection]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin[connection]
    pdmax = model.Pdmax[connection]
    return model.lamd[connection] <= pj - pi + pdmax * (yp - yn + 1)


model.weymouth3 = Constraint(model.PIPE, rule=weymouth_rule3)


# model.weymouth3.pprint()

def weymouth_rule4(model, connection):
    yp = model.yp[connection]
    yn = model.yn[connection]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin[connection]
    pdmax = model.Pdmax[connection]
    return model.lamd[connection] <= pi - pj + pdmin * (yp - yn - 1)


model.weymouth4 = Constraint(model.PIPE, rule=weymouth_rule4)


# model.weymouth4.pprint()

def weymouth_rule5(model, connection):
    f = model.flow[connection]
    w = model.Resistance[connection]
    sp = model.Sp[connection]
    return sp * w * model.lamd[connection] >= f ** 2


model.weymouth5 = Constraint(model.PIPE, rule=weymouth_rule5)


# model.weymouth5.pprint()

def weymouth_rule1_ne(model, ne_connection):
    yp_ne = model.yp_ne[ne_connection]
    yn_ne = model.yn_ne[ne_connection]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin_ne[ne_connection]
    pdmax = model.Pdmax_ne[ne_connection]
    return model.lamd_ne[ne_connection] >= pj - pi + pdmin * (yp_ne - yn_ne + 1)


model.weymouth1_ne = Constraint(model.NE_PIPE, rule=weymouth_rule1_ne)
model.weymouth1_ne.pprint()


def weymouth_rule2_ne(model, ne_connection):
    yp_ne = model.yp_ne[ne_connection]
    yn_ne = model.yn_ne[ne_connection]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin_ne[ne_connection]
    pdmax = model.Pdmax_ne[ne_connection]
    return model.lamd_ne[ne_connection] >= pi - pj + pdmax * (yp_ne - yn_ne - 1)


model.weymouth2_ne = Constraint(model.NE_PIPE, rule=weymouth_rule2_ne)
model.weymouth2_ne.pprint()


def weymouth_rule3_ne(model, ne_connection):
    yp_ne = model.yp_ne[ne_connection]
    yn_ne = model.yn_ne[ne_connection]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin_ne[ne_connection]
    pdmax = model.Pdmax_ne[ne_connection]
    return model.lamd_ne[ne_connection] <= pj - pi + pdmax * (yp_ne - yn_ne + 1)


model.weymouth3_ne = Constraint(model.NE_PIPE, rule=weymouth_rule3_ne)
model.weymouth3_ne.pprint()


def weymouth_rule4_ne(model, ne_connection):
    yp_ne = model.yp_ne[ne_connection]
    yn_ne = model.yn_ne[ne_connection]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    pdmin = model.Pdmin_ne[ne_connection]
    pdmax = model.Pdmax_ne[ne_connection]
    return model.lamd_ne[ne_connection] <= pi - pj + pdmin * (yp_ne - yn_ne - 1)


model.weymouth4_ne = Constraint(model.NE_PIPE, rule=weymouth_rule4_ne)
model.weymouth4_ne.pprint()


def weymouth_rule5_ne(model, ne_connection):
    f = model.flow_ne[ne_connection]
    w = model.Resistance[ne_connection]
    zp = model.zp[ne_connection]
    return w * zp * model.lamd_ne[ne_connection] >= f ** 2


model.weymouth5_ne = Constraint(model.NE_PIPE, rule=weymouth_rule5_ne)


# model.weymouth5_ne.pprint()


# %%
# Compression falls within the compression ratio limits of the compressors
# Equation 8 to 11
def compression_rule1(model, connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection]
    yn = model.yn[connection]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]

    return pj - max_ratio ** 2 * pi <= (1 - yp) * (j_pmax ** 2)


model.compression1 = Constraint(model.COMPRESSOR, rule=compression_rule1)


# model.compression1.pprint()

def compression_rule2(model, connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection]
    yn = model.yn[connection]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return min_ratio ** 2 * pi - pj <= (1 - yp) * (i_pmax ** 2)


model.compression2 = Constraint(model.COMPRESSOR, rule=compression_rule2)


# model.compression2.pprint()

def compression_rule3(model, connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection]
    yn = model.yn[connection]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return pi - pj <= (1 - yn) * (i_pmax ** 2)


model.compression3 = Constraint(model.COMPRESSOR, rule=compression_rule3)


# model.compression3.pprint()

def compression_rule4(model, connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection]
    yn = model.yn[connection]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return pj - pi <= (1 - yn) * (j_pmax ** 2)


model.compression4 = Constraint(model.COMPRESSOR, rule=compression_rule4)


# model.compression4.pprint()

# Compressor ratio for new compressors (directed,compression from i to j, no compression from j to i)
# equation 8-11
def compression_ne_rule1(model, ne_connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp_ne[ne_connection]
    yn = model.yn_ne[ne_connection]
    max_ratio = model.Max_ratio_ne[ne_connection]
    min_ratio = model.Min_ratio_ne[ne_connection]
    zc = model.zc[ne_connection]

    return pj - max_ratio ** 2 * pi <= (2 - yp - zc) * (j_pmax ** 2)


model.compression_ne1 = Constraint(model.NE_COMPRESSOR, rule=compression_ne_rule1)


# model.compression_ne1.pprint()

def compression_ne_rule2(model, ne_connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp_ne[ne_connection]
    yn = model.yn_ne[ne_connection]
    max_ratio = model.Max_ratio_ne[ne_connection]
    min_ratio = model.Min_ratio_ne[ne_connection]
    zc = model.zc[ne_connection]

    return min_ratio ** 2 * pi - pj <= (2 - yp - zc) * (i_pmax ** 2)


model.compression_ne2 = Constraint(model.NE_COMPRESSOR, rule=compression_ne_rule2)


# model.compression_ne2.pprint()

def compression_ne_rule3(model, ne_connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp_ne[ne_connection]
    yn = model.yn_ne[ne_connection]
    max_ratio = model.Max_ratio_ne[ne_connection]
    min_ratio = model.Min_ratio_ne[ne_connection]
    zc = model.zc[ne_connection]

    return pi - pj <= (2 - yn - zc) * (i_pmax ** 2)


model.compression_ne3 = Constraint(model.NE_COMPRESSOR, rule=compression_ne_rule3)


# model.compression_ne3.pprint()

def compression_ne_rule4(model, ne_connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (ne_connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp_ne[ne_connection]
    yn = model.yn_ne[ne_connection]
    max_ratio = model.Max_ratio_ne[ne_connection]
    min_ratio = model.Min_ratio_ne[ne_connection]
    zc = model.zc[ne_connection]

    return pj - pi <= (2 - yn - zc) * (j_pmax ** 2)


model.compression_ne4 = Constraint(model.NE_COMPRESSOR, rule=compression_ne_rule4)


# model.compression_ne4.pprint()

# %%New compressor flow limit, flow is 0 when zc is 0
def compressor_flow_ne1_rule(model, necompressor):
    return model.flow_ne[necompressor] <= model.zc[necompressor] * Max_mass_flow


model.compressor_flow_ne1 = Constraint(model.NE_COMPRESSOR, rule=compressor_flow_ne1_rule)


# model.compressor_flow_ne.pprint()

def compressor_flow_ne2_rule(model, necompressor):
    return model.flow_ne[necompressor] >= -model.zc[necompressor] * Max_mass_flow


model.compressor_flow_ne2 = Constraint(model.NE_COMPRESSOR, rule=compressor_flow_ne2_rule)


def pipe_flow_ne1_rule(model, nepipe):
    return model.flow_ne[nepipe] <= model.zp[nepipe] * Max_mass_flow


model.pipe_flow_ne1 = Constraint(model.NE_PIPE, rule=pipe_flow_ne1_rule)


def pipe_flow_ne2_rule(model, nepipe):
    return model.flow_ne[nepipe] >= -model.zp[nepipe] * Max_mass_flow


model.pipe_flow_ne2 = Constraint(model.NE_PIPE, rule=pipe_flow_ne2_rule)

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
model.GASNODE.pprint()

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
def heatrate_rule(model, consumer):
    # d = model.fl_elec[consumer]
    # H2 = sum(model.Heatrate_linear[generator] for generator in model.GENERATORS if (generator,consumer) in model.GENERATOR_CONSUMER)
    # pg = (model.supply[generator]  for generator in model.GENERATORS if (generator,consumer) in model.GENERATOR_CONSUMER)
    return model.fl_elec[consumer] == 0.167 * energy_factor * sum(
        model.Heatrate_linear[generator] * mvaBase * model.supply[generator] for generator in model.GENERATORS if
        (generator, consumer) in model.GENERATOR_CONSUMER)


model.heatrate = Constraint(model.GASNODE, rule=heatrate_rule)
# model.heatrate.pprint()

# %%Add unserved electricity and natural gas as variables

# power_penalty = 10 ** 6
# gas_penalty = 220 * 10 ** 6
penalty = 10 ** 6  # Penalty cost for 1 MWh of energy not served

# %%Objective of the model
# gascost1 = 86400*86400*(5.917112528134105e-8)*(belg['baseQ']**2)
# print('gas cost quadratic coefficient',gascost1)
# 0.01822660578829469
costq = 0.0778  # $/m3
gascost =  costq
print('gas cost linear coefficient', gascost)

baseQ = belg['baseQ']


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
    return ((24 * sum(model.supply[generator] * model.Cost1[generator] \
                      + model.Cost2[generator] * (model.supply[generator]) ** 2 for generator in model.NONGASGEN)) \
            + sum( 86400 * model.fl_elec[consumer] / standard_density * baseQ * gascost for consumer in model.CONSUMER if
                  consumer in model.GASNODE)) \
           + penalty * (sum(model.uspower[node] for node in model.NODES) * mvaBase * 24 \
                        + 86400 * 2.78 * 10 ** (-10) * sum(
                model.usgas[consumer] for consumer in model.CONSUMER) * baseQ / (standard_density * heat_value))


model.Obj = Objective(rule=Obj_rule, sense=minimize)
model.Obj.pprint()

# Create a variable to show the total cost since the objective value was not accessible
model.totalcost = Var()

def total_cost_rule(model):
    return ((24 * sum(model.supply[generator] * model.Cost1[generator] \
                      + model.Cost2[generator] * (model.supply[generator]) ** 2 for generator in
                      model.NONGASGEN)) \
            + sum( 86400 *
                model.fl_elec[consumer] / standard_density * baseQ * gascost for consumer in model.CONSUMER if
                consumer in model.GASNODE)) \
           + penalty * (sum(model.uspower[node] for node in model.NODES) * mvaBase * 24 \
                        + 86400 * 2.78 * 10 ** (-10) * sum(
                model.usgas[consumer] for consumer in model.CONSUMER) * baseQ / (standard_density * heat_value)) == model.totalcost


model.totalcost_cons = Constraint(rule=total_cost_rule)

# Create a variable to document the operation cost of power and gas system
model.opcost = Var()


def opcost_rule(model):
    return ((24 * sum(model.supply[generator] * model.Cost1[generator] \
                      + model.Cost2[generator] * (model.supply[generator]) ** 2 for generator in
                      model.NONGASGEN)) \
            + sum( 86400 *
                model.fl_elec[consumer] / standard_density * baseQ * gascost for consumer in model.CONSUMER if
                consumer in model.GASNODE)) == model.opcost


model.opcost_cons = Constraint(rule=opcost_rule)

# create a variable to document the total unserved energy in MWh for 1 day
model.usenergy = Var()

def usenergy_rule(model):
    return (sum(model.uspower[node] for node in model.NODES) * mvaBase * 24 \
            + 86400 * 2.78 * 10 ** (-10) * sum(
                model.usgas[consumer] for consumer in model.CONSUMER) * baseQ / (standard_density * heat_value)) == model.usenergy

model.usenergy_cons = Constraint(rule=usenergy_rule)

# base p, junction pressure normalization
# base q, volume flow normalization

# %% Additional constraints
# Set the reference bus
model.angle[1].fix(0)

# %%create power transmission lines/branch list

branch_list = list(zip(branch[:, 0], branch[:, 1]))

# %%pipeline dictionary
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


# %% Read in GCF events

row_count = row_start

event_df = pd.read_csv('events_rf.csv', index_col=0)  # Read in the matrix of 100 random failures

contingency_list = []

while row_count >= row_start and row_count <= row_end:

    fc_list = eval(event_df.iloc[row_count, 0])
    contingency_list.append(fc_list)

    row_count += 1

print(contingency_list)

# %%Generate a list of list of network additions to be used as columns
# import random
tadd = []
padd = []

# #The number of columsn matches the number in pool function, set at 30 for now
# for i in range(col):
#     ttemp=[]
#     ptemp=[]

#     for j in range(20):
#         if random.random()<=1/20:
#             ttemp.append(1)
#         else:
#             ttemp.append(0)
#     tadd.append(ttemp)

#     for k in range(24):
#         if random.random()<=1/24:
#             ptemp.append(1)
#         else:
#             ptemp.append(0)
#     padd.append(ptemp)       

# print('tadd: ',tadd)
# print(('padd: ',padd))

# with open('tadd.txt','w') as f:
#     f.writelines(str(tadd))

# with open('padd.txt','w') as f:
#     f.writelines(str(padd))

file = open('tadd.txt', 'r')
tadd = file.read()
tadd = eval(tadd)

file = open('padd.txt', 'r')
padd = file.read()
padd = eval(padd)

# %% Run Monte Carlo


# ct=0 #counting the number of runs for columns, also use to extract the network additions

def opt_func(x):

    return_list = []

    ne_list = []  # a list to document what new candidates are being built for this sampled network

    # Status parameters for existing lines are determined by event location, set as 1 before
    # every run, changed into 0 if the existing line is impacted

    # Status parameter for new lines are determined by Bernoulli distribution
    # for transmission lines, average is 1, p=1/20=0.05
    # for pipelines, average is 1, p=1/24

    ttemp = tadd[x] #Ttemp is the sample power network structure out of the 30 network structures, tadd is a list of 30 lists
    ptemp = padd[x]

    j = 0

    for i in branch_list:
        model.zt[i] = ttemp[j]
        if ttemp[j] == 1: ne_list.append(i)
        j += 1

    k = 0

    for i in belg['pipe'].keys():
        model.zp[i] = ptemp[k]
        if ptemp[k] == 1: ne_list.append(i)
        k += 1

    return_list.append(ne_list)

    # dict for total cost, total unserved energy, and results for all contingency scenarios for one sample network structure

    cost_dict = {}
    usenergy_dict = {}
    opcost_dict = {}
    df_dict = {}
    ne_dict = {} # document the network additions for that run

    # 20 transmission lines and 24 gas pipelines that may fail
    # Change the status parameter from 1 to 0 for failed components

    event_number = row_start
    for event in contingency_list:

        # Beforing solving contingency,setting the parameter value back to 1 first
        for i in belg['pipe'].keys():
            model.Sp[i] = 1
        # model.Sp.pprint()

        for i in branch_list:
            model.St[i] = 1
        # model.St.pprint()

        name_list = []  # document the name of transmission lines and pipelines that broke down

        lst = event
        for j in lst:
            if j >= 1 and j <= 20:  # 1-20 is transmission lines
                line = (branch[j - 1, 0], branch[j - 1, 1])
                model.St[line] = 0
                name_list.append(line)
            else:  # 21 to 44 is gas pipelines
                pipe = pipe_dict[j - 20]
                model.Sp[pipe] = 0
                name_list.append(pipe)

        # model.St.pprint()
        # model.Sp.pprint()
        print('event location:', event)
        print('failed component:', lst)
        print('network addition:', ne_list)

        solver = SolverFactory('cplex')
        results = solver.solve(model, tee=True, timelimit=None)

        # document df
        # create the dataframe that documents results of each run (each scenario and each network structure)
        result_list = []
        result_list.append(lst)
        result_list.append(name_list)
        result_list.append(model.totalcost.value)

        total_usgas = sum(model.usgas.extract_values().values())
        total_uspower = sum(model.uspower.extract_values().values())
        result_list.append(total_usgas)
        result_list.append(total_uspower)

        result_list.append(format(results.solver.status))
        result_list.append(format(results.solver.termination_condition))

        net = []  # list of new transmission lines being built
        for idx, component in model.zt.extract_values().items():
            if component > 0: net.append(idx)
        result_list.append(net)

        nep = []  # list of new transmission lines being built
        for idx, component in model.zp.extract_values().items():
            if component > 0: nep.append(idx)
        result_list.append(nep)

        result_list.append(ne_list)

        # print(result_list)
        df_dict.update({event_number: result_list})

        # append the total cost and total unserved energy to a list (which will become series)
        cost_dict.update({event_number: model.totalcost.value})

        usenergy_dict.update({event_number: model.usenergy.value})

        opcost_dict.update({event_number: model.opcost.value})

        ne_dict.update({event_number:ne_list})

        event_number += 1

    return_list.append(cost_dict)
    return_list.append(usenergy_dict)
    return_list.append(opcost_dict)
    return_list.append(ne_dict)
    return_list.append(df_dict)


    # print(return_list)

    return return_list

# %% multiprocess to split the jobs

if __name__ == '__main__':
    pool = mp.Pool(processes=20)
    result = pool.map(opt_func, range(col))

    pool.close()
    pool.join()


text_file = open('rf_pool%d.txt' % nu, 'w')
text_file.writelines(str(result))
text_file.close()

result_df = pd.DataFrame(result)
result_df.to_csv('rf_pool%d.csv' % nu)

print('results---------------------------')
print(result)
