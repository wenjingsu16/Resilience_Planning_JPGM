#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:30:08 2020

@author: carriesu
"""
# This version of code is to run 100 random failure events and get information about
# where the rows are contingency scenarios and columns are each gas or electricity node
# The entry is unserved energy

# %%
from __future__ import division

from pandas import Series
from pandas.core.arrays import ExtensionArray
from pyomo.environ import *
from numpy import array
import json
import math
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Point, Polygon, LineString
import pandas as pd

model = ConcreteModel()

# %% Construct the IEEE 14 bus power system model
# import data of 14 bus
from busdata import *

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
c = list(zip(branch[:, 0], branch[:, 1]))
model.LINES = Set(initialize=c, within=model.NODES * model.NODES)
# model.LINES.pprint()

d = dict(zip(c, branch[:, 3]))
model.Reactance = Param(model.LINES, initialize=d)

e = dict(zip(c, rateA))
model.Flowlimit = Param(model.LINES, initialize=e)

# The status parameters for transmission lines which will be used for defining contingency
model.St = Param(model.LINES, initialize=dict(zip(c, [1 for i in range(20)])), mutable=True)
# model.St.pprint()

model.lineflow = Var(model.LINES)


def lineflow_rule1(model, i, j):
    return -model.St[i, j] * model.Flowlimit[i, j] <= model.lineflow[i, j]


model.lineflow1 = Constraint(model.LINES, rule=lineflow_rule1)


# model.lineflow.pprint()

def lineflow_rule2(model, i, j):
    return model.lineflow[i, j] <= model.St[i, j] * model.Flowlimit[i, j]


model.lineflow2 = Constraint(model.LINES, rule=lineflow_rule2)

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
           + model.uspower[node] \
           == 0


model.NodeBalance = Constraint(model.NODES, rule=NodeBalance_rule)

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

# %% Construct natural gas system part of the model
# import gas data from json and turn into dictionary

# figure out how to turn dictionary into param. Maybe i can use dict directly.
belg = {}
with open('belgian.json', 'r') as f:
    belg = json.load(f)

print(belg.keys())

# %% Read the gas data

# NE data might also needs to be added if new expansion set is not empty

# Compute the maximum volume for gas modeL
Max_flow = 0
for idx, producer in belg['producer'].items():
    if producer['qgmax'] > 0:
        Max_flow += producer['qgmax']

# Compute the max mass flow in the gas moel
Max_mass_flow = Max_flow * belg['standard_density']

# Ensures that status exists as a field in connections
lst = ['pipe', 'compressor', 'short_pipe', 'resistor', 'valve', 'control_valve', 'ne_pipe', 'ne_compressor', 'junction',
       'consumer', 'producer']
for key in belg.keys():
    for k in lst:
        for idx, component in belg[k].items():
            if 'status' not in component:
                component.update({'status': 1})

# The field yp and yn are used for directionality. If directed in included, we add yp and yn
for key in belg.keys():
    for k in lst:
        for idx, component in belg[k].items():
            if 'directed' in component:
                if component['directed'] == 1:
                    component.update({'yp': 1, 'yn': 0})
                if component['directed'] == -1:
                    component.update({'yp': 0, 'yn': 1})

# Ensure that consumer priority exists as a field in loads
for idx, component in belg['consumer'].items():
    if 'priority' not in component:
        component.update({'priority': 1})

# Add degree information on junction? No parallel connection in the belgian network
if 'parallel_connections' in belg:
    print('There is parallel connections and ADD DEGREE')
if 'all_parallel_connections' in belg:
    print('There is all parallel connection and CHANGE DEGREE')

if belg['multinetwork'] is False:
    print('There is no multi-network')

fromlst = []
tolst = []
# Add the bound for minimum and maximum pressure for the pipes/connections

for idx, connection in belg['connection'].items():
    i_idx = connection['f_junction']
    j_idx = connection['t_junction']

    fromlst.append(i_idx)
    tolst.append(j_idx)

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
    return (-Max_flow, Max_flow)


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
# model.yn = Param(model.CONNECTION,domain=Binary,default=0,initialize=yndict)
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
# model.Sp.pprint()

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
    return (0, model.Fgmax[producer])


model.fg_extra = Var(model.PRODUCER, bounds=fg_extra_bound)
# model.fg_extra.pprint()

# model.gas_waste= Var(model.PRODUCER,bounds=fg_extra_bound)
# def junction_gas_production_rule(model,producer):
#    return model.Fgmin[producer]<= model.gas_waste[producer]+model.fg_extra[producer]<= model.Fgmax[producer]
# model.junction_gas_production = Constraint(model.PRODUCER,rule=junction_gas_production_rule)
# model.junction_gas_production.pprint()

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

# Connections for each junction
# coonnection_from
# Model.CONNECTION_FROM=Set(model.JUNCTION,initialize)


# %% Set, param and variable associate with new connnection, new pipe and new compressors
try:
    model.NE_CONNECTION = Set(initialize=belg['new_connection'].keys())
    # model.NE_CONNECTION.pprint()

    # variable associated with mass flow in expansion planning
    model.flow_ne = Var(model.NE_CONNECTION, bounds=(0 - Max_mass_flow, Max_mass_flow))
    # model.flow_ne.pprint()

    # binary variable associated with the direction of flow on the connection
    model.yp_ne = Var(model.NE_CONNECTION, domain=Binary)
    model.yn_ne = Var(model.NE_CONNECTION, domain=Binary)
    # model.yp_ne.pprint()
    # model.yn_ne.pprint()
except:
    print('There is no new connection being defined based on the data set')

# New pipe
try:
    model.NE_PIPE = Set(initialize=belg['ne_pipe'].keys())
    # model.NE_PIPE.pprint()

    # binary variable associated with building pipes
    model.zp = Var(model.NE_PIPE, domain=Binary)
    # model.zp.pprint()
except:
    print('New pipe set and building deicison are not being defined')

# Vavle and its operation
try:
    model.VALVE = Set(initialize=belg['valve'].keys())
    # model.VALVE.pprint()

    # binary variable associated with operating valves
    model.v = Var(model.VALVE, domain=Binary)
    # model.v.pprint()
except:
    print('Valve set and operating decision are not being defined')

# new compressor
try:
    model.NE_COMPRESSOR = Set(initialize=belg['ne_compressor'].keys())
    # model.NE_COMPRESSOR.pprint()

    model.zc = Var(model.NE_COMPRESSOR, domain=Binary)
    # model.zc.pprint()
except:
    print('New compressor set is not being defined')


# %%Define constraints that does not involve new expansion or load shedding

# mass flow balance equation for junctions where demand and production is fixed. Equation 2
def junction_mass_flow_balance_rule(model, junction):
    return sum(
        model.fg_extra[producer] for producer in model.PRODUCER if (producer, junction) in model.PRODUCER_JUNCTION) \
           - sum(model.fl_elec[consumer] for consumer in model.CONSUMER if
                 (consumer, junction) in model.CONSUMER_JUNCTION) \
           + sum(model.flow[connection] for connection in model.CONNECTION if
                 (connection, junction) in model.CONNECTION_JUNCTIONT) \
           - sum(model.flow[connection] for connection in model.CONNECTION if
                 (connection, junction) in model.CONNECTION_JUNCTIONF) \
           == 0


model.junction_mass_flow_balance = Constraint(model.JUNCTION, rule=junction_mass_flow_balance_rule)


# model.junction_mass_flow_balance.pprint()

# constraint on the flow direction on connection. Equation 3
def flow_direction_choice_rule(model, connection):
    return model.yp[connection] + model.yn[connection] == 1


model.flow_direction_choice = Constraint(model.COMPRESSOR, rule=flow_direction_choice_rule)


def flow_direction_rule2(model, connection):
    return model.yp[connection] + model.yn[connection] == model.Sp[connection]


model.flow_direction_choice2 = Constraint(model.PIPE, rule=flow_direction_rule2)


# Constraint on pressure drop across pipes equation. APPLIES TO CONNECTION. Equation 7
# Pressure drop alighs with flow direction
def on_off_pressure_drop1_rule(model, pipe):
    yp = model.yp[pipe]
    yn = model.yn[pipe]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONT)
    return (1 - yp) * model.Pdmin[pipe] <= pi - pj


model.on_off_pressure_drop1 = Constraint(model.PIPE, rule=on_off_pressure_drop1_rule)


# model.on_off_pressure_drop1.pprint()

def on_off_pressure_drop2_rule(model, pipe):
    yp = model.yp[pipe]
    yn = model.yn[pipe]
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (pipe, i) in model.CONNECTION_JUNCTIONT)
    return pi - pj <= (1 - yn) * model.Pdmax[pipe]


model.on_off_pressure_drop2 = Constraint(model.PIPE, rule=on_off_pressure_drop2_rule)


# model.on_off_pressure_drop2.pprint()

# Constraint that ensure the flow directionality is tied to the sign of flow.
# FOR CONNECTION. Equation 5 and 6
# Connect the flow direction yp yn with flow. flow between 0 and maxflow based on the direction
def on_off_pipe_flow_direction1_rule(model, pipe):
    yp = model.yp[pipe]
    mf = Max_mass_flow
    pdmin = model.Pdmin[pipe]
    pdmax = model.Pdmax[pipe]
    w = model.Resistance[pipe]
    f = model.flow[pipe]
    return -(1 - yp) * min(mf, sqrt(w * max(pd_max, abs(pd_min)))) <= model.flow[pipe]


model.on_off_pipe_flow_direction1 = Constraint(model.PIPE, rule=on_off_pipe_flow_direction1_rule)


# model.on_off_connection_flow_direction1.pprint()

def on_off_pipe_flow_direction2_rule(model, pipe):
    yn = model.yn[pipe]
    mf = Max_mass_flow
    pdmin = model.Pdmin[pipe]
    pdmax = model.Pdmax[pipe]
    w = model.Resistance[pipe]
    f = model.flow[pipe]
    return model.flow[pipe] <= (1 - yn) * min(mf, sqrt(w * max(pd_max, abs(pd_min))))


model.on_off_connection_flow_direction2 = Constraint(model.PIPE, rule=on_off_pipe_flow_direction2_rule)


# model.on_off_connection_flow_direction2.pprint()

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


# model.weymouth1.pprint()

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


# Compression falls within the compression ratio limits of the compressors
# Equation 8 to 11
def compressor_ratio_rule1(model, connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection]
    yn = model.yn[connection]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]

    return pj - max_ratio ** 2 * pi <= (1 - yp) * (j_pmax ** 2)


model.compressor_ratio1 = Constraint(model.COMPRESSOR, rule=compressor_ratio_rule1)


# model.compressor_ratio1.pprint()

def compressor_ratio_rule2(model, connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection]
    yn = model.yn[connection]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return min_ratio ** 2 * pi - pj <= (1 - yp) * (i_pmax ** 2)


model.compressor_ratio2 = Constraint(model.COMPRESSOR, rule=compressor_ratio_rule2)


# model.compressor_ratio2.pprint()

def compressor_ratio_rule3(model, connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection]
    yn = model.yn[connection]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return pi - pj <= (1 - yn) * (i_pmax ** 2)


model.compressor_ratio3 = Constraint(model.COMPRESSOR, rule=compressor_ratio_rule3)


# model.compressor_ratio3.pprint()

def compressor_ratio_rule4(model, connection):
    pi = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    pj = sum(model.pressure_sqr[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    j_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONF)
    i_pmax = sum(model.Pmax[i] for i in model.JUNCTION if (connection, i) in model.CONNECTION_JUNCTIONT)
    yp = model.yp[connection]
    yn = model.yn[connection]
    max_ratio = model.Max_ratio[connection]
    min_ratio = model.Min_ratio[connection]
    return pj - pi <= (1 - yn) * (j_pmax ** 2)


model.compressor_ratio4 = Constraint(model.COMPRESSOR, rule=compressor_ratio_rule4)
# model.compressor_ratio4.pprint()

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
def heatrate_rule(model, consumer):
    # d = model.fl_elec[consumer]
    # H2 = sum(model.Heatrate_linear[generator] for generator in model.GENERATORS if (generator,consumer) in model.GENERATOR_CONSUMER)
    # pg = (model.supply[generator]  for generator in model.GENERATORS if (generator,consumer) in model.GENERATOR_CONSUMER)
    return model.fl_elec[consumer] == 0.167 * energy_factor * sum(
        model.Heatrate_linear[generator] * mvaBase * model.supply[generator] for generator in model.GENERATORS if
        (generator, consumer) in model.GENERATOR_CONSUMER)


model.heatrate = Constraint(model.GASNODE, rule=heatrate_rule)
# model.heatrate.pprint()

# %% Add unserved electricity and natural gas as variables

# power_penalty = 10 ** 6
# gas_penalty = 220 * 10 ** 6  # Make the unserved energy same price for one unit of energy
penalty = 10 ** 6

# %%Objective of the model
# gascost1 = 86400*86400*(5.917112528134105e-8)*(belg['baseQ']**2)
# print('gas cost quadratic coefficient',gascost1)
# 0.01822660578829469
costq = 0.0778  # $/m3
gascost = 86400 * costq
print('gas cost linear coefficient', gascost)
# base p, junction pressure normalization
# base q, volume flow normalization

# (model.fl_elec[consumer]/standard_density)**2 *gascost1\
model.totalpowercost = Var()


def total_power_cost_rule(model):
    return model.totalpowercost == (24 * sum(model.supply[generator] * model.Cost1[generator] \
                                             + model.Cost2[generator] * (model.supply[generator]) ** 2 for generator in
                                             model.NONGASGEN))


model.total_power_cost = Constraint(rule=total_power_cost_rule)
# model.totalpowercost.pprint()

model.totalgascost = Var()


def total_gas_cost_rule(model):
    return model.totalgascost == sum(
        model.fl_elec[consumer] * gascost / standard_density * belg['baseQ'] for consumer in model.CONSUMER if
        consumer in model.GASNODE)


model.total_gas_cost = Constraint(rule=total_gas_cost_rule)


# model.totalgascost.pprint()


def Obj_rule(model):
    return (24 * sum(model.supply[generator] * model.Cost1[generator] \
                     + model.Cost2[generator] * (model.supply[generator]) ** 2 for generator in model.NONGASGEN) \
            + sum(model.fl_elec[consumer] / standard_density * belg['baseQ'] * gascost for consumer in model.CONSUMER if
                  consumer in model.GASNODE) \
            + 24 * mvaBase * penalty * sum(model.uspower[node] for node in model.NODES) \
            + 86400 * penalty * 2.78 * 10 ** (-10) * belg['baseQ'] * sum(
                model.usgas[consumer] for consumer in model.CONSUMER) / heat_value)


model.Obj = Objective(rule=Obj_rule, sense=minimize)
model.Obj.pprint()

model.totalcost = Var()

def total_cost_rule(model):
    return model.totalcost == \
           (24 * sum(model.supply[generator] * model.Cost1[generator] \
                     + model.Cost2[generator] * (model.supply[generator]) ** 2 for generator in model.NONGASGEN) \
            + sum(model.fl_elec[consumer] / standard_density * belg['baseQ'] * gascost for consumer in model.CONSUMER if
                  consumer in model.GASNODE) \
            + 24 * mvaBase * penalty * sum(model.uspower[node] for node in model.NODES) \
            + 86400 * penalty * 2.78 * 10 ** (-10) * belg['baseQ'] * sum(
                       model.usgas[consumer] for consumer in model.CONSUMER) / heat_value)

model.total_cost = Constraint(rule=total_cost_rule)



# %% Additional constraints
# Set the reference bus
model.angle[1].fix(0)

# %% Read in 100 random failure events
gcf_df = pd.read_csv('events_gcf.csv', index_col=0)

# Create the pipe dictionary
c = 1
pipe_dict = {}
for idx, component in belg['pipe'].items():
    print(c, idx, (component['f_junction'], component['t_junction']))
    pipe_dict[c] = idx
    c += 1

# %% Run Monte Carlo

df1_list = []  # df1 is to document the general information about the 100 runs such as total system and total unserved energy
df2_list = []  # df2 is to document the unserved gas or electricity in each gas or power node

# file =open('Monte_Carlo_Results_mc100.txt','w')
# file.close()

# 20 transmission lines and 24 gas pipelines that may fail
# Change the status parameter from 1 to 0 for failed components during each event

for idx, row in gcf_df.iterrows():

    # Beforing solving contingency scenarios,setting the status parameter back to 1 first
    for i in belg['pipe'].keys():
        model.Sp[i] = 1
    # model.Sp.pprint()

    for i in list(zip(branch[:, 0], branch[:, 1])):
        model.St[i] = 1
    # model.St.pprint()

    # Set the status parameter as 0 for the components in the failed components list
    fc_lst = eval(row[1])  # The list of component failures from csv file
    name_list = []  # Actual components whose status parameters are set as 0
    for j in fc_lst:
        if j >= 1 and j <= 20:  # 1-20 is transmission lines
            line = (branch[j - 1, 0], branch[j - 1, 1])
            model.St[line] = 0
            name_list.append(line)
        else:  # 21 to 44 is gas pipelines
            pipe = pipe_dict[j - 20]
            model.Sp[pipe] = 0
            name_list.append(pipe)

    # Print out the status parameters to verify the failed components
    # model.St.pprint()
    # model.Sp.pprint()

    solver = SolverFactory('cplex')
    results = solver.solve(model, tee=True, timelimit=None)

    # write out the solver output
    # with open('Monte_Carlo_Results_rf100.txt','a') as file:
    #     file.write(str(idx))
    #     file.write('------------------------------')        
    #     file.write(str(results))
    # file.close()
    print('---------------------------------')
    print('Run time', idx)
    print(results)

    result_list = []
    result_list.append(row[0]) # Append event location
    result_list.append(fc_lst) # failed component number list
    result_list.append(name_list) # failed component name list
    result_list.append(model.totalcost.value) # total system cost

    result_list.append(model.totalpowercost.value) # Total power system operation cost
    result_list.append(model.totalgascost.value) # Total gas system cost

    total_usgas = sum(model.usgas.extract_values().values())
    total_uspower = sum(model.uspower.extract_values().values())
    result_list.append(total_usgas) # total unserved gas
    result_list.append(total_uspower) # total unserved electricity

    total_usenergy = 24 * mvaBase * total_uspower + 86400 * 2.78 * 10 ** (-10) * belg['baseQ'] * total_usgas / heat_value
    result_list.append(total_usenergy) # total unserved energy

    nodal_power_generation_list = model.supply.extract_values()
    total_power_generation = sum(model.supply.extract_values().values())
    result_list.append(nodal_power_generation_list) # Electricity generated at each node
    result_list.append(total_power_generation) # Total electricity generated = total electricity demand served

    result_list.append((model.totalpowercost.value)/total_power_generation)

    result_list.append(format(results.solver.status))
    result_list.append(format(results.solver.termination_condition))

    # print(result_list)
    df1_list.append(result_list)

    # usenergy_dict = {**model.usgas.extract_values(), **model.uspower.extract_values()}
    usenergy_dict = {}
    usenergy_dict.update(model.usgas.extract_values())
    usenergy_dict.update(model.uspower.extract_values())
    #    df2=df2.append(usenergy_dict,ignore_index=True)
    df2_list.append(usenergy_dict)

df1 = pd.DataFrame(df1_list, columns=['Event location', 'Input FC list', 'Actual FC list', 'System cost','Total power cost',
                                      'Total gas cost', 'Unserved gas', 'Unserved power', 'Unserved energy','Nodal power generation',
                                      'Total power generation','Electricity price','Solver status', 'Termination condition'])
df1.to_csv('df1_gcf100.csv', encoding='utf-8-sig')

df2 = pd.DataFrame(df2_list)
df2.to_csv('df2_gcf100.csv', encoding='utf-8-sig')


