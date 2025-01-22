#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 01:12:00 2025

@author: hussein.sharadga
"""

import gurobipy as gp
from gurobipy import GRB


# Formulation 3
# This is based on https://or.stackexchange.com/questions/11391/which-of-these-formulations-has-the-tightest-linear-relaxation/12820?noredirect=1#comment27464_12820
# How to modify this formulation for 4 steps or more?

# intialize the model
m=gp.Model()

# m.Params.Presolve=0
# Add variables
x=m.addVars(8, vtype=GRB.BINARY)

L=8-1


x0=1

# Add constraints

for i in range(0,L):
    #print (t)
    
    if i==0:  # intial condition
        m.addConstr(x[0]<=x0+x[1])
        m.addConstr(x[0]<=x0+x[2])
        
        
    elif i==L-1:
        m.addConstr(x[i]<=x[i-1]+x[i+1])
        m.addConstr(x[i]<=x[i-1]+1)  # assume x(final+1) = 1
        print('yes', i)
    else:     
        m.addConstr(x[i]<=x[i-1]+x[i+1])
        m.addConstr(x[i]<=x[i-1]+x[i+2])



m.addConstr(x[2]==0)
m.addConstr(x[5]==0)


# set objective function (z)  or (obj) 
obj=gp.quicksum(x[i] for i in range(8))
m.setObjective(-obj)



# run the model
m.optimize()

# Check optimization status
if m.status == GRB.OPTIMAL:
    print('Optimal solution found')
    # Print variable values
    for var in m.getVars():
        print(f'{var.varName}: {var.x}')
    # Print objective value
    print(f'Objective: {m.objVal}')
else:
    print('No solution found')

