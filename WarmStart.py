#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 04:08:53 2025

@author: hussein.sharadga
"""

from gurobipy import Model, GRB

# Create a new model
model = Model("WarmStartExample")

# Add variables
x = model.addVar(vtype=GRB.BINARY, name="x")
y = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=10, name="y")
z = model.addVar(vtype=GRB.INTEGER, name="z")

# Add constraints
model.addConstr(x + y + z <= 15, "c1")
model.addConstr(x + 2 * y - z >= 10, "c2")

# Set objective
model.setObjective(x + y + z, GRB.MAXIMIZE)

# Assign warm start values
x.Start = 1
y.Start = 10
z.Start = 4

# Optimize
model.optimize()

# Print the solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution:")
    for v in model.getVars():
        print(f"{v.varName} = {v.x}")
