#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 00:54:48 2025

@author: hussein.sharadga
"""

'''
That might be useful to reduce the size of the MIP problem by not introducing all constraints upfront.
'''


from gurobipy import Model, GRB
import math



# Traveling Salesperson Problem (TSP)
# %%  Formulation 1: Subtour Elimination Using Explicit Constraints


# Example coordinates for cities
coordinates = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 5)]
n = len(coordinates)

# Compute Euclidean distances between cities
c = [[math.dist(coordinates[i], coordinates[j]) for j in range(n)] for i in range(n)]

# Create the model
model = Model("TSP")

# Add variables
x = model.addVars(n, n, vtype=GRB.BINARY, name="x")  # Binary decision variables
u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u", lb=0, ub=n-1)  # Order variables

# Set objective: Minimize travel cost
model.setObjective(sum(c[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

# Add constraints: Each city must be entered and exited exactly once
model.addConstrs((x.sum(i, '*') == 1 for i in range(n)), "Enter")
model.addConstrs((x.sum('*', j) == 1 for j in range(n)), "Exit")

for i in range(n):
    model.addConstr(x[i, i] == 0, "same")

# Add MTZ subtour elimination constraints
for i in range(1, n): # index start at 1 so u[0]=0  and start at 0 vertex,
                      # the next u takes 0 and then the next u takes 1 and
                      # the next takes 2, and so.
                      # i.e., u takes the value of zero twice.
    for j in range(1, n):
        if i != j:
            model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, f"SubtourElim_{i}_{j}")

# Solve the model
model.optimize()

# Display the solution
if model.status == GRB.OPTIMAL:
    solution = model.getAttr('X', x)
    print("Optimal tour: variable added")
    for i in range(n):
        for j in range(n):
            if solution[i, j] > 0.5:
                print(f"Travel from city {i} to city {j}")
                
                
 
# %% Formulation 2: Subtour Elimination Using  Callback
                
from gurobipy import Model, GRB
import math

# Example coordinates for cities
coordinates = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 5)]
n = len(coordinates)

# Compute Euclidean distances between cities
c = [[math.dist(coordinates[i], coordinates[j]) for j in range(n)] for i in range(n)]

# Create the model
model = Model("TSP")

# Suppress Gurobi output
model.setParam("OutputFlag", 0)

# Add variables
x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

# Set objective: Minimize travel cost
model.setObjective(sum(c[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

# Add constraints: Each city must be entered and exited exactly once
model.addConstrs((x.sum(i, '*') == 1 for i in range(n)), "Enter")
model.addConstrs((x.sum('*', j) == 1 for j in range(n)), "Exit")


# Function to find subtours
def find_subtours(vals):
    visited = [False] * n
    tours = []

    def visit(node, tour):
        for j in range(n):
            if not visited[j] and vals[node, j] > 0.5:
                visited[j] = True
                tour.append(j)
                visit(j, tour)

    for i in range(n):
        if not visited[i]:
            tour = [i]
            visited[i] = True
            visit(i, tour)
            tours.append(tour)

    return tours


# Callback function for subtour elimination
def subtour_elim(model, where):
    if where == GRB.Callback.MIPSOL:
        #print('here we go!',GRB.Callback.MIPSOL)
        #print(where == GRB.Callback.MIPSOL)
        vals = model.cbGetSolution(x)
        tours = find_subtours(vals)
        #print('here we go:',tours)
        for tour in tours:
            if len(tour) < n:   # If the trip does not pass every node,
                                # add a constraint to make the solution infeasible.
                model.cbLazy(sum(x[i, j] for i in tour for j in tour) <= len(tour) - 1)






# Set parameters and solve the model
model.Params.LazyConstraints = 1
model.optimize(subtour_elim)

# Display the solution
if model.status == GRB.OPTIMAL:
    solution = model.getAttr('X', x)
    print("Optimal tour: Callback")
    for i in range(n):
        for j in range(n):
            if solution[i, j] > 0.5:
                print(f"Travel from city {i} to city {j}")       
       
                
