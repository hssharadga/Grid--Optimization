

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:20:32 2024

@author: hussein.sharadga
"""


import gurobipy as gp
from gurobipy import GRB


# Formulation 1
# This is based on eq (2): https://yetanothermathprogrammingconsultant.blogspot.com/2018/03/production-scheduling-minimum-up-time.html    

# intialize the model
m=gp.Model()

# m.Params.Presolve=0
# Add variables
x=m.addVars(8, vtype=GRB.BINARY)

L=8-1


x0=1

# Add constraints

for t in range(0,L):
    print (t)
    
    if t==0:  # intial condition
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3))>=3*(x[t]-x0))
    elif t==L-1:
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3-1))>=(3-1)*(x[t]-x[t-1]))
        print('yes')
    else:     
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3))>=3*(x[t]-x[t-1]))




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









# Formulation 2: disaggrated. It is tighter which might converage faster

# intialize the model
m=gp.Model()

# m.Params.Presolve=0
# Add variables
x=m.addVars(8, vtype=GRB.BINARY)

L=8-1


x0=1

# Add constraints

for t in range(0,L):
    print (t)
    
    if t==0:  # intial condition
        for k in range (t,t+3):
            m.addConstr(x[k]>=(x[t]-x0))
    elif t==L-1:
        for k in range (t,t+3-1):
            m.addConstr(x[k]>=(x[t]-x[t-1]))
        #print('yes')
    else:     
        for k in range (t,t+3):
            m.addConstr(x[k] >=(x[t]-x[t-1]))




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







# Formulation 1 minimum down-time

# intialize the model
m=gp.Model()

# m.Params.Presolve=0
# Add variables
x=m.addVars(8, vtype=GRB.BINARY)

L=8-1


x0=1

# Add constraints

for t in range(0,L):
    print (t)
    
    if t==0:  # intial condition
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3))<=3*(x[t]-x0+1))
    elif t==L-1:
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3-1))<=(3-1)*(x[t]-x[t-1]+1))
        print('yes')
    else:     
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3))<=3*(x[t]-x[t-1]+1))




m.addConstr(x[2]==0)
# m.addConstr(x[5]==0)


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




# %%
# Formulation 2 minimum down-time: tighter version

# intialize the model
m=gp.Model()

# m.Params.Presolve=0
# Add variables
x=m.addVars(8, vtype=GRB.BINARY)

L=8-1


x0=1

# Add constraints

for t in range(0,L):
    print (t)
    
    if t==0:  # intial condition
        for k in range (t,t+3):
            m.addConstr(x[k] <=(x[t]-x0+1))
            
    elif t==L-1:
        for k in range (t,t+3-1):
            m.addConstr(x[k]<=(x[t]-x[t-1]+1))
        print('yes')
    else:     
        for k in range (t,t+3):
            m.addConstr(x[k]<=(x[t]-x[t-1]+1))




m.addConstr(x[6]==0)
# m.addConstr(x[5]==0)


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




# Formulation 1: uptime & downtime

# intialize the model
m=gp.Model()

# m.Params.Presolve=0
# Add variables
x=m.addVars(8, vtype=GRB.BINARY)

L=8-1


x0=1

# Add constraints uptime

for t in range(0,L):
    print (t)
    
    if t==0:  # intial condition
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3))>=3*(x[t]-x0))
    elif t==L-1:
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3-1))>=(3-1)*(x[t]-x[t-1]))
        #print('yes')
    else:     
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3))>=3*(x[t]-x[t-1]))


# Add constraints downtime

for t in range(0,L):
    print (t)
    
    if t==0:  # intial condition
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3))<=3*(x[t]-x0+1))
    elif t==L-1:
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3-1))<=(3-1)*(x[t]-x[t-1]+1))
        #print('yes')
    else:     
        m.addConstr(gp.quicksum(x[k] for k in range (t,t+3))<=3*(x[t]-x[t-1]+1))




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





# Formulation 2: uptime & downtime

# intialize the model
m=gp.Model()

# m.Params.Presolve=0
# Add variables
x=m.addVars(8, vtype=GRB.BINARY)

L=8-1


x0=1

# Add constraints uptime
for t in range(0,L):
    print (t)
    
    if t==0:  # intial condition
        for k in range (t,t+3):
            m.addConstr(x[k]>=(x[t]-x0))
    elif t==L-1:
        for k in range (t,t+3-1):
            m.addConstr(x[k]>=(x[t]-x[t-1]))
        #print('yes')
    else:     
        for k in range (t,t+3):
            m.addConstr(x[k] >=(x[t]-x[t-1]))


# Add constraints downtime
for t in range(0,L):
    print (t)
    
    if t==0:  # intial condition
        for k in range (t,t+3):
            m.addConstr(x[k] <=(x[t]-x0+1))
            
    elif t==L-1:
        for k in range (t,t+3-1):
            m.addConstr(x[k]<=(x[t]-x[t-1]+1))
        print('yes')
    else:     
        for k in range (t,t+3):
            m.addConstr(x[k]<=(x[t]-x[t-1]+1))


#m.addConstr(x[2]==0)
m.addConstr(x[6]==0)


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







