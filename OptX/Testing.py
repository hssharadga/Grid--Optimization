# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:19:53 2025

@author: hussein.sharadga
"""

import json, copy, sys, time
import numpy as np
#import ipopt
from scipy.sparse import coo_matrix

import gurobipy as gp
from gurobipy import GRB


print('TPEC- Testing')


# Methods
warm_st=0            # defult 0
semi_blacoks=1       # defult 0
compact=0            # defult 0
count=0

ramp_const=0         # defult 1

lazy_const=0         # defult 0
c1=0
tight_const=0       # defult 0

switch1=1            # defult 0    # one time switching
switch2=0            # defult 0
bounds=1             # defult 0




sc=[21, 22, 23, 31, 32, 33]
s=sc[1]
src_data = r'C:\Users\hussein.sharadga\Desktop\OptX\grids\C3E2D1_20230510\D1\C3E2N01576D1\scenario_0'+str(s)+'.json'
  

file = open(src_data)
data = json.load(file)
    
# Numbers of each element
n_b = len(data['network']['bus']) # number of buses
n_s = len(data['network']['shunt'])
n_j = len(data['network']['simple_dispatchable_device']) 
n_t = data['time_series_input']['general']['time_periods']
n_t_ = data['time_series_input']['general']['time_periods']
# n_t_=1
delta_t = [t for t in data['time_series_input']['general']['interval_duration']]
delta_t_ = [t for t in data['time_series_input']['general']['interval_duration']]
t_map = {} # map of timesteps to hr
t0 = 0



# # note: currently only branch slacks is not supported
# bus_slacks = True
# branch_slacks = True
# branch_lims = True
# q_bounds = True
# if n_b > 2000:
#     branch_slacks = False
#     branch_lims = False
# if n_b >= 2000:

#     delta_t = [delta_t[0]]
#     n_t0 = copy.deepcopy(n_t)
#     n_t = 1
#if n_b > 5000:
#    q_bounds = False

#delta_t = [delta_t[0]]
#n_t = 1

t0_ = time.time()

pviol_cost = data["network"]["violation_cost"]["p_bus_vio_cost"]
qviol_cost = data["network"]["violation_cost"]["q_bus_vio_cost"]
sviol_cost = data["network"]["violation_cost"]["s_vio_cost"]

for t in range(len(delta_t)):
    t_map[t0] = t
    t0 += delta_t[t]

# ==== Parameters ===

# Bus voltage - do not vary with time
v_min = []
v_max = []
v0 = []
theta0 = []

# Shunts
gs = []
bs = []
ush_min = []
ush_max = []
ush0 = []

# Devices
pl0 = np.zeros((n_b*n_t,)) # uncontrollable base load
pj0 = np.zeros((n_j,)) # The initial consumption of each device at t=0
qj0 = np.zeros((n_j,)) # The initial consumption of each device at t=0
pj_min = np.zeros((n_j*n_t))
pj_min_all=np.zeros((n_j*n_t_))
pj_max = np.zeros((n_j*n_t))
pj_max_all = np.zeros((n_j*n_t_))



pj_min_ = np.zeros((n_j,n_t))
pj_min_all_ = np.zeros((n_j,n_t_))
pj_max_all_ = np.zeros((n_j,n_t_))
pj_max_=np.zeros((n_j,n_t)) 

qj_min = np.zeros((n_j*n_t))
qj_max = np.zeros((n_j*n_t))
en_min = np.zeros((n_j,))
en_max = np.zeros((n_j,))
pru = np.zeros((n_j,))
prd = np.zeros((n_j,))
prgu_ub = np.zeros((n_j,))
prgd_ub = np.zeros((n_j,))
syn_ub = np.zeros((n_j,))
nsyn_ub = np.zeros((n_j,))
rru_on_ub = np.zeros((n_j,))
rru_off_ub = np.zeros((n_j,))
rrd_on_ub = np.zeros((n_j,))
rrd_off_ub = np.zeros((n_j,))

#pru_su = np.zeros((n_j,))
#prd_sd = np.zeros((n_j,))
# >
prusu=[] 
prdsd=[]



# if n_b<2000:
#     n_tt=n_t_ 
# elif Seq_2!=0: # Seq-Ipopt 
#     n_tt=n_t_
# else: # one-step Ipopt
#     n_tt=1
 

n_tt=n_t_
# if div>1:
#     n_tt=n_t_
# else:
#     n_tt=1
    
    
prgu_cost = np.zeros((n_j*n_tt,))
# prgu_cost_all = np.zeros((n_j*n_tt,))
prgd_cost = np.zeros((n_j*n_tt,))
# prgd_cost_all = np.zeros((n_j*n_tt,))
syn_cost = np.zeros((n_j*n_tt,))
# syn_cost_all = np.zeros((n_j*n_tt,))
nsyn_cost = np.zeros((n_j*n_tt,))
cru_on = np.zeros((n_j*n_tt,))
cru_off = np.zeros((n_j*n_tt,))
crd_on = np.zeros((n_j*n_tt,))
crd_off = np.zeros((n_j*n_tt,))

# zones
azn = {}
n_az = len(data['network']['active_zonal_reserve'])
azr_map = {}
rzr_map = {}
for z in range(n_az):
    azr_map[data['network']['active_zonal_reserve'][z]['uid']] = z
    azn[z] = np.zeros((n_b))

# BUSES
bus_map = {}
for i in range(n_b):
    # Params
    v_min.append(data['network']['bus'][i]['vm_lb'])
    v_max.append(data['network']['bus'][i]['vm_ub'])
    v0.append(data['network']['bus'][i]['initial_status']['vm'])
    theta0.append(data['network']['bus'][i]['initial_status']['va'])
    
    # Maps
    bus_map[data['network']['bus'][i]['uid']] = i
    for uid in data['network']['bus'][i]['active_reserve_uids']:
        azn[azr_map[uid]][i] = 1.0
    #ar_map['uid'] = 
    #rr_map['uid'] = data['network']['bus'][i]['reactive_reserve_uids']

v_min = np.array(v_min*n_t)
v_max = np.array(v_max*n_t)
v0 = np.array(v0*n_t)
theta0 = np.array(theta0*n_t)

# SHUNTS
rows = [] # to build connected to matrix
cols = []
vals = []
rows2 = []
cols2 = []
vals2 = []
shunt_map = {}

for s in range(n_s):
    shunt_map[data['network']['shunt'][s]['uid']] = s
    # Params
    gs.append(data['network']['shunt'][s]['gs'])
    bs.append(data['network']['shunt'][s]['bs'])
    ush_min.append(data['network']['shunt'][s]['step_ub'])
    ush_max.append(data['network']['shunt'][s]['step_lb'])
    ush0.append(data['network']['shunt'][s]['initial_status']['step'])
    
    # Maps
    for t in range(n_t):
        rows += [bus_map[data['network']['shunt'][s]['bus']]+t*n_b]
        cols += [s+t*n_s]

# gs_ = np.array(gs*n_t_)
# bs_ = np.array(bs*n_t_)

gs = np.array(gs*n_t)
bs = np.array(bs*n_t)
ush0 = np.array(ush0*n_t)
ush_min = np.array(ush_min*n_t)
ush_max = np.array(ush_max*n_t)

# Matrix mapping
s_connected = coo_matrix(([1.]*(n_s*n_t), (rows, cols)),shape=(n_b*n_t,n_s*n_t))

j_bus_map = {}
j_map = {}
sign = {} # pos (generation) or negative (consumption)


sign_ = {} # negative (generation) or Postive (consumption)

  
su_cost=[] # startup cost
sd_cost=[]
on_cost=[]
j_on_0 = []
# (56)-(57)
down_time=[] 
up_time=[]
startups_ub={}
# (68)-(69)
Wen_max={}          
Wen_min={}



j_on_fixed = []


en_ub = {}
en_lb = {}

#q_bound_map = {}
rows = [] # for mapping p or q to the q bounds
cols = []
n_qb = 0
n_qc = 0
qc_map = {}
qb_map = {}
beta = []
q0 = []
beta = []
q0max = []
q0min = []
beta_min = []
beta_max = []
for j in range(n_j):
    en_ub[j] = []
    en_lb[j] = []

    j_on_fixed.append(data['network']['simple_dispatchable_device'][j]['initial_status']['on_status'])    
    
    #
   
    j_on_0.append(data['network']['simple_dispatchable_device'][j]['initial_status']['on_status'])
    
    pj0[j] = data['network']['simple_dispatchable_device'][j]['initial_status']['p']
    qj0[j] = data['network']['simple_dispatchable_device'][j]['initial_status']['q']
    if data['network']['simple_dispatchable_device'][j]['device_type'] == 'consumer':
        sign[data['network']['simple_dispatchable_device'][j]['uid']] = -1
        

        sign_[data['network']['simple_dispatchable_device'][j]['uid']] = 1
            
    else:
        sign[data['network']['simple_dispatchable_device'][j]['uid']] = 1
        
        sign_[data['network']['simple_dispatchable_device'][j]['uid']] = -1 
        
    #j_sgn.append(sign[data['network']['simple_dispatchable_device'][j]['uid']])
    j_bus_map[j] = bus_map[data['network']['simple_dispatchable_device'][j]['bus']]
    j_map[data['network']['simple_dispatchable_device'][j]['uid']] = j
    
    if data['network']['simple_dispatchable_device'][j]['q_bound_cap'] == 1:
        beta_min.append(data['network']['simple_dispatchable_device'][j]['beta_ub'])
        beta_max.append(data['network']['simple_dispatchable_device'][j]['beta_lb'])
        q0max.append(data['network']['simple_dispatchable_device'][j]['q_0_ub'])
        q0min.append(data['network']['simple_dispatchable_device'][j]['q_0_lb'])
        qb_map[j] = n_qb
        n_qb += 1
    if data['network']['simple_dispatchable_device'][j]['q_linear_cap'] == 1:
        beta.append(data['network']['simple_dispatchable_device'][j]['beta'])
        q0.append(data['network']['simple_dispatchable_device'][j]['q_0'])
        qc_map[j] = n_qc
        n_qc += 1
    
    # Ramping constraints
    pru[j] = data['network']['simple_dispatchable_device'][j]['p_ramp_up_ub']
    prd[j] = data['network']['simple_dispatchable_device'][j]['p_ramp_down_ub']
    #pru_su[j] = data['network']['simple_dispatchable_device'][j]['p_startup_ramp_ub'] 
    #prd_sd[j] = data['network']['simple_dispatchable_device'][j]['p_shutdown_ramp_ub']
    
    
    # >
    prusu.append(data['network']['simple_dispatchable_device'][j]['p_startup_ramp_ub'])
    prdsd.append(data['network']['simple_dispatchable_device'][j]['p_shutdown_ramp_ub'])
    
    
    # Reserve upper bounds
    prgu_ub[j] = data['network']['simple_dispatchable_device'][j]['p_reg_res_up_ub']
    prgd_ub[j] = data['network']['simple_dispatchable_device'][j]['p_reg_res_down_ub']
    syn_ub[j] = data['network']['simple_dispatchable_device'][j]['p_syn_res_ub']
    nsyn_ub[j] = data['network']['simple_dispatchable_device'][j]['p_nsyn_res_ub']
    rru_on_ub[j] = data['network']['simple_dispatchable_device'][j]['p_ramp_res_up_online_ub']
    rru_off_ub[j] = data['network']['simple_dispatchable_device'][j]['p_ramp_res_up_offline_ub']
    rrd_on_ub[j] = data['network']['simple_dispatchable_device'][j]['p_ramp_res_down_online_ub']
    rrd_off_ub[j] = data['network']['simple_dispatchable_device'][j]['p_ramp_res_down_offline_ub']
    
    

    su_cost.append( data['network']['simple_dispatchable_device'][j]['startup_cost']) # startup cost
    sd_cost.append(data['network']['simple_dispatchable_device'][j]['shutdown_cost'])   
    on_cost.append(data['network']['simple_dispatchable_device'][j]['on_cost']) 
    down_time.append(data['network']['simple_dispatchable_device'][j]['down_time_lb'])
    up_time.append(data['network']['simple_dispatchable_device'][j]['in_service_time_lb'])       
    startups_ub[j]=data['network']['simple_dispatchable_device'][j]['startups_ub']
    Wen_max[j]=data['network']['simple_dispatchable_device'][j]['energy_req_ub']
    Wen_min[j]=data['network']['simple_dispatchable_device'][j]['energy_req_lb']
    



    
j_sgn = np.ones((n_j*n_t,))

j_sgn_ = np.ones((n_j,))



# if n_b<2000:
#     n_tt=n_t_ 
# elif Seq_2!=0: # Seq-Ipopt large Networks 
#     n_tt=n_t_
# else: # one-step Ipopt
#     n_tt=1
    


    
    
    
        
beta = np.array(beta*n_t)
q0 = np.array(q0*n_t)
beta_min = np.array(beta_min*n_t)
beta_max = np.array(beta_max*n_t)
q0min = np.array(q0min*n_t)
q0max = np.array(q0max*n_t)

# For devices we need two steps: one isolating cost blocks and modelling each as seperate device

# Then a matrix which maps each load id to a list of load indices.

# Ramping limits will be on sum of block. Cost is conex/concave so order is unimportant

n_je = 0
je_i_map  = {}
je_t_map  = {}
je_j_map  = {}
c_en = []
pje_max = []
j_t_je_map = {}

rows = [] # mapping for je to bus, time
cols = []
vals = []
rows2 = [] # mapping for j,t to bus, time
cols2 = []
vals2 = []
pje0 = []
delta_tj = []
# simple dispathcable device

# for ramping constraints we will also need a matrix which maps je_to_j
rows3 = []
cols3 = []

# for fixed load factor
rows4 = []
cols4 = []

# rows for bounded load factor
rows5 = []
cols5 = []

# for energy constraint
#rows4 = []
#cols4 = []
#vals4 = []

j_je={}   # (device j, time t): je indices
j_j2 = {}


# upper and lower bound on devices
j_on_ub={}
j_on_lb={}

    
prup = np.array([None]*(n_j*n_t))
prdn = np.array([None]*(n_j*n_t))




if ramp_const==1:
    
    #  Controllable load Version for MIP  
    for j2 in range(n_j):
        # GAHHH ORDER NOT PRESETVED BETWEEN LISTS
        uid = data['time_series_input']['simple_dispatchable_device'][j2]['uid']
        j = j_map[uid]
        i = j_bus_map[j]
    
        j_t_je_map[j] = {}
        j_j2[j] = j2
        
        j_sgn_[j] = sign_[uid]
    
        j_on_ub[j]=data['time_series_input']['simple_dispatchable_device'][j2]['on_status_ub']    
        j_on_lb[j]=data['time_series_input']['simple_dispatchable_device'][j2]['on_status_lb']
    
        for t in range(n_t_):
            pj_min_all_[j][t] = data['time_series_input']['simple_dispatchable_device'][j2]['p_lb'][t]
            pj_max_all_[j][t] = data['time_series_input']['simple_dispatchable_device'][j2]['p_ub'][t]
        
        for t in range(n_t):
            j_t_je_map[j][t] = []
    
            pmax = data['time_series_input']['simple_dispatchable_device'][j2]['p_ub'][t]
            pmin = data['time_series_input']['simple_dispatchable_device'][j2]['p_lb'][t]
    
            # Q upper and lower bounds
            # qj_max[j+t*n_j] = data['time_series_input']['simple_dispatchable_device'][j2]['q_ub'][t]
            # qj_min[j+t*n_j] = data['time_series_input']['simple_dispatchable_device'][j2]['q_lb'][t]
    
    
            pj_min_[j][t] = data['time_series_input']['simple_dispatchable_device'][j2]['p_lb'][t]
            pj_max_[j][t] = data['time_series_input']['simple_dispatchable_device'][j2]['p_ub'][t]
    
    
            # First let's seperate out the uncontrollable generation or demand
            #pl0[i+t*n_j] += pmin*sign[uid]
            # _p0 = copy.deepcopy(pj0[j])
    
            # We are minimizing COST of generation, so cost should be positive, benefit should be negative
            cost_blocks = [[float(b[0])*sign_[uid], b[1]] for b in data['time_series_input']['simple_dispatchable_device'][j2]['cost'][t]]
    
            cost_blocks = sorted(cost_blocks, reverse=True) # Highest cost consuming  / lowest cost of producing first
                                                            # That is to maximize the market surplus
    
            # print(cost_blocks)                                                
            _p = 0
            first = 1
    
            j_je_local=[]
            for b in range(len(cost_blocks)):
                # if _p+cost_blocks[b][1] <= pmin:
                #     low+=1
                # if pmax<=cost_blocks[b][1]:
                #     high+=1
                    # print(j2)
                    # print(t)
    
    
                if _p >= pmax:
                    continue
                # if _p+cost_blocks[b][1] <= pmin: # skip if whole block uncontrollable
                #     continue
                # _p is total load not including current block
                # pmin is total uncontrollable load
                # pmax is the total possible load (uncontrolled + controllable)
                # _p0 is the total initial load
                # cost_blocks[b][1] is the amount of load IN THIS BLOCK (not cumulative)
    
                # pj_0.append(min(max(0,_p0-_p),cost_blocks[b][1]+first*(_p-pmin)))
    
    
                # pje_max.append(min(pmax-_p,cost_blocks[b][1]+first*(_p-pmin)))
                pje_max.append(min(pmax-_p,cost_blocks[b][1]))
                first = 0 
    
                c_en.append(cost_blocks[b][0])
                delta_tj.append(delta_t[t])
                _p += cost_blocks[b][1]            
    
                je_i_map[n_je]  = i
                je_t_map[n_je]  = t
                j_t_je_map[j][t].append(n_je)
                # rows += [i+n_b*t]
                # cols += [n_je]
                # vals += [sign[uid]]
    
                # rows3 += [j+n_j*t]
                # cols3 += [n_je]
    
                j_je_local+=[n_je]
    
    
                n_je += 1
    
            j_je[j,t]=j_je_local




else:
    
    
    # pre-processing step - artifically change upper and lower bounds
    p_max_new = np.zeros((n_j,n_t_))
    p_min_new = np.zeros((n_j,n_t_))
    for j2 in range(n_j):
        # GAHHH ORDER NOT PRESETVED BETWEEN LISTS
        uid = data['time_series_input']['simple_dispatchable_device'][j2]['uid']
        j = j_map[uid]
        for tt in range(n_t_):
            p_max_new[j2,tt] = copy.deepcopy(data['time_series_input']['simple_dispatchable_device'][j2]['p_ub'][tt])
            p_min_new[j2,tt] = copy.deepcopy(data['time_series_input']['simple_dispatchable_device'][j2]['p_lb'][tt])
        #'''
        for tt in np.arange(n_t_-2,-1,-1):
            p_max_new[j2,tt] = min(p_max_new[j2,tt],p_max_new[j2,tt+1]+prd[j]*delta_t_[tt])
            p_min_new[j2,tt] = max(p_min_new[j2,tt],p_min_new[j2,tt+1]-pru[j]*delta_t_[tt])
    
        for tt in range(1,n_t_):
            p_max_new[j2,tt] = min(p_max_new[j2,tt],p_max_new[j2,tt-1]+pru[j]*delta_t_[tt-1])
            p_min_new[j2,tt] = max(p_min_new[j2,tt],p_min_new[j2,tt-1]-prd[j]*delta_t_[tt-1])
        
        
    pj_min_=p_min_new
    pj_max_=p_max_new
    
    #  Controllable load Version for MIP  
    for j2 in range(n_j):
        # GAHHH ORDER NOT PRESETVED BETWEEN LISTS
        uid = data['time_series_input']['simple_dispatchable_device'][j2]['uid']
        j = j_map[uid]
        i = j_bus_map[j]
    
        j_t_je_map[j] = {}
        j_j2[j] = j2
        
        j_sgn_[j] = sign_[uid]
    
        j_on_ub[j]=data['time_series_input']['simple_dispatchable_device'][j2]['on_status_ub']    
        j_on_lb[j]=data['time_series_input']['simple_dispatchable_device'][j2]['on_status_lb']
    
        # for t in range(n_t_):
        #     pj_min_all_[j][t] = data['time_series_input']['simple_dispatchable_device'][j2]['p_lb'][t]
        #     pj_max_all_[j][t] = data['time_series_input']['simple_dispatchable_device'][j2]['p_ub'][t]
        
        for t in range(n_t):
            j_t_je_map[j][t] = []
    
            #pmax = data['time_series_input']['simple_dispatchable_device'][j2]['p_ub'][t]
            pmax=p_max_new[j2,t]
            #pmin = data['time_series_input']['simple_dispatchable_device'][j2]['p_lb'][t]
            pmin=p_min_new[j2,t]
            
            # Q upper and lower bounds
            # qj_max[j+t*n_j] = data['time_series_input']['simple_dispatchable_device'][j2]['q_ub'][t]
            # qj_min[j+t*n_j] = data['time_series_input']['simple_dispatchable_device'][j2]['q_lb'][t]
    
    
            # pj_min_[j][t] = data['time_series_input']['simple_dispatchable_device'][j2]['p_lb'][t]
            # pj_max_[j][t] = data['time_series_input']['simple_dispatchable_device'][j2]['p_ub'][t]
    
    
            # First let's seperate out the uncontrollable generation or demand
            #pl0[i+t*n_j] += pmin*sign[uid]
            # _p0 = copy.deepcopy(pj0[j])
    
            # We are minimizing COST of generation, so cost should be positive, benefit should be negative
            cost_blocks = [[float(b[0])*sign_[uid], b[1]] for b in data['time_series_input']['simple_dispatchable_device'][j2]['cost'][t]]
    
            cost_blocks = sorted(cost_blocks, reverse=True) # Highest cost consuming  / lowest cost of producing first
                                                            # That is to maximize the market surplus
    
            # print(cost_blocks)                                                
            _p = 0
            first = 1
    
            j_je_local=[]
            for b in range(len(cost_blocks)):
                # if _p+cost_blocks[b][1] <= pmin:
                #     low+=1
                # if pmax<=cost_blocks[b][1]:
                #     high+=1
                    # print(j2)
                    # print(t)
    
    
                if _p >= pmax:
                    continue
                # if _p+cost_blocks[b][1] <= pmin: # skip if whole block uncontrollable
                #     continue
                # _p is total load not including current block
                # pmin is total uncontrollable load
                # pmax is the total possible load (uncontrolled + controllable)
                # _p0 is the total initial load
                # cost_blocks[b][1] is the amount of load IN THIS BLOCK (not cumulative)
    
                # pj_0.append(min(max(0,_p0-_p),cost_blocks[b][1]+first*(_p-pmin)))
    
    
                # pje_max.append(min(pmax-_p,cost_blocks[b][1]+first*(_p-pmin)))
                pje_max.append(min(pmax-_p,cost_blocks[b][1]))
                first = 0 
    
                c_en.append(cost_blocks[b][0])
                delta_tj.append(delta_t[t])
                _p += cost_blocks[b][1]            
    
                je_i_map[n_je]  = i
                je_t_map[n_je]  = t
                j_t_je_map[j][t].append(n_je)
                # rows += [i+n_b*t]
                # cols += [n_je]
                # vals += [sign[uid]]
    
                # rows3 += [j+n_j*t]
                # cols3 += [n_je]
    
                j_je_local+=[n_je]
    
    
                n_je += 1
    
            j_je[j,t]=j_je_local
            
            

        
# Coverting the dic to matrix to comply with gurobi format

j_on_lb_=np.zeros((n_j,n_t_))
j_on_ub_=np.zeros((n_j,n_t_))

for j in range(n_j):
    for t in range(n_t_):

        j_on_lb_[j][t]=j_on_lb[j][t]
        j_on_ub_[j][t]=j_on_ub[j][t]

j_on_lb=j_on_lb_
j_on_ub=j_on_ub_

#                  
pj_max_ = np.array(pj_max_)
c_en = np.array(c_en)
delta_tj = np.array(delta_tj)




#branch_map = {} not sure if we will need this or not for now
rows = []
cols = []
rows2 = []

fr_to={}  # device(j): from bus  to bus 

# branches time - do AC first then DC
f = 0
f_ = 0
n_f = (len(data['network']['ac_line'])+len(data['network']['two_winding_transformer']))
branch_map = {}
f_on = np.ones((n_f)) # TEMPORARY

# 
f_on_0 = np.ones((n_f,)) # TEMPORARY
Bf = np.zeros((n_f,))


sf = np.zeros((n_f,))
bf = np.zeros((n_f,))
bfCH = np.zeros((n_f,))
gf = np.zeros((n_f,))
g_fr = np.zeros((n_f,))
g_to = np.zeros((n_f,))
b_fr = np.zeros((n_f,))
b_to = np.zeros((n_f,))
tauf = np.ones((n_f,))
thetaf = np.zeros((n_f,))

# 
s_max = np.zeros((n_f,)) # branch flow limit (130)-(131)

# 
pdc_max_ = {}
su_cost_f=np.zeros((n_f)) # (50)-(51)
sd_cost_f=np.zeros((n_f))



for ac in data['network']['ac_line']:
    f_on[f] = ac['initial_status']['on_status']
    

    f_on_0[f]=ac['initial_status'] ['on_status']
    fr_to[f]=[bus_map[ac['fr_bus']],bus_map[ac['to_bus']]]
    
    su_cost_f[f]=ac['connection_cost']
    sd_cost_f[f]=ac['disconnection_cost']
    
    x = ac['x']
    Bf[f]=1/x
    s_max[f]=ac['mva_ub_nom']
    
    
    for t in range(n_t):
        rows += [bus_map[ac['fr_bus']]+t*n_b]
        rows2 += [bus_map[ac['to_bus']]+t*n_b]
        cols += [t*n_f+f]
    r = ac['r']
    x = ac['x']
    z = r*r+x*x
    bf[f] = -x/z
    gf[f] = r/z
    sf[f] = ac['mva_ub_nom']
    bfCH[f] = ac['b'] # check that this is the right parameter
    if ac['additional_shunt'] == 1:
        g_fr[f] = ac['g_fr']
        g_to[f] = ac['g_to']
        b_fr[f] = ac['b_fr']
        b_to[f] = ac['b_to']
    branch_map[ac['uid']] = f
    f += 1
    
    #
    f_ += 1
    

dc_f=[] # dc devices
for dc in data['network']['dc_line']:
    # print(f)
    dc_f+=[f_]

    fr_to[f_]=[bus_map[dc['fr_bus']],bus_map[dc['to_bus']]]

    # s_max[f]=dc['mva_ub_nom']     # does not apply to dc lines
    pdc_max_[f_] = dc['pdc_ub']

    #f_ += 1
    
    

for xfm in data['network']['two_winding_transformer']:
    f_on[f] = xfm['initial_status']['on_status']

    f_on_0[f_]=xfm['initial_status'] ['on_status']
    fr_to[f_]=[bus_map[xfm['fr_bus']],bus_map[xfm['to_bus']]]

    su_cost_f[f_]=xfm['connection_cost']
    sd_cost_f[f_]=xfm['disconnection_cost']

    
    for t in range(n_t):
        rows += [bus_map[xfm['fr_bus']]+t*n_b]
        rows2 += [bus_map[xfm['to_bus']]+t*n_b]
        cols += [t*n_f+f]
    thetaf[f] = xfm['initial_status']['ta'] # FOR NOW JUST LEAVING XFM IN INITIAL POSITION
    tauf[f] = xfm['initial_status']['tm']
    r = xfm['r']
    x = xfm['x']
    z = r*r+x*x
    bf[f] = -x/z
    gf[f] = r/z
    bfCH[f] = xfm['b']
    sf[f] = xfm['mva_ub_nom']
    if xfm['additional_shunt'] == 1:
        g_fr[f] = xfm['g_fr']
        g_to[f] = xfm['g_to']
        b_fr[f] = xfm['b_fr']
        b_to[f] = xfm['b_to']
    

    x = xfm['x']
    # z = r*r+x*x
    # bf[f] = -x/z
    Bf[f_]=1/x
    s_max[f_]=xfm['mva_ub_nom']
    
    
    branch_map[xfm['uid']] = f
    
    f += 1
    f_ += 1



 
# temporary
# f_on = [1.]*n_f 
# f_on_0= [1.]*n_f 


f_on = np.array(list(f_on)*n_t)
sf = np.array(list(sf)*n_t)
bf = np.array(list(bf)*n_t)
gf = np.array(list(gf)*n_t)
bfCH = np.array(list(bfCH)*n_t)
tauf = np.array(list(tauf)*n_t)
thetaf = np.array(list(thetaf)*n_t)
g_fr = np.array(list(g_fr)*n_t)
b_fr = np.array(list(b_fr)*n_t)
g_to = np.array(list(g_to)*n_t)
b_to = np.array(list(b_to)*n_t)

# Only after ALL branches are done
fo_connected = coo_matrix(([1.]*(n_f*n_t), (rows, cols)),shape=(n_b*n_t,n_f*n_t))
fd_connected = coo_matrix(([1.]*(n_f*n_t), (rows2, cols)),shape=(n_b*n_t,n_f*n_t))

dc_map = {}
rows = []
rows2 = []
cols = []

n_dc = len(data['network']['dc_line']) # still need to do this
e = 0
pdc_max = []
qdc_fr_max = []
qdc_to_max = []
qdc_fr_min = []
qdc_to_min = []
pdc0 = []
qdc_fr0 = []
qdc_to0 = []

for dc in data['network']['dc_line']:
    dc_map[dc['uid']] = e
    for t in range(n_t):
        rows += [bus_map[dc['fr_bus']]+t*n_b]
        rows2 += [bus_map[dc['to_bus']]+t*n_b]
        cols += [e+t*n_dc]
    
    pdc_max.append(0)#dc['pdc_ub'])
    qdc_fr_max.append(0)#dc['qdc_fr_ub'])
    qdc_to_max.append(0)#dc['qdc_to_ub'])
    qdc_fr_min.append(0)#dc['qdc_fr_lb'])
    qdc_to_min.append(0)#dc['qdc_to_lb'])
    pdc0.append(0)#dc['initial_status']['pdc_fr'])
    #pdc_max.append(dc['pdc_ub'])
    #qdc_fr_max.append(dc['qdc_fr_ub'])
    #qdc_to_max.append(dc['qdc_to_ub'])
    #qdc_fr_min.append(dc['qdc_fr_lb'])
    #qdc_to_min.append(dc['qdc_to_lb'])
    #pdc0.append(dc['initial_status']['pdc_fr'])
    qdc_fr0.append(dc['initial_status']['qdc_fr'])
    qdc_to0.append(dc['initial_status']['qdc_to'])
              
    e += 1

pdc_max = np.array(pdc_max*n_t)
qdc_fr_max = np.array(qdc_fr_max*n_t)
qdc_to_max = np.array(qdc_to_max*n_t)
qdc_fr_min = np.array(qdc_fr_min*n_t)
qdc_to_min = np.array(qdc_to_min*n_t)
pdc0 = np.array(pdc0*n_t)
qdc_fr0 = np.array(qdc_fr0*n_t)
qdc_to0 = np.array(qdc_to0*n_t)

do_connected = coo_matrix(([1.]*(n_dc*n_t), (rows, cols)),shape=(n_b*n_t,n_dc*n_t))
dd_connected = coo_matrix(([1.]*(n_dc*n_t), (rows2, cols)),shape=(n_b*n_t,n_dc*n_t))


file.close()





# MIP Code
fr_bus={}   # bus: device (j) or devices (j, ..)
to_bus={}   # bus: device (j) or devices (j, ..)

for i in range(n_b):

    fr_bus[i]={}
    to_bus[i]={}


for f in range(n_f):
    if f not in (dc_f):
        fr_bus[fr_to[f][0]]=list(fr_bus[fr_to[f][0]])+[f]
        to_bus[fr_to[f][1]]=list(to_bus[fr_to[f][1]])+[f]


bus_j_map={} # bus: devices 

for i in range(n_b):
    colect=[]
    for j in range (n_j):

        if j_bus_map[j]==i:
            colect+=[j]
    bus_j_map[i]=colect
   


#     ZON=[delta_t[t]*on_cost[j]*j_on_0[j] for t in range(n_t) for j in range (n_j)]
#     ZON=sum(ZON)
#     print('ZON of NLP =', ZON)
#     ZSU=0
#     ZSD=0

    

# %%    
# >> MIP Model        
sol=1
# Time Count
import timeit
start = timeit.default_timer()

if switch2==0:

    # Model
    m=gp.Model()
    
    # Suppress the solver output
    m.setParam("OutputFlag", 1)
    
    #m.setParam('Threads', 90)  # Set to the number of threads you want to use

    # m.params.presolve=0
    # m.Params.FeasibilityTol = 1e-3
    # IntFeasTol=1e-2
    
    #m.Params.MIPGap=0.01
    
    # if div==1:
    #     m.Params.TIME_LIMIT = 60*3
    # elif div==2:
    #     m.Params.TIME_LIMIT = 60*25
    # else:
    #     m.Params.TIME_LIMIT = 60*25
        

    # m.Params.NoRelHeurTime=20 
    # m.Params.Threads=1

    # m.setParam("Method",3) #
    # m.setParam("DegenMoves",0) 
    # m.Params.MIPFocus=3

    # Assign variables
    theta = m.addVars(n_b, n_t, vtype='C', lb=-np.pi , ub=np.pi)
    pje = m.addVars(n_je, vtype='C', lb=0 , ub=pje_max)

    # Pj can be zero according to linear formulation. lb is pjmin in the quadratic formulation
    # pj = m.addVars(n_j,n_t, vtype='C', lb=pj_min_ , ub=pj_max_)
    
    
    pj = m.addVars(n_j,n_t, vtype='C') 

    # >>
    j_on = m.addVars(n_j,n_t,vtype=GRB.BINARY,lb=j_on_lb, ub=j_on_ub) 
    # j_on = m.addVars(n_j,n_t,vtype='C',lb=j_on_lb, ub=j_on_ub) 

    f_on_= np.transpose([f_on_0] * n_t)


    # bounds on power
    # ub_1=np.max(pj_max_)*n_j
    # lb_1=-ub_1
    if bounds==1:
        ub_=np.max(pj_max_)*n_j
        lb_=-ub_
    else:
        ub_=GRB.INFINITY
        lb_=-GRB.INFINITY
    
    # if bounds==1:
    #     # new
    #     ub_=np.zeros((n_b))
    #     for i in range(n_b):
  
        
    #         for j_at_bus in bus_j_map[i]:
    #             ub_[i]=ub_[i]+np.max(pj_max_[j_at_bus])
            
        
    #     ub_ = np.tile(ub_, (n_t, 1))
    #     ub_=np.transpose(ub_)
    #     lb_=-ub_
    




    # bus power (one_time usage)
    P= m.addVars(n_b, n_t, vtype='C', lb=lb_, ub=ub_)      # Consuming(+)  Producing (-)
    P_fr= m.addVars(n_b, n_t, vtype='C', lb=lb_ , ub=ub_)  # 
    P_to= m.addVars(n_b, n_t, vtype='C', lb=lb_, ub=ub_)   # 
    Pmis= m.addVars(n_b, n_t, vtype='C', lb=lb_ , ub=ub_ ) # 
    Pmis_plus= m.addVars(n_b, n_t, vtype='C', lb=0  , ub=ub_) # 

    # Branch
    P_fr_f = m.addVars(n_f,n_t, vtype='C', lb=lb_ , ub=ub_) 
    P_to_f = m.addVars(n_f,n_t, vtype='C', lb=lb_, ub=ub_) 

    # Branch Limits
    s_plus= m.addVars(n_f,n_t,vtype='C', lb=0) # (128)
    zen= m.addVars(n_j, n_t, vtype='C', lb=-GRB.INFINITY, ub=GRB.INFINITY) 




    # Absolut value of power mismatch Constraints (4)-(7) (6 and 7 eliminated):

    for i in range(n_b):
        for t in range (n_t):
            m.addConstr(Pmis_plus[i,t]>=Pmis[i,t],name='c0')
            m.addConstr(Pmis_plus[i,t]>=-Pmis[i,t], name='c1')

    # Power Balance Constraint (10):
    # start:

    for t in range (n_t):  
        for i in range(n_b):

            j_at_bus=bus_j_map[i]   # find the devices belongs to this bus       

            # linear version:
            m.addConstr(P[i,t]==gp.quicksum(pj[j,t]*j_sgn_[j] for j in j_at_bus), name='c2' )


            #  P_fr
            f_fr=fr_bus[i] # devices from bus that belongs to this bus
            # P_fr[i,t]=0
            for f in f_fr:
                if f not in dc_f:
                    theta_index=fr_to[f]
                    m.addConstr(P_fr_f[f,t]==Bf[f]*(theta[theta_index[0],t]-theta[theta_index[1],t]), name='c3')


                    # Constraint (130): Branch flow limits
                    # linear
                    m.addConstr(P_fr_f[f,t]<=s_max[f]+s_plus[f,t])


            m.addConstr(P_fr[i,t]==gp.quicksum(P_fr_f[f,t]*f_on_[f,t] for f in f_fr) )


            #  P_to
            f_to=to_bus[i]
            # P_to[i,t]=0
            for f in f_to:
                if f not in dc_f:
                    theta_index=fr_to[f]
                    m.addConstr(P_to_f[f,t]==-Bf[f]*(theta[theta_index[0],t]-theta[theta_index[1],t]), name='c5')



                    # Constraint (131): Branch flow limits                
                    # linear
                    m.addConstr(P_to_f[f,t]<=s_max[f]+s_plus[f,t])


            m.addConstr(P_to[i,t]==gp.quicksum(P_to_f[f,t]*f_on_[f,t] for f in f_to) )




            # P_s
            # shunts=bus_shunt_map[i] # shunts of the current bus

            # m.addConstr(Ps[i,t]==gp.quicksum(gs[s]*ush[s]*v2[i,t] for s in shunts) , name='c8')
            # m.addConstr(Ps[i,t]==gp.quicksum(gs[s]*ush[s] for s in shunts) , name='c8')



    # with shunt power            
    # m.addConstrs((P[i,t]+Ps[i,t]+P_fr[i,t]+P_to[i,t]==Pmis[i,t] for i in range(n_b) for t in range (n_t)), name='c9')

    # no shunt power
    m.addConstrs((P[i,t]+P_fr[i,t]+P_to[i,t]==Pmis[i,t] for i in range(n_b) for t in range (n_t)), name='c9')




    # end of constraint (10)







    # linear-version trick (slide 5) to remove the quadratic terms:

    if semi_blacoks==0:    
        for j in range(n_j):
            for t in range(n_t):
    
                m.addConstr(pj[j,t]>=j_on[j,t]*pj_min_[j,t])
                count+=1
                m.addConstr(pj[j,t]<=j_on[j,t]*pj_max_[j,t])
                count+=1


    # # v2=v^2
    # m.addConstrs((v2[i,t]==v[i,t]*v[i,t]   for i in range(n_b) for t in range (n_t)), name='c10')




    # pj>=pj_min_: that is can be enforced by pj lower bound (lb=pj_min_) 
    # Pj[j,t]=sum(pje)
    
    if semi_blacoks==0:
        for t in range(n_t):
            for j in range (n_j):  
    
                indexx=j_je[j,t] # find pje indices  of one device J at given time t
    
                m.addConstr(pj[j,t]==gp.quicksum(pje[index] for index in indexx), name='c11')
                count+=1
    else:
        
        for t in range(n_t):
            for j in range (n_j):  
    
                indexx=j_je[j,t] # find pje indices  of one device J at given time t
                m.addConstr(pj[j,t]==gp.quicksum(pje[index] for index in indexx), name='c11')
                count+=1
                
                if compact==0:
                    for index in indexx:
                        count+=1
                        
                        pje[index]<=pje_max[index]*j_on[j,t] 
                else: 
                        count+=1
                        gp.quicksum(pje[index] for index in indexx)<=gp.quicksum(pje_max[index]*j_on[j,t] for index in indexx) 
                    
        
        



    # (142)-(146) 
    # Dc line limits
    for f in dc_f:
        for t in range(n_t):

            m.addConstr(P_fr_f[f,t]<=pdc_max_[f])   #(142)
            m.addConstr(P_fr_f[f,t]>=-pdc_max_[f])

            m.addConstr(P_to_f[f,t]<=pdc_max_[f])   #(144)
            m.addConstr(P_to_f[f,t]>=-pdc_max_[f])

            m.addConstr(P_fr_f[f,t]+P_to_f[f,t]==0)   #(146)





    # on-off, startup, shoutdown
    # (42) & (43)
    # j_su = m.addVars(n_j,n_t,vtype=GRB.BINARY)   
    # j_sd = m.addVars(n_j,n_t,vtype=GRB.BINARY)
    
    # >>
    j_su = m.addVars(n_j,n_t,vtype='C')   
    j_sd = m.addVars(n_j,n_t,vtype='C')
    
    
    



    # # (46)-(47)
    for j in range (n_j):
        t=0
        m.addConstr(j_on[j,t]-j_on_0[j]==j_su[j,t]-j_sd[j,t])
        for t in range (1,n_t):
            m.addConstr(j_on[j,t]-j_on[j,t-1]==j_su[j,t]-j_sd[j,t], name='p1')




    ## (48)
    # for t in range(n_t):
    #     for j in range (n_j):
    #         m.addConstr(j_su[j,t]+j_sd[j,t]<=1)
            
    # # one time switching
    if switch1==1:
        for j in range (n_j):
            m.addConstr(gp.quicksum(j_su[j,t]+j_sd[j,t] for t in range(n_t))<=1)

        
    # # no shut down (temporary)       
    # for j in range (n_j):
    #     m.addConstr(j_on_0[j]<=j_on[j,0])       
    # for j in range(n_j):
    #     for t in range(n_t-1):
    #         m.addConstr(j_on[j,t]<=j_on[j,t+1]) 





    
    


    # on-off startup shutdown costs
    # (49),(50),(51)

    Zon=m.addVar(vtype='C', lb=0) # Cost function        
    m.addConstr(Zon==gp.quicksum(delta_t[t]*on_cost[j]*j_on[j,t] for t in range(n_t) for j in range (n_j)))


    Zsu=m.addVar(vtype='C', lb=0) # Cost function  

    m.addConstr(Zsu==gp.quicksum(su_cost[j]*j_su[j,t] for t in range(n_t) for j in range (n_j)))

    Zsd=m.addVar(vtype='C', lb=0) # Cost function
 
    m.addConstr(Zsd==gp.quicksum(sd_cost[j]*j_sd[j,t] for t in range(n_t) for j in range (n_j)))




    T=np.zeros(n_t)

    for t in range(n_t):

        if t>=1:
            T[t]=T[t-1]+delta_t[t]
        else:
            T[t]=delta_t[t]
    

    if lazy_const==0:
        # # (56)-(58):
        # # (56): Minimum downtime
        for j in range(n_j):
            
            down_time_=down_time[j]
            if down_time_>0:
                T_elap=delta_t[0] # T3
                # T_elap=0 # T2
                st_ind=0
    
                for t in range(1,n_t):
                    et_ind=t
                    
                    if tight_const==0:
                        m.addConstr(j_su[j,t]<=1-gp.quicksum(j_sd[j,t_] for t_ in range (st_ind,et_ind))) 
                    else:
                        for t_ in range (st_ind,et_ind):
                            m.addConstr(j_su[j,t]<=1-j_sd[j,t_])
                        
                        
                    c1+=1
    
                    dt=delta_t[t]
                    T_elap+=dt
    
                    if T_elap>=down_time_:
                        st_ind=st_ind+1
                        #T_elap=0



        # # (57): Minimum uptime
        for j in range(n_j):
            up_time_=up_time[j]
            if up_time_>0:
    
                T_elap=delta_t[0] # T3
                # T_elap=0          # T2
                st_ind=0
                m.addConstr(j_sd[j,0]==0)
    
                for t in range(1,n_t):
                    et_ind=t
                    if j_on_0[0]==1 and st_ind==0:
                        m.addConstr(j_sd[j,t]<=0)
                        c1+=1
    
                    else: 
                        if tight_const==0:
                            m.addConstr(j_sd[j,t]<=1-gp.quicksum(j_su[j,t_] for t_ in range (st_ind,et_ind)))
                        else: 
                            for t_ in range (st_ind,et_ind):
                                m.addConstr(j_sd[j,t]<=1-j_su[j,t_])
                            
                        c1+=1
    
    
                    dt=delta_t[t]
                    T_elap+=dt
    
                    if T_elap>=up_time_:
                        st_ind=st_ind+1
                        #T_elap=0
    
        # (58)   
    
        for j in range(n_j):
            startups_j=startups_ub[j]
    
            if len(startups_j)>0:
    
                for startups in startups_j: # it may have more than one matrix
    
    
                    st=startups[0]
                    et=startups[1]
    
                    for t in range(n_t):
                        if st<T[t]:
                            st_ind=t
                            break
    
                    et_ind=len(T)-1 # In case the end time is higher than the control horizon     
                    for t in range(n_t):
                        if et<=T[t]:
                            et_ind=t
                            break  
    
    
                    m.addConstr(gp.quicksum(j_su[j,t_] for t_ in range(st_ind,et_ind+1))<=startups[2])
                    c1+=1





    # Costs & Objective

    # zen=0  # producing & consuming
    # tt=0
    for j in range (n_j):
        for t in range (n_t):
            indexx=j_je[j,t]
            m.addConstr(zen[j,t]==gp.quicksum(c_en[index]*pje[index]*delta_tj[index] for index in indexx), name='c12')






    # (8) bus power violation/mismatch cost
    cp=data['network']['violation_cost']['p_bus_vio_cost']
    Zp=m.addVar(vtype='C', lb=0) # Cost function
    m.addConstr(Zp==gp.quicksum(delta_t[t]*cp*Pmis_plus[i,t] for i in range (n_b) for t in range (n_t)))
    # Zp=0

    # (129) Branch limit penalty
    cs=data['network']['violation_cost']['s_vio_cost']   
    Zs=m.addVar(vtype='C', lb=0) # Cost function
    m.addConstr(Zs==gp.quicksum(delta_t[t]*cs*s_plus[f,t] for f in range (n_f) for t in range (n_t)))








    # ## Section 2.6.3  Maximum/Minimum Energy Over intervals
    # (68)-(69)
    for j in range(n_j):

        Wen_max_j=Wen_max[j]

        if len(Wen_max_j)>0:

            for Wen_max_ in Wen_max_j:  # it might have more than one device

                st=Wen_max_[0]
                et=Wen_max_[1]

                for t in range(n_t):
                    if st<T[t]:
                        st_ind=t
                        break
                et_ind=len(T)-1 # In case the end time is higher than the control horizon                
                for t in range(n_t):
                    if et<=T[t]:
                        et_ind=t
                        break        


                m.addConstr(gp.quicksum(delta_t[t]*pj[j,t] for t in range(st_ind+1,et_ind))+(T[st_ind]-st)*pj[j,st_ind]+(et-T[et_ind-1])*pj[j,et_ind]<=Wen_max_[2])


        Wen_min_j=Wen_min[j]

        if len(Wen_min_j)>0:
            for Wen_min_ in Wen_min_j:  # it might have more than one device


                st=Wen_min_[0]
                et=Wen_min_[1]

                for t in range(n_t):
                    if st<T[t]:
                        st_ind=t
                        break
                et_ind=len(T)-1 # In case the end time is higher than the control horizon                  

                for t in range(n_t):
                    if et<=T[t]:
                        et_ind=t
                        break 

                m.addConstr(gp.quicksum(delta_t[t]*pj[j,t] for t in range(st_ind+1,et_ind))+pj[j,st_ind]*(T[st_ind]-st)+pj[j,et_ind]*(et-T[et_ind-1])>=Wen_min_[2])

                
                

                
#         #   # No shutting down at t=0 
#         # for j in range(n_j):
#         #     m.addConstr(j_on_0[j]<=j_on[j,0])             
    # no startup (temporary constraint)
    #m.addConstrs( j_su[j,t]==0 for j in range(n_j) for t in range(n_t))

    
    
    # T3
    # ramping:
    ## start>>        
    # ramping down constraints at t=0 (additional constraint)
    # B_0=m.addVars(n_j,vtype=GRB.BINARY) 
    for j in range(n_j):
#             m.addConstr(float(pj0[j])<=pj_min_[j,0]+(1-j_sd[j,0])*GRB.INFINITY)
        m.addConstr(float(pj0[j])<=pj_min_[j,0]+(1-j_sd[j,0])*pj0[j])
        # M=3
        # m.addConstr(j_on_0[j]>=j_on[j,0]-M*(1-j_sd[j,0]))
        # m.addConstr(j_on_0[j]<=j_on[j,0]+M*j_sd[j,0]) 
                

    # ramping down constraints (additional constraint)
    # B_=m.addVars(n_j,n_t,vtype=GRB.BINARY)
    for j in range(n_j):
        for t in range(n_t-1):
           m.addConstr(pj[j,t]<=pj_min_[j,t]+(1-j_sd[j,t+1])*(pj_max_[j,t]-pj_min_[j,t]))
           # M=3
           # m.addConstr(j_on[j,t]>=j_on[j,t+1]-M*(1-j_sd[j,t+1]))
           # m.addConstr(j_on[j,t]<=j_on[j,t+1]+M*j_sd[j,t+1]) 
    


    # # Ramping Limits: (71)-(74)
    
    
    if ramp_const==1:
    
        ts=0     # ramp up
        # if n_b==73:
        #     for j in range(n_j):
        #         if max(pj_min_[j])<min(delta_t)*prusu[j]:
        #             dxd=0 # do nothing
        #         else:
        #             m.addConstr(pj[j,ts]-pj0[j]<=delta_t[ts]*(pru[j]*(j_on[j,ts]-j_su[j,ts])+prusu[j]*(j_su[j,ts]+1-j_on[j,ts])) )        
        #             m.addConstrs(pj[j,t]-pj[j,t-1]<=delta_t[t]*(pru[j]*(j_on[j,t]-j_su[j,t])+prusu[j]*(j_su[j,t]+1-j_on[j,t])) for t in range(1,n_t))
        # else:
        m.addConstrs(pj[j,ts]-pj0[j]<=delta_t[ts]*(pru[j]*(j_on[j,ts]-j_su[j,ts])+prusu[j]*(j_su[j,ts]+1-j_on[j,ts])) for j in range(n_j) )        
        m.addConstrs(pj[j,t]-pj[j,t-1]<=delta_t[t]*(pru[j]*(j_on[j,t]-j_su[j,t])+prusu[j]*(j_su[j,t]+1-j_on[j,t]))   for j in range(n_j) for t in range(1,n_t) )
    
        
        ts=0      # ramp down
        m.addConstrs(pj[j,ts]-pj0[j]>=-delta_t[ts]*(prd[j]*j_on[j,ts]+prdsd[j]*(1-j_on[j,ts])) for j in range(n_j))          
        m.addConstrs(pj[j,t]-pj[j,t-1]>=-delta_t[t]*(prd[j]*j_on[j,t]+prdsd[j]*(1-j_on[j,t])) for j in range(n_j) for t in range(1,n_t) )     




    
    
    # # T4
    # # ramping:
    # ## start>>      
    # # ramping down constraints at t=0 (additional constraint) 
    # for j in range(n_j):
    #     if pj_min_[j,0]>0:
    #         m.addConstr(float(pj0[j])<=pj_min_[j,0]+(1-j_sd[j,0])*pj0[j])
    #         M=3
    #         m.addConstr(j_on_0[j]>=j_on[j,0]-M*(1-j_sd[j,0]))
    #         m.addConstr(j_on_0[j]<=j_on[j,0]-M*j_sd[j,0]) 
                

    # # ramping down constraints (additional constraint)
    # for j in range(n_j):
    #     for t in range(n_t-1):
    #        m.addConstr(pj[j,t]<=pj_min_[j,t]+(1-j_sd[j,t+1])*(pj_max_[j,t]-pj_min_[j,t]))
    #        M=3
    #        m.addConstr(j_on[j,t]>=j_on[j,t+1]-M*(1-j_sd[j,t+1]))
    #        m.addConstr(j_on[j,t]<=j_on[j,t+1]+M*j_sd[j,t+1]) 
    


    # # # Ramping Limits: (71)-(74)
    # # ts=0     # ramp up
    # m.addConstrs(pj[j,ts]-pj0[j]<=delta_t[ts]*(pru[j]*(j_on[j,ts]-j_su[j,ts])+prusu[j]*(j_su[j,ts]+1-j_on[j,ts])) for j in range(n_j))        
    # m.addConstrs(pj[j,t]-pj[j,t-1]<=delta_t[t]*(pru[j]*(j_on[j,t]-j_su[j,t])+prusu[j]*(j_su[j,t]+1-j_on[j,t])) for j in range(n_j) for t in range(1,n_t))

    # ts=0      # ramp down
    # m.addConstrs(pj[j,ts]-pj0[j]>=-delta_t[ts]*(prd[j]*j_on[j,ts]+prdsd[j]*(1-j_on[j,ts])) for j in range(n_j))          
    # m.addConstrs(pj[j,t]-pj[j,t-1]>=-delta_t[t]*(prd[j]*j_on[j,t]+prdsd[j]*(1-j_on[j,t])) for j in range(n_j) for t in range(1,n_t) )  



    
    # # # T2
    # # # ramping down constraints (T2)
    # B_=m.addVars(n_j,n_t,vtype=GRB.BINARY) 
    # for j in range(n_j):
    #     for t in range(n_t-1):

    #        m.addConstr(pj[j,t]<=pj_min_[j,t]+(1-B_[j,t+1])*(pj_max_[j,t]-pj_min_[j,t]))
    #        M=3
    #        m.addConstr(j_on[j,t]>=j_on[j,t+1]-M*(1-B_[j,t+1]))
    #        m.addConstr(j_on[j,t]<=j_on[j,t+1]-M*B_[j,t+1]) 



    
    ## end <<
    
    
    
      
    obj=gp.quicksum(zen[j,t] for j in range (n_j) for t in range (n_t))-Zp-Zs-Zon-Zsu-Zsd

    m.setObjective(obj, GRB.MAXIMIZE)
    # def mycallback(model, where):
    #     if where == GRB.Callback.MESSAGE:
    #         if 'Variable types' in m.cbGet(GRB.Callback.MSG_STRING):
    #             if '(0 binary)' in m.cbGet(GRB.Callback.MSG_STRING):
    #               m.terminate()
                  
                  
    if warm_st==1:
        # Set warm start values
        for j in range(n_j):
            for t in range(n_t):
                
                if j_on_ub[j,t]!=0:
                    j_on[j, t].Start = 1  # Set initial value to 1
                
                
    def lazy(model, where):
        if where == GRB.Callback.MIPSOL:
            time_elapsed = model.cbGet(GRB.Callback.RUNTIME)
            # print('here we go!')
            # print('here we go!',GRB.Callback.MIPSOL)
            # print(where == GRB.Callback.MIPSOL)
            j_sd1 = model.cbGetSolution(j_sd)
            j_su1 = model.cbGetSolution(j_su)

            # # (56): Minimum downtime
            for j in range(n_j):
                
                down_time_=down_time[j]
                if down_time_>0:
                    T_elap=delta_t[0] # T3
                    # T_elap=0 # T2
                    st_ind=0
        
                    for t in range(1,n_t):
                        et_ind=t
                        
                        if j_su1[j,t]>1-sum(j_sd1[j,t_] for t_ in range (st_ind,et_ind)):
                            m.cbLazy(j_su[j,t]<=1-gp.quicksum(j_sd[j,t_] for t_ in range (st_ind,et_ind)))

                        dt=delta_t[t]
                        T_elap+=dt
        
                        if T_elap>=down_time_:
                            st_ind=st_ind+1
                            #T_elap=0


            # # (57): Minimum uptime
            for j in range(n_j):
                up_time_=up_time[j]
                if up_time_>0:
        
                    T_elap=delta_t[0] # T3
                    # T_elap=0          # T2
                    st_ind=0
                    
                    if j_sd1[j,0]!=0:
                        m.cbLazy(j_sd[j,0]==0)
        
                    for t in range(1,n_t):
                        et_ind=t
                        if j_on_0[0]==1 and st_ind==0:
                            
                            if j_sd1[j,t]>0:
                                m.cbLazy(j_sd[j,t]<=0)
        
                        else: 
                            if j_sd1[j,t]>1-sum(j_su1[j,t_] for t_ in range (st_ind,et_ind)):
                                m.cbLazy(j_sd[j,t]<=1-gp.quicksum(j_su[j,t_] for t_ in range (st_ind,et_ind)))

                        dt=delta_t[t]
                        T_elap+=dt
        
                        if T_elap>=up_time_:
                            st_ind=st_ind+1
                            #T_elap=0
        
            # (58)   
        
            for j in range(n_j):
                startups_j=startups_ub[j]
        
                if len(startups_j)>0:
        
                    for startups in startups_j: # it may have more than one matrix
        
        
                        st=startups[0]
                        et=startups[1]
        
                        for t in range(n_t):
                            if st<T[t]:
                                st_ind=t
                                break
        
                        et_ind=len(T)-1 # In case the end time is higher than the control horizon     
                        for t in range(n_t):
                            if et<=T[t]:
                                et_ind=t
                                break  
        
                        if sum(j_su1[j,t_] for t_ in range(st_ind,et_ind+1))>startups[2]:
                            m.cbLazy(gp.quicksum(j_su[j,t_] for t_ in range(st_ind,et_ind+1))<=startups[2])

         
                

    
    if lazy_const==1:
        m.Params.LazyConstraints = 1          
        m.optimize(lazy)
    else:    
        m.optimize()        






    stop = timeit.default_timer()
    time0=stop - start
    print('Time- mins:',np.round(time0/60,3))
    

    if m.STATUS!=11 and m.STATUS!=3:
        if m.SolCount==0 and div>1:
            sol=0
   
   
    if m.SolCount>=1:
        # >>
        # j_on_=[np.round(j_on[j,t].x,0)  for t in range(n_t) for j in range(n_j)]
        j_on_=[(j_on[j,t].x)  for t in range(n_t) for j in range(n_j)]
        j_on=j_on_

        # # Must Run / Planned Outage
        # # check if device has must-run or planned-outage condiation
        # count=0
        # for j in range(n_j):
        #     for t in range(n_t_):
        #         if j_on_lb[j,t]==j_on_ub[j,t]:
        #             count=count+1     
        #     if count==0:                   # if there is no must-run or planned outage 
        #         continue
        #     for t in range(n_t_):
        #         if j_on_lb[j,t]==j_on_ub[j,t]:
        #             j_on[j+n_j*t]=j_on_lb[j,t]
        #         else:
        #             if t==0:
        #                 j_on[j]=j_on_0[j]
        #             else:
        #                 j_on[j+n_j*t]=j_on[j+n_j*(t-1)]
                        
    if  m.STATUS==11: # fixed
        # j_on_fixed=list(np.ones(n_j)) # all on
        
        # j_on_fixed=[] # set to upper bound
        # for t in range(n_t_):
        #     for j in range (n_j):
        #         j_on_fixed.append(j_on_ub[j,t])
        # j_on=j_on_fixed   
        
        j_on_fixed = np.array(j_on_fixed*n_tt)
        j_on=j_on_fixed
        
        # # Must Run / Planned Outage
        # # check if device has must-run or planned-outage condiation
        # count=0
        # for j in range(n_j):
        #     for t in range(n_t_):
        #         if j_on_lb[j,t]==j_on_ub[j,t]:
        #             count=count+1     
        #     if count==0:                   # if there is no must-run or planned outage 
        #         continue
        #     for t in range(n_t_):
        #         if j_on_lb[j,t]==j_on_ub[j,t]:
        #             j_on[j+n_j*t]=j_on_lb[j,t]
        #         else:
        #             if t==0:
        #                 j_on[j+n_j*t]=j_on_0[j]
        #             else:
        #                 j_on[j+n_j*t]=j_on_[j,t-1]


    if (m.SolCount==0 and div==1) or m.STATUS==3: 

        # all_on if all-were-off
        if sum(j_on_fixed)==0:
            for j in range(n_j):
                if j_on_fixed[j]==0:
                    if pj_min_[j,0]<=delta_t[0]*prusu[j]:
                        j_on_fixed[j]==1
        j_on_fixed = np.array(j_on_fixed*n_tt)
        j_on=j_on_fixed
        
        # Must Run / Planned Outage
        # check if device has must-run or planned-outage condiation
        count=0
        for j in range(n_j):
            for t in range(n_t_):
                if j_on_lb[j,t]==j_on_ub[j,t]:
                    count=count+1     
            if count==0:                   # if there is no must-run or planned outage 
                continue
            for t in range(n_t_):
                if j_on_lb[j,t]==j_on_ub[j,t]:
                    j_on[j+n_j*t]=j_on_lb[j,t]
                else:
                    if t==0:
                        j_on[j+n_j*t]=j_on_0[j]
                    else:
                        j_on[j+n_j*t]=j_on_[j,t-1]


# %%
if switch2==1:
# if n_b>=2000 or sol==0:   # large Networks n_b>=2000     + small networks if gurobi does not converage
    if sol==0:
        Seq_2=1
    for stage in range(2):
        # if stage==1 and (Seq_2!=1): # no need for two-time SW for Mod Seq-Ipopt (Seq_2=2) and One-step Ipopt (Seq_2=0)
        #     continue
        # Model
        m=gp.Model()


        # Assign variables
        theta = m.addVars(n_b, n_t, vtype='C', lb=-np.pi , ub=np.pi)
        pje = m.addVars(n_je, vtype='C', lb=0 , ub=pje_max)

        # Pj can be zero according to linear formulation. lb is pjmin in the quadratic formulation
        # pj = m.addVars(n_j,n_t, vtype='C', lb=pj_min_ , ub=pj_max_) 
        pj = m.addVars(n_j,n_t, vtype='C') 

        j_on = m.addVars(n_j,lb=j_on_lb[:,0], ub=j_on_ub[:,0], vtype=GRB.BINARY) 

        f_on_= np.transpose([f_on_0] * n_t)


        # bounds on power
        ub_=np.max(pj_max_)*n_j
        lb_=-ub_
        # ub_=GRB.INFINITY
        # lb_=-GRB.INFINITY


        # bus power (one_time usage)
        P= m.addVars(n_b, n_t, vtype='C', lb=lb_, ub=ub_)      # Consuming(+)  Producing (-)
        P_fr= m.addVars(n_b, n_t, vtype='C', lb=lb_ , ub=ub_)  # 
        P_to= m.addVars(n_b, n_t, vtype='C', lb=lb_, ub=ub_)   # 
        Pmis= m.addVars(n_b, n_t, vtype='C', lb=lb_ , ub=ub_ ) # 
        Pmis_plus= m.addVars(n_b, n_t, vtype='C', lb=0  , ub=ub_) # 


        # Branch
        P_fr_f = m.addVars(n_f,n_t, vtype='C', lb=lb_ , ub=ub_) 
        P_to_f = m.addVars(n_f,n_t, vtype='C', lb=lb_ , ub=ub_) 



        # Branch Limits
        s_plus= m.addVars(n_f,n_t,vtype='C', lb=0) # (128)

        zen= m.addVars(n_j, n_t, vtype='C', lb=-GRB.INFINITY, ub=GRB.INFINITY) 


        # Absolut value of power mismatch Constraints (4)-(7) (6 and 7 eliminated):

        for i in range(n_b):
            for t in range (n_t):
                m.addConstr(Pmis_plus[i,t]>=Pmis[i,t],name='c0')
                m.addConstr(Pmis_plus[i,t]>=-Pmis[i,t], name='c1')


        # Power Balance Constraint (10):
        # start:

        for t in range (n_t):  
            for i in range(n_b):

                j_at_bus=bus_j_map[i]   # find the devices belongs to this bus       



                # linear version:
                m.addConstr(P[i,t]==gp.quicksum(pj[j,t]*j_sgn_[j] for j in j_at_bus), name='c2' )


                #  P_fr
                f_fr=fr_bus[i] # devices from bus that belongs to this bus
                # P_fr[i,t]=0
                for f in f_fr:
                    if f not in dc_f:
                        theta_index=fr_to[f]
                        m.addConstr(P_fr_f[f,t]==Bf[f]*(theta[theta_index[0],t]-theta[theta_index[1],t]), name='c3')


                        # Constraint (130): Branch flow limits
                        # linear
                        m.addConstr(P_fr_f[f,t]<=s_max[f]+s_plus[f,t])


                m.addConstr(P_fr[i,t]==gp.quicksum(P_fr_f[f,t]*f_on_[f,t] for f in f_fr) )



                #  P_to
                f_to=to_bus[i]
                # P_to[i,t]=0
                for f in f_to:
                    if f not in dc_f:
                        theta_index=fr_to[f]
                        m.addConstr(P_to_f[f,t]==-Bf[f]*(theta[theta_index[0],t]-theta[theta_index[1],t]), name='c5')



                        # Constraint (131): Branch flow limits                
                        # linear
                        m.addConstr(P_to_f[f,t]<=s_max[f]+s_plus[f,t])


                m.addConstr(P_to[i,t]==gp.quicksum(P_to_f[f,t]*f_on_[f,t] for f in f_to) )



        # no shunt power
        m.addConstrs((P[i,t]+P_fr[i,t]+P_to[i,t]==Pmis[i,t] for i in range(n_b) for t in range (n_t)), name='c9')




        # end of constraint (10)


        # linear-version trick (slide 5) to remove the quadratic terms:

        for j in range(n_j):
            for t in range(n_t):
                
                if j_on_lb[j][t]<j_on_ub[j][t]:
                    m.addConstr(pj[j,t]>=j_on[j]*pj_min_[j,t])
                    # m.addConstr(pj[j,t]<=j_on[j]*pj_max_[j,t])
                    if stage==0:
                        m.addConstr(pj[j,t]<=j_on[j]*pj_max_[j,t])
                    else:
                        m.addConstr(pj[j,t]<=j_on[j]*np.max(pj_max_all_[j]))

                
                else:
                    m.addConstr(pj[j,t]>=j_on_ub[j][t]*pj_min_[j,t])
                    m.addConstr(pj[j,t]<=j_on_ub[j][t]*pj_max_[j,t])


        # # v2=v^2
        # m.addConstrs((v2[i,t]==v[i,t]*v[i,t]   for i in range(n_b) for t in range (n_t)), name='c10')




        # pj>=pj_min_: that is can be enforced by pj lower bound (lb=pj_min_) 
        # Pj[j,t]=sum(pje)
        for t in range(n_t):
            for j in range (n_j):  

                indexx=j_je[j,t] # find pje indices  of one device J at given time t

                m.addConstr(pj[j,t]==gp.quicksum(pje[index] for index in indexx), name='c11')




        # (142)-(146) 
        # Dc line limits
        for f in dc_f:
            for t in range(n_t):

                m.addConstr(P_fr_f[f,t]<=pdc_max_[f])   #(142)
                m.addConstr(P_fr_f[f,t]>=-pdc_max_[f])

                m.addConstr(P_to_f[f,t]<=pdc_max_[f])   #(144)
                m.addConstr(P_to_f[f,t]>=-pdc_max_[f])

                m.addConstr(P_fr_f[f,t]+P_to_f[f,t]==0)   #(146)





        # on-off, startup, shoutdown
        # (42) & (43)
        j_su = m.addVars(n_j,vtype=GRB.BINARY)   
        j_sd = m.addVars(n_j,vtype=GRB.BINARY) 




        # # (46)-(47)
        for j in range (n_j):
            t=0
            m.addConstr(j_on[j]-j_on_0[j]==j_su[j]-j_sd[j])
            #for t in range (1,n_t):
            #    m.addConstr(j_on[j,t]-j_on[j,t-1]==j_su[j,t]-j_sd[j,t], name='p1')




        ## (48)
        for t in range(n_t):
            for j in range (n_j):
                m.addConstr(j_su[j]+j_sd[j]<=1)




        # on-off startup shutdown costs
        # (49),(50),(51)

        Zon=m.addVar(vtype='C', lb=0) # Cost function        
        m.addConstr(Zon==gp.quicksum(delta_t[t]*on_cost[j]*j_on[j] if j_on_lb[j][t]<j_on_ub[j][t] else delta_t[t]*on_cost[j]*j_on_ub[j][t]    for t in range(n_t) for j in range (n_j)   ))


        Zsu=m.addVar(vtype='C', lb=0) # Cost function  

        m.addConstr(Zsu==gp.quicksum(su_cost[j]*j_su[j]  for j in range (n_j)))



        Zsd=m.addVar(vtype='C', lb=0) # Cost function

        m.addConstr(Zsd==gp.quicksum(sd_cost[j]*j_sd[j]  for j in range (n_j)))




#         # # (56)-(58):

#         T=np.zeros(n_t)

#         for t in range(n_t):

#             if t>=1:
#                 T[t]=T[t-1]+delta_t[t]
#             else:
#                 T[t]=delta_t[t]

#         T=np.zeros(n_t_)

#         for t in range(n_t_):

#             if t>=1:
#                 T[t]=T[t-1]+delta_t_[t]
#             else:
#                 T[t]=delta_t_[t]



        # # (56)-(58):
        T=np.zeros(n_t)

        for t in range(n_t):

            if t>=1:
                T[t]=T[t-1]+delta_t_[t]
            else:
                T[t]=delta_t_[t]



        if stage==0:
             # # (57): Minimum uptime
            for j in range(n_j):
                up_time_=up_time[j]
                if up_time_>0:
                    if j_on_0[j]==1:
                        m.addConstr(j_on[j]==1)
     

        # Costs & Objective

        # zen=0  # producing & consuming
        # tt=0
        for j in range (n_j):
            for t in range (n_t):
                indexx=j_je[j,t]
                m.addConstr(zen[j,t]==gp.quicksum(c_en[index]*pje[index]*delta_tj[index] for index in indexx), name='c12')



        # (8) bus power violation/mismatch cost
        cp=data['network']['violation_cost']['p_bus_vio_cost']
        Zp=m.addVar(vtype='C', lb=0) # Cost function
        m.addConstr(Zp==gp.quicksum(delta_t[t]*cp*Pmis_plus[i,t] for i in range (n_b) for t in range (n_t)))
        # Zp=0

        # (129) Branch limit penalty
        cs=data['network']['violation_cost']['s_vio_cost']   
        Zs=m.addVar(vtype='C', lb=0) # Cost function
        m.addConstr(Zs==gp.quicksum(delta_t_[t]*cs*s_plus[f,t] for f in range (n_f) for t in range (n_t)))




        # ## Section 2.6.3  Maximum/Minimum Energy Over intervals
        # (68)-(69)
        for j in range(n_j):

            Wen_max_j=Wen_max[j]

            if len(Wen_max_j)>0:

                for Wen_max_ in Wen_max_j:  # it might have more than one device

                    st=Wen_max_[0]
                    et=Wen_max_[1]

                    for t in range(n_t):
                        if st<T[t]:
                            st_ind=t
                            break
                    et_ind=len(T)-1 # In case the end time is higher than the control horizon                
                    for t in range(n_t):
                        if et<=T[t]:
                            et_ind=t
                            break        


                    m.addConstr(gp.quicksum(delta_t_[t]*pj[j,t] for t in range(st_ind+1,et_ind))+(T[st_ind]-st)*pj[j,st_ind]+(et-T[et_ind-1])*pj[j,et_ind]<=Wen_max_[2])


            Wen_min_j=Wen_min[j]

            if len(Wen_min_j)>0:
                for Wen_min_ in Wen_min_j:  # it might have more than one device


                    st=Wen_min_[0]
                    et=Wen_min_[1]

                    for t in range(n_t):
                        if st<T[t]:
                            st_ind=t
                            break
                    et_ind=len(T)-1 # In case the end time is higher than the control horizon                  

                    for t in range(n_t):
                        if et<=T[t]:
                            et_ind=t
                            break 

                    m.addConstr(gp.quicksum(delta_t_[t]*pj[j,t] for t in range(st_ind+1,et_ind))+pj[j,st_ind]*(T[st_ind]-st)+pj[j,et_ind]*(et-T[et_ind-1])>=Wen_min_[2])

        
        # ramping down constraints at t=0 (additional constraint)   
        for j in range(n_j):
            if pj_min_[j,0]>0:
                m.addConstr(float(pj0[j])<=pj_min_[j,0]+(1-j_sd[j])*pj0[j])

        
        # # Ramping Limits: (71)-(74)
        
        # ramp up
        ts=0 
        # if stage==0:
        m.addConstrs(pj[j,ts]-pj0[j]<=delta_t[ts]*(pru[j]*(j_on[j]-j_su[j])+prusu[j]*(j_su[j]+1-j_on[j]))  for j in range(n_j)) 
        # else:
        #     m.addConstrs(pj[j,ts]<=delta_t[ts]*(pru[j]*(j_on[j]-j_su[j])+prusu[j]*(j_su[j]+1-j_on[j]))  for j in range(n_j)) 

        # ramp down
        for j in range(n_j):
            if pj_min_[j,0]>0:
                m.addConstr(pj[j,ts]-pj0[j]>=-delta_t[ts]*(prd[j]*j_on[j]+prdsd[j]*(1-j_on[j])))             

        
        # duplicated 1
        # # ramping down constraints at t=0 (additional constraint)   
        # for j in range(n_j):
        #     if pj_min_[j,0]>0:
        #         m.addConstr(float(pj0[j])<=pj_min_[j,0]+(1-j_sd[j])*pj0[j])

        
        # # Ramping Limits: (71)-(74)
        # ramp up
        ts=0   
#         m.addConstrs(pj[j,ts]-pj0[j]<=delta_t[ts]*(pru[j]*(j_on[j]-j_su[j])+prusu[j]*(j_su[j]+1-j_on[j]))  for j in range(n_j)) 


        # duplicated
        # # ramp down
        # for j in range(n_j):
        #     if pj_min_[j,0]>0:
        #         m.addConstr(pj[j,ts]-pj0[j]>=-delta_t[ts]*(prd[j]*j_on[j]+prdsd[j]*(1-j_on[j])))               
            
            # m.addConstr(float(pj0[j])<=pj_min_[j,0]+(1-j_sd[j,0])*GRB.INFINITY)
            # m.addConstr(float(pj0[j])<=pj_min_[j,0]+(1-j_sd[j,0])*pj_max_[j,0])

        
#         # # Ramping Limits: (71)-(74)
#         ts=0     # ramp up
#         m.addConstrs(pj[j,ts]-pj0[j]<=delta_t[ts]*(pru[j]*(j_on[j]-j_su[j])+prusu[j]*(j_su[j]+1-j_on[j])) for j in range(n_j))        
#         # ramp down
#         m.addConstrs(pj[j,ts]-pj0[j]>=-delta_t[ts]*(prd[j]*j_on[j]+prdsd[j]*(1-j_on[j])) for j in range(n_j))   
        

        
        obj=gp.quicksum(zen[j,t] for j in range (n_j) for t in range (n_t))-Zp-Zs-Zon-Zsu-Zsd

        m.setObjective(obj, GRB.MAXIMIZE)
        
        # m.optimize()
        
        def mycallback(model, where):
            if where == GRB.Callback.MESSAGE:
                if 'Variable types' in m.cbGet(GRB.Callback.MSG_STRING):
                    if '(0 binary)' in m.cbGet(GRB.Callback.MSG_STRING):
                      m.terminate()   
        m.optimize(mycallback)





        stop = timeit.default_timer()
        time0=stop - start
        print('Time in mins- gurobi',' of stage ', stage, ': ',time0/60)








        # End of MIP Model   

        # >>
        # match with NLP code  
        # j_on=  [np.round(j_on[j].x,0)   for t in range(n_t) for j in range(n_j)]

        
        # must-run planned-outage condition
        # n_t_=1 # temporary

        
        if m.STATUS==11 or m.SolCount==0: # initial condition are optimal, fixed
            # j_on=j_on_fixed
            if stage==0:
                if m.STATUS==11:
                    j_on_1=j_on_fixed
                else: 
                    j_on_1=j_on_fixed
                    # all on if all were off
                    if sum(j_on_fixed)==0:
                        for j in range(n_j):
                            if j_on_1[j]==0:
                                if pj_min_[j,0]<=delta_t[0]*prusu[j]:
                                    j_on_1[j]==1
                    
            else:
                if m.STATUS==11:
                    j_on_2=j_on_fixed
                else: # m.SolCount==0
                    j_on_2=j_on_1
        else:    
            if stage==0:
                j_on_1=[np.round(j_on[j].x,0) for j in range(n_j)]
            else:
                j_on_2=[np.round(j_on[j].x,0) for j in range(n_j)]
   


    
    # match with NLP code + must-run or planned-outage condiation   +  ramping down limit

    # T=np.zeros(n_t_)
    # if Seq_2==1:                     # Seq-Ipopt one-step then 2 times SW
    #     j_on_= np.array([j_on_1]*n_t_)
    #     j_on_=j_on_.T
        
    #     for j in range(n_j):
    #         up_time_=up_time[j]
    #         # down_time_=down_time[j]

    #         # check if device has must-run or planned-outage condiation
    #         count=0
    #         for t in range(n_t_):
    #             if j_on_lb[j,t]==j_on_ub[j,t]:
    #                 count=count+1     
    #         if count==0:                   # if there is no must-run or planned outage 
    #             if div>1:
    #                 for t in range(n_t_-1):
    #                     if t>=1:
    #                         T[t]=T[t-1]+delta_t_[t]
    #                     else:
    #                         T[t]=delta_t_[t]
                            
    #                     if j_on_2[j]<j_on_1[j]: # if shutting down check A- min-up-time AND B- the ramping down limit
    #                         if T[t]>=up_time_: #  switch after min-up-time is elapsed
    #                             if pj0[j]-np.sum([delta_t_[t_loc]*prd[j] for t_loc in range(t+1)])<=pj_min_all_[j,t]: 
    #                                 j_on_[j,t+1]=j_on_2[j]
                                    
    #                     # if j_on_2[j]>j_on_1[j]: # if starting up down check  min-down-time 
    #                     #     if T[t]>=down_time_ or j_sd[j].x==0: #  switch after min-down-time is elapsed
    #                     #             j_on_[j,t+1]=j_on_2[j]                                    

    #             continue

            
    #         for t in range(n_t_):
    #             if j_on_lb[j,t]==j_on_ub[j,t]:
    #                 j_on_[j,t]=j_on_lb[j,t]
    #             else:
    #                 if t==0:
    #                     j_on_[j,t]=j_on_0[j]
    #                 else:
    #                     j_on_[j,t]=j_on_[j,t-1]

    #     # match with NLP code        
    #     j_on=[j_on_[j,t] for t in range(n_t_) for j in range(n_j)] 
        
    # else:                    # one-step Ipopt | Modified Seq-Ipopt
             
    #     # j_on=  [np.round(j_on[j].x,0)   for t in range(n_t) for j in range(n_j)] 
        
    #     if Seq_2==2:     # Modified Seq-Ipopt. The SW are fixed to their status at t=0 but duplicated
    #         j_on=  [j_on_1[j]   for t in range(n_t_) for j in range(n_j)]
    #     else: # one-step Ipopt
    #         j_on=j_on_1
    

#print(count)
#print('Number of constraints removed by lazy',c1)
print(m.objval)
import matplotlib.pyplot as plt
x_values = np.arange(len(j_on))  # Create x-values as indices
plt.scatter(x_values, j_on, s=5)  # Scatter plot with generated x-values
plt.show()