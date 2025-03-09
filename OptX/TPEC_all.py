# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 01:54:10 2025

@author: hussein.sharadga
"""
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



# %%
# 1576 
# E2
sc=[21, 22, 23, 31, 32, 33]
for s in sc:
    print('-------')
    print('1576-'+str(s))
    src_data = r'C:\Users\hussein.sharadga\Desktop\OptX\grids\C3E2D1_20230510\D1\C3E2N01576D1\scenario_0'+str(s)+'.json'
    exec(open("TPEC.py").read())

# # E3
src_data = r'C:\Users\hussein.sharadga\Desktop\OptX\grids\C3E3.1D1_20240606\E3.1\D1\C3E3N01576D1\scenario_027.json'
print('-------')
exec(open("TPEC.py").read())


# # # # %%
# # # # 4224 
sc=[31, 32, 33, 36, 37, 38, 41, 42, 43, 46, 47, 48]

for s in sc:
    src_data = r'C:\Users\hussein.sharadga\Desktop\OptX\grids\C3E3.1D1_20240606\E3.1\D1\C3E3N04224D1\scenario_1'+str(s)+'.json'
    print('-------')
    print('4224-'+str(s))
    exec(open("TPEC.py").read())


# sc=[22, 32, 33]
for s in sc:
    src_data=r'C:\Users\hussein.sharadga\Desktop\OptX\grids\C3E2D1_20230510\D1\C3E2N04224D1\scenario_0'+str(s)+'.json'
    print('-------')
    print('4224-'+str(s))
    exec(open("TPEC.py").read())









#execfile("TPEC.py")
# import TPEC




# # %%
# # 6049
# sc=[31, 32, 33, 41, 42, 43]

# for s in sc:
#     src_data = r'C:\Users\hussein.sharadga\Desktop\OptX\grids\C3E3.1D1_20240606\E3.1\D1\C3E3N06049D1\scenario_0'+str(s)+'.json'
#     print('-------')
#     print('6049-'+str(s))
#     exec(open("TPEC.py").read())

# import matplotlib.pyplot as plt
# x_values = np.arange(len(j_on))  # Create x-values as indices
# plt.scatter(x_values, j_on, s=5)  # Scatter plot with generated x-values
# plt.show()
