# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 04:00:19 2025

@author: hussein.sharadga
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 23:07:57 2025

@author: hussein.sharadga
"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    "Cost Blocks": ["Tighter Blocks"] * 6 + ["Original Blocks"] * 6,
    "Time [mins]": [1.217, 1.101, 1.325, 1.293, 1.261, 1.184, 
              1.901, 1.705, 1.326, 2.457, 2.135, 1.366]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
# Set theme
sns.set_style("whitegrid")
sns.boxplot(x="Cost Blocks", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
#sns.stripplot(x="Type", y="Value", data=df, hue='Type', size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], inner="box", linewidth=1)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alpha=.6)
# plt.subplots_adjust(hspace=0.3)
# plt.title("Distribution of Tighter vs. Original Blocks")

# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Cost Blocks")["Time [mins]"].mean().reset_index()
# Overlay mean values using a point plot
sns.pointplot(x="Cost Blocks", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img1.png", dpi=300, bbox_inches="tight")

plt.show()


# %%
# # # Data preparation
# data = {
#     "Cost Blocks": ["Tighter Blocks"] * 7 + ["Original Blocks"] * 7,
#     "Time [mins]": [1.217, 1.101, 1.325, 1.293, 1.261, 1.184,2.645, 
#               1.901, 1.705, 1.326, 2.457, 2.135, 1.366, 1.66]
# }

# df = pd.DataFrame(data)

# # Box Plot with Swarm Plot
# plt.figure(figsize=(3, 5))
# sns.boxplot(x="Cost Blocks", y="Time [mins]", data=df, width=0.5, palette=["blue", "black"])
# #sns.stripplot(x="Type", y="Value", data=df, hue='Type', size=7)

# # plt.title("Distribution of Tighter vs. Original Blocks")
# plt.grid(axis="y", linestyle="--", alpha=0.6)
# plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation with the new values
data = {
    "Cost Blocks": ["Tighter Blocks"] * 15 + ["Original Blocks"] * 15,
    "Time [mins]": [
        1.741, 1.329, 1.459, 1.284, 1.259, 2.774, 1.945, 2.301, 1.767, 1.683, 1.552, 1.407, 1.253, 1.349, 1.841,  # Tighter Blocks
        7.617, 1.491, 2.049, 2.54, 2.698, 2.314, 2.728, 2.521, 3.902, 2.217, 2.487, 2.614, 1.789, 1.656, 1.489   # Original Blocks
    ]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Cost Blocks", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})




# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Cost Blocks")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Cost Blocks", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img2.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Data preparation with the new values
# data = {
#     "Cost Blocks": ["Tighter Blocks"] * 15 + ["Original Blocks"] * 15,
#     "Time [mins]": [
#         1.741, 1.329, 1.459, 1.284, 1.259, 2.774, 1.945, 2.301, 1.767, 1.683, 1.552, 1.407, 1.253, 1.349, 1.841,  # Tighter Blocks
#         7.617, 1.491, 2.049, 2.54, 2.698, 2.314, 2.728, 2.521, 3.902, 2.217, 2.487, 2.614, 1.789, 1.656, 1.489   # Original Blocks
#     ]
# }

# df = pd.DataFrame(data)

# # Calculate the average (mean) for each group
# mean_values = df.groupby("Cost Blocks")["Time [mins]"].mean()

# # Box Plot
# plt.figure(figsize=(6, 5))
# sns.boxplot(x="Cost Blocks", y="Time [mins]", data=df, width=0.5, palette=["blue", "black"])

# # Overlay the mean values
# for i, mean_value in enumerate(mean_values):
#     plt.text(i, mean_value, f'{mean_value:.2f}', horizontalalignment='center', color='red', weight='bold', fontsize=12)

# # Add grid
# plt.grid(axis="y", linestyle="--", alpha=0.6)

# # Show plot
# plt.title("Boxplot with Average (Mean) Values")
# plt.show()












# %%



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    "Ramping Constraints": ["Eliminated"] * 6 + ["Included"] * 6,
    "Time [mins]": [1.217, 1.101, 1.325, 1.293, 1.261, 1.184, 
                     0.794, 0.743, 0.863, 0.731, 0.775, 0.841]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Ramping Constraints", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
#sns.stripplot(x="Type", y="Value", data=df, hue='Type', size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], inner="box", linewidth=1)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alpha=.6)
# plt.subplots_adjust(hspace=0.3)
# plt.title("Distribution of Tighter vs. Original Blocks")
# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Ramping Constraints")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Ramping Constraints", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img3.png", dpi=300, bbox_inches="tight")
plt.show()






# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation with the new values
data = {
    "Ramping Constraints": ["Eliminated"] * 15 + ["Included"] * 15,
    "Time [mins]": [
        1.741, 1.329, 1.459, 1.284, 1.259, 2.774, 1.945, 2.301, 1.767, 1.683, 1.552, 1.407, 1.253, 1.349, 1.841,  # Eliminated
        1.648, 2.625, 2.961, 2.729, 2.722, 2.944, 4.069, 2.364, 3.806, 2.455, 2.709, 4.672, 2.203, 1.903, 2.157   # Included
    ]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Ramping Constraints", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})




# Uncomment the next line if you want to add a swarm plot on top of the boxplot
# sns.stripplot(x="Cost Blocks", y="Time [mins]", data=df, hue="Cost Blocks", size=7)
# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Ramping Constraints")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Ramping Constraints", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img4.png", dpi=300, bbox_inches="tight")
plt.show()




# %%


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    "Switch": ["Single"] * 7 + ["Multiple"] * 7,
    "Time [mins]": [1.217, 1.101, 1.325, 1.293, 1.261, 1.184, 
                    0.903, 0.796, 0.777, 1.034, 1.161, 1.092, 
                    2.645, 3.218]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Switch", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
# sns.stripplot(x="Cost Blocks", y="Time [mins]", data=df, hue='Cost Blocks', size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], inner="box", linewidth=1)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alph
# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Switch")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Switch", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img5.png", dpi=300, bbox_inches="tight")
plt.show()



# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation with the new values
data = {
    "Switch": ["Single"] * 15 + ["Multiple"] * 15,
    "Time [mins]": [
        1.741, 1.329, 1.459, 1.284, 1.259, 2.774, 1.945, 2.301, 1.767, 1.683, 1.552, 1.407, 1.253, 1.349, 1.841,  # Eliminated
        1.933, 1.921, 1.794, 1.611, 1.386, 1.316, 1.787, 1.721, 1.551, 2.362, 1.689, 1.639, 1.82, 1.791, 2.134   # Included
    ]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Switch", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})

# Uncomment the next line if you want to add a swarm plot on top of the boxplot
# sns.stripplot(x="Ramping Constraints", y="Time [mins]", data=df, hue="Ramping Constraints", size=7)
# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Switch")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Switch", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img6.png", dpi=300, bbox_inches="tight")
plt.show()




# %%


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    "Switch": ["Single"] * 7 + ["Multiple"] * 7,
    "Time [mins]": [2.716, 3.17, 1.186, 1.498, 1.401, 1.524, 1.23, 
                    0.955, 0.989, 1.181, 1.023, 0.94, 1.224, 4.096]
}


df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Switch", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
# sns.stripplot(x="Cost Blocks", y="Time [mins]", data=df, hue='Cost Blocks', size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], inner="box", linewidth=1)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alph
# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Switch")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Switch", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img7.png", dpi=300, bbox_inches="tight")
plt.show()



# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation with the new values
data = {
    "Switch": ["Single"] * 15 + ["Multiple"] * 15,
    "Time [mins]": [
        1.935, 2.243, 3.524, 2.787, 2.058, 2.037, 2.306, 2.608, 3.005, 1.629,
        1.884, 1.962, 7.347, 2.838, 8,  
        4.56, 35.132, 6.303, 5.777, 3.319, 34.742, 10.407, 7.427, 8.327, 8.331,
        7.674, 11.483, 52.505, 3.311, 4.923  
    ]
}
df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Switch", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
# sns.violinplot(x="Switch", y="Time [mins]", data=df, linewidth=1.5, cut=0)
# sns.violinplot(x="Switch", y="Time [mins]", data=df, linewidth=1.5, cut=0)

# Uncomment the next line if you want to add a swarm plot on top of the boxplot
# sns.stripplot(x="Ramping Constraints", y="Time [mins]", data=df, hue="Ramping Constraints", size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alpha=.6)

# Set y-axis ticks every 5 units
plt.yticks(range(0, int(df["Time [mins]"].max()) + 3, 3))

# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Switch")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Switch", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img8.png", dpi=300, bbox_inches="tight")
plt.show()

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
# Data preparation
data = {
    "Constraints": ["Tight"] * 7 + ["Original"] * 7,
    "Time [mins]": [1.156, 1.122, 1.151, 1.116, 1.149, 1.358, 2.566, 
                    1.217, 1.101, 1.325, 1.293, 1.261, 1.184, 2.645]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Constraints", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
# sns.stripplot(x="Cost Blocks", y="Time [mins]", data=df, hue='Cost Blocks', size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], inner="box", linewidth=1)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alph
# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Constraints")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Constraints", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img9.png", dpi=300, bbox_inches="tight")
plt.show()



# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation with the new values
data = {
    "Constraints": ["Tight"] * 15 + ["Original"] * 15,
    "Time [mins]": [
        1.246, 1.42, 1.517, 1.378, 1.329, 1.29, 1.357, 1.309, 2.559, 1.332, 
        1.411, 2.327, 1.113, 1.455, 1.37, 1.741, 1.329, 1.459, 1.284, 1.259, 
        2.774, 1.945, 2.301, 1.767, 1.683, 1.552, 1.407, 1.253, 1.349, 1.841
    ]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Constraints", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
# sns.violinplot(x="Switch", y="Time [mins]", data=df, linewidth=1.5, cut=0)
# sns.violinplot(x="Switch", y="Time [mins]", data=df, linewidth=1.5, cut=0)

# Uncomment the next line if you want to add a swarm plot on top of the boxplot
# sns.stripplot(x="Ramping Constraints", y="Time [mins]", data=df, hue="Ramping Constraints", size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alpha=.6)

# # Set y-axis ticks every 5 units
# plt.yticks(range(0, int(df["Time [mins]"].max()) + 3, 3))

# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Constraints")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Constraints", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img10.png", dpi=300, bbox_inches="tight")
plt.show()


# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
# Data preparation
data = {
    "Constraints": ["Lazy"] * 7 + ["Original"] * 7,
    "Time [mins]": [0.867, 0.712, 0.528, 0.674, 0.651, 0.527, 1.515, 
                    1.217, 1.101, 1.325, 1.293, 1.261, 1.184, 2.645]
}

df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Constraints", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
# sns.stripplot(x="Cost Blocks", y="Time [mins]", data=df, hue='Cost Blocks', size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], inner="box", linewidth=1)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alph
# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Constraints")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Constraints", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img11.png", dpi=300, bbox_inches="tight")
plt.show()



# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation with the new values
data = {
    "Constraints": ["Lazy"] * 15 + ["Original"] * 15,
    "Time [mins]": [
        1.402, 1.453, 1.367, 1.393, 1.372, 1.482, 1.556, 1.356, 2.243, 1.596, 
        1.479, 1.549, 1.957, 1.396, 1.431, 1.741, 1.329, 1.459, 1.284, 1.259, 
        2.774, 1.945, 2.301, 1.767, 1.683, 1.552, 1.407, 1.253, 1.349, 1.841
    ]
}
df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="Constraints", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
# sns.violinplot(x="Switch", y="Time [mins]", data=df, linewidth=1.5, cut=0)
# sns.violinplot(x="Switch", y="Time [mins]", data=df, linewidth=1.5, cut=0)

# Uncomment the next line if you want to add a swarm plot on top of the boxplot
# sns.stripplot(x="Ramping Constraints", y="Time [mins]", data=df, hue="Ramping Constraints", size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alpha=.6)

# # Set y-axis ticks every 5 units
# plt.yticks(range(0, int(df["Time [mins]"].max()) + 3, 3))

# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("Constraints")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="Constraints", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black


plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img12.png", dpi=300, bbox_inches="tight")
plt.show()


# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    "J^{su}/J^{sd}": ["Relaxed"] * 6 + ["Binary"] * 6,
    "Time [mins]": [
        0.531, 0.598, 0.84, 0.759, 0.77, 0.833,
        1.217, 1.101, 1.325, 1.293, 1.261, 1.184
    ]
}

# # Data preparation
# data = {
#     "J^{su}/J^{sd}": ["Relaxed"] * 7 + ["Binary"] * 7,
#     "Time [mins]": [
#         0.531, 0.598, 0.84, 0.759, 0.77, 0.833, 1.018,
#         1.217, 1.101, 1.325, 1.293, 1.261, 1.184, 2.645
#     ]
# }


df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
# Set theme
sns.set_style("whitegrid")
sns.boxplot(x="J^{su}/J^{sd}", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})

# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black

# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("J^{su}/J^{sd}")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="J^{su}/J^{sd}", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Set x-axis label with superscripts
ax.set_xlabel("$u_{jt}^{su}/u_{jt}^{sd}$")
plt.grid(axis="y", linestyle="-", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img13.png", dpi=300, bbox_inches="tight")
plt.show()

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation with the new values
data = {
    "J^{su}/J^{sd}": ["Relaxed"] * 15 + ["Binary"] * 15,
    "Time [mins]": [
        1.606, 1.344, 1.442, 1.327, 1.289, 1.414, 1.878, 1.286, 1.338, 1.806, 
        1.39, 1.339, 1.356, 1.366, 1.359,  # Relaxed values
        1.741, 1.329, 1.459, 1.284, 1.259, 2.774, 1.945, 2.301, 1.767, 1.683, 
        1.552, 1.407, 1.253, 1.349, 1.841   # Binary values
    ]
}
df = pd.DataFrame(data)

# Box Plot with Swarm Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x="J^{su}/J^{sd}", y="Time [mins]", data=df, width=0.3, linewidth=2, boxprops={'facecolor': 'none'})
# sns.violinplot(x="Switch", y="Time [mins]", data=df, linewidth=1.5, cut=0)
# sns.violinplot(x="Switch", y="Time [mins]", data=df, linewidth=1.5, cut=0)

# Uncomment the next line if you want to add a swarm plot on top of the boxplot
# sns.stripplot(x="Ramping Constraints", y="Time [mins]", data=df, hue="Ramping Constraints", size=7)
# sns.violinplot(x="Cost Blocks", y="Time [mins]", data=df, palette=["blue", "black"], linewidth=1.5, cut=0, alpha=.6)

# # Set y-axis ticks every 5 units
# plt.yticks(range(0, int(df["Time [mins]"].max()) + 3, 3))

# Set the color of the axes spines (borders)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
# Set the color of the ticks
ax.tick_params(axis='both', colors='black')  # Change tick color to black

# Compute the mean values
means = df.groupby("J^{su}/J^{sd}")["Time [mins]"].mean().reset_index()

# Overlay mean values using a point plot
sns.pointplot(x="J^{su}/J^{sd}", y="Time [mins]", data=means, join=False, 
              color='red', markers="x", scale=1)

# Customize plot aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Change spine color to black
ax.tick_params(axis='both', colors='black')  # Change tick color to black

ax.set_xlabel("$u_{jt}^{su}/u_{jt}^{sd}$")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(r"C:\Users\hussein.sharadga\Desktop\OptX\TPEC Paper\Pics_mean\img14.png", dpi=300, bbox_inches="tight")
plt.show()