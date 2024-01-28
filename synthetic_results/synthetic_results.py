import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.style.use('fivethirtyeight')

# Define a palette
colors = sns.color_palette("deep", 5)

############
## Read data
############

df = pd.read_table("synthetic_results.txt", delimiter=' ', header=None, )
df.fillna(0, inplace=True)
data = df.values[:,0:10]

print(data[0:3,:])
max_tasks = 50
n_tasks = np.arange(5,max_tasks+1)

############################################
## Plot number of found harmonic assignments
############################################

harmonic = data[:,2].reshape(max_tasks-4,1000)
harmonic = np.sum(harmonic, axis=1)

font = {'size'   : 13}
plt.rc('font', **font)

fig, ax1 = plt.subplots(figsize=(4.6, 2.8))

ax1.plot(n_tasks, harmonic, "-", color=colors[0], linewidth=2)
ax1.set_ylim(-50,1050)
ax1.set_xticks(np.arange(5,max_tasks+1,5))

ax1.set_xlabel("# Tasks")
ylabel = ax1.set_ylabel("# Harmonic Assign.")
label_position = ylabel.get_position()
ylabel.set_position((label_position[0], label_position[1] - 0.1))

plt.tight_layout()
name = "synthetic_num_harmonic"
plt.savefig(f"{name}.eps",bbox_inches='tight')
plt.savefig(f"{name}.png",bbox_inches='tight')

#####################
## Create data series
#####################

n_projections = data[:,3].reshape(max_tasks-4,1000)
n_projections = np.max(n_projections, axis=1)

harmonic_times = data[:,4].reshape(max_tasks-4,1000)
harmonic_times = np.max(harmonic_times, axis=1)

n_phis = data[:,5].reshape(max_tasks-4,1000)
n_phis = np.max(n_phis, axis=1)

n_regions = data[:,6].reshape(max_tasks-4,1000)
n_regions = np.max(n_regions, axis=1)

elastic_times_generate = data[:,7].reshape(max_tasks-4,1000)
elastic_times_generate = np.max(elastic_times_generate, axis=1)

elastic_times_fast = data[:,8].reshape(max_tasks-4,1000) #ns
elastic_times_fast = np.max(elastic_times_fast, axis=1)

elastic_times_naive = data[:,9].reshape(max_tasks-4,1000)
elastic_times_naive = np.max(elastic_times_naive, axis=1)

ms = 1000
s = 1000000

###########################################
## Projected Harmonic Zones and Search Time
###########################################

font = {'size'   : 13}
plt.rc('font', **font)

fig, ax1 = plt.subplots(figsize=(4.6, 2.8))

ax1.set_xlabel("# Tasks")
ax1.set_xticks(np.arange(5,max_tasks+1,5))

color = colors[0]
ax1.set_ylabel("Max Zones", color=color)
ax1.plot(n_tasks, n_projections, "-", color=color, linewidth=2)
ax1.set_ylim(0,np.max(n_projections)*1.05)

ax2 = ax1.twinx()

color = colors[1]
ax2.set_ylabel("Max Time (ms)", color=color)
ax2.plot(n_tasks, harmonic_times/ms, "--", color=color, linewidth=2)
ax2.set_ylim(0,np.max(harmonic_times/ms)*1.05)

plt.tight_layout()
name = "synthetic_projected_harmonic_zones"
plt.savefig(f"{name}.eps",bbox_inches='tight')
plt.savefig(f"{name}.png",bbox_inches='tight')

##################################################################
## Correlation between PHIs, generate time, and naive search time
##################################################################

font = {'size'   : 13}
plt.rc('font', **font)

fig, ax1 = plt.subplots(figsize=(4.6, 2.8))

ax1.set_xlabel("# Tasks")
ax1.set_xticks(np.arange(5,max_tasks+1,5))

color = colors[0]
ax1.set_ylabel("Max PHIs", color=color)
ax1.plot(n_tasks, n_phis, "-", color=color, linewidth=2)
ax1.set_ylim(0,np.max(n_phis)*1.05)

ax2 = ax1.twinx()

color = colors[1]
ax2.set_ylabel("Max Time", color=color)
ax2.plot(n_tasks, elastic_times_generate/s, "--", color=color, label="Gen LUT (s)", linewidth=2)
ax2.plot(n_tasks, elastic_times_naive/ms, "-.", color='green', label="Naive Search (ms)", linewidth=2)
ax2.set_ylim(0,np.max(elastic_times_generate/s)*1.05)

plt.legend(loc='upper left', framealpha=1.0)

plt.tight_layout()
name = "synthetic_projected_harmonic_intervals"
plt.savefig(f"{name}.eps",bbox_inches='tight')
plt.savefig(f"{name}.png",bbox_inches='tight')

###################################
## Lookup Table Size vs Search Time
###################################

font = {'size'   : 13}
plt.rc('font', **font)

fig, ax1 = plt.subplots(figsize=(4.6, 2.8))

ax1.set_xlabel("# Tasks")
ax1.set_xticks(np.arange(5,max_tasks+1,5))

color = colors[0]
ax1.set_ylabel("Max LUT Size", color=color)
ax1.plot(n_tasks, n_regions, "-", color=color, linewidth=2)
ax1.set_ylim(0,np.max(n_regions)*1.05)

ax2 = ax1.twinx()

color = colors[1]
ax2.set_ylabel("Max Time ($\\mu$s)", color=color)
ax2.plot(n_tasks, elastic_times_fast/1000, "--", color=color, linewidth=2)
ax2.set_ylim(0,np.max(elastic_times_fast/1000)*1.05)

plt.tight_layout()
name = "synthetic_lookup_table"
plt.savefig(f"{name}.eps",bbox_inches='tight')
plt.savefig(f"{name}.png",bbox_inches='tight')