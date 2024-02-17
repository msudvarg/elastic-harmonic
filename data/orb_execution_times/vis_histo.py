import matplotlib.pyplot as plt

import seaborn as sns

# Define a palette
colors = sns.color_palette("deep", 5)


# Use the fivethirtyeight style
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(4.2, 3))

# paths to read
paths = [] 
prefix = "./histo_data/MH03/experiment_1/"
#prefix = "./"

paths.append('us_imu_exe_times_file')
paths.append('us_left_camera_exe_times_file')
paths.append('ms_tracking_exe_times_file')
paths.append('ms_ba_exe_times_file')

# units
units = []
units.append('us')
units.append('us')
units.append('ms')
units.append('ms')

for i in range(len(paths)):
    data = []
    # read data
    with open(prefix + paths[i] + '.txt', 'r') as file:
        for line in file:
            content = line.split(',')
            data.append (float(content[1]))

    # Create histogram
    ax.hist(data, bins=30, density=True, alpha=0.8, edgecolor='black', linewidth=1.5, color=colors[i])
    ax.set_yscale('log')
    ax.set_xlabel('Execution Time (' + units[i] + ')', )
    ax.set_ylabel('Frequency')

    # Display the plot
    plt.tight_layout()

    # Save the figure
    fig.savefig(paths[i] + '.eps', format="eps")
    fig.show()

    # Clear the current figure
    ax.clear()

