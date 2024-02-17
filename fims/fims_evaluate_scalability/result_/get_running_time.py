import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

utilizations = [0.1, 0.2, 0.3, 0.4, 0.5]
components=['/image_process_time.txt', '/inversion_time.txt', '/hk_reading_time.txt']

data = []
for component in components:
    component_data = []
    for utilization in utilizations:
        file = 'run_time_' + str(utilization) + component
        cur_data = np.loadtxt(file)
        data1 = cur_data[1:]
        component_data.append(data1)
    data.append(component_data)


data = []
for component in components:
    component_data = []
    for utilization in utilizations:
        data1 = []
        file = 'run_time_' + str(utilization) + component
        cur_data = np.loadtxt(file)
        data1 = cur_data[1:]
        component_data.append(data1)
    data.append(component_data)


components=['image_process', 'data_inversion', 'hk_reading']
for i in range(len(data)):
    for j in range(len(data[i])):
        average = np.mean(data[i][j])
        print(f'{components[i]} utilization {utilizations[j]} Avg Time: {average}')
        maximun = np.max(data[i][j])
        print(f'{components[i]} utilization {utilizations[j]} Max Time: {maximun}')


# Define a palette
colors = sns.color_palette("deep", 5)

# Use the fivethirtyeight style
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(4.2, 3))

for j in range(len(data)):
    for i, d in enumerate(data[j]):
        ax.hist(d, bins=30, density=True, alpha=0.8, edgecolor='black', linewidth=1.5, color=colors[i])
        ax.set_yscale('log')
        ax.set_xlabel('Execution Time (ms)', )
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        fig.savefig(f'{components[j]}_ut0.{i+1}.png')
        # plt.show()  # Use plt.show() instead of fig.show()
        ax.clear()


